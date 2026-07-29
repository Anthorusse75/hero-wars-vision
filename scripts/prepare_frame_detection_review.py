from __future__ import annotations

import argparse
import csv
import html
import shutil
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PRIORITY_METHODS = {
    "layout_prior_fallback",
    "contour_prior_completed",
    "contour_prior_guard",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prépare un paquet compact pour contrôler les détections "
            "géométriques prioritaires d'un batch."
        )
    )
    parser.add_argument(
        "--batch",
        required=True,
        help="Nom du lot, par exemple hero_batch_004.",
    )
    parser.add_argument(
        "--extra-review",
        type=int,
        default=30,
        help=(
            "Nombre de cas REVIEW issus de la méthode contour à ajouter "
            "en plus des méthodes prioritaires. Défaut : 30."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remplace un paquet déjà existant.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise RuntimeError(f"Manifeste absent : {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream, delimiter=";")
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not fields:
        raise RuntimeError("Le manifeste ne possède pas d'en-tête.")

    return fields, rows


def as_float(value: str) -> float:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return 999.0


def as_int(value: str) -> int:
    try:
        return int(float(str(value or "").strip()))
    except ValueError:
        return 0


def method_field(fieldnames: list[str]) -> str:
    for candidate in (
        "method",
        "detection_method",
        "geometry_method",
    ):
        if candidate in fieldnames:
            return candidate

    raise RuntimeError(
        "Aucune colonne de méthode trouvée dans le manifeste."
    )


def resolve_debug_path(
    batch_root: Path,
    row: dict[str, str],
) -> Path:
    raw = str(row.get("debug_file") or "").strip()

    if raw:
        path = Path(raw)

        if path.exists():
            return path

        candidate = Path.cwd() / path

        if candidate.exists():
            return candidate

    source_stem = Path(
        str(row.get("source_filename") or "")
    ).stem

    fallback = (
        batch_root
        / "reports"
        / "frame_detection_v1"
        / "debug"
        / f"{source_stem}__frame_detection.jpg"
    )

    return fallback


def risk_key(row: dict[str, str]) -> tuple[int, float, int]:
    strongest_support = max(
        as_int(row.get("left_support", "")),
        as_int(row.get("right_support", "")),
    )

    alignment_error = as_float(
        row.get("mean_alignment_error", "")
    )

    candidate_count = as_int(
        row.get("candidate_count", "")
    )

    return (
        strongest_support,
        -alignment_error,
        candidate_count,
    )


def select_rows(
    rows: list[dict[str, str]],
    method_column: str,
    extra_review: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    selected_names: set[str] = set()

    def add(row: dict[str, str]) -> None:
        source = str(row.get("source_filename") or "")

        if source in selected_names:
            return

        selected.append(row)
        selected_names.add(source)

    for row in rows:
        status = str(row.get("status") or "").strip().upper()
        method = str(row.get(method_column) or "").strip()

        if status == "FAIL" or method in PRIORITY_METHODS:
            add(row)

    contour_reviews = [
        row
        for row in rows
        if (
            str(row.get("status") or "").strip().upper() == "REVIEW"
            and str(row.get(method_column) or "").strip() == "contour"
            and str(row.get("source_filename") or "") not in selected_names
        )
    ]

    contour_reviews.sort(key=risk_key)

    for row in contour_reviews[:max(0, extra_review)]:
        add(row)

    selected.sort(
        key=lambda row: (
            str(row.get("status") or "").strip().upper() != "FAIL",
            str(row.get(method_column) or ""),
            risk_key(row),
            str(row.get("source_filename") or ""),
        )
    )

    return selected


def copy_debug_images(
    batch_root: Path,
    rows: list[dict[str, str]],
    destination: Path,
) -> dict[str, str]:
    images_dir = destination / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, str] = {}

    for index, row in enumerate(rows, start=1):
        source_filename = str(row.get("source_filename") or "")
        debug_path = resolve_debug_path(batch_root, row)

        if not debug_path.exists():
            copied[source_filename] = ""
            continue

        destination_name = (
            f"{index:03d}__{debug_path.name}"
        )

        shutil.copy2(
            debug_path,
            images_dir / destination_name,
        )

        copied[source_filename] = (
            Path("images") / destination_name
        ).as_posix()

    return copied


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def create_html(
    path: Path,
    rows: list[dict[str, str]],
    method_column: str,
    image_paths: dict[str, str],
) -> None:
    cards: list[str] = []

    for row in rows:
        source_filename = str(row.get("source_filename") or "")
        image_path = image_paths.get(source_filename, "")
        status = str(row.get("status") or "")
        method = str(row.get(method_column) or "")
        error = str(row.get("mean_alignment_error") or "")
        left_support = str(row.get("left_support") or "")
        right_support = str(row.get("right_support") or "")
        candidates = str(row.get("candidate_count") or "")
        message = str(row.get("error") or "")

        image_html = (
            f'<img src="{html.escape(image_path)}" '
            f'alt="{html.escape(source_filename)}">'
            if image_path
            else "<p><strong>Image de contrôle absente.</strong></p>"
        )

        cards.append(
            f"""
            <article class="card">
                <h2>{html.escape(source_filename)}</h2>
                {image_html}
                <table>
                    <tr><th>Statut</th><td>{html.escape(status)}</td></tr>
                    <tr><th>Méthode</th><td>{html.escape(method)}</td></tr>
                    <tr><th>Support L/R</th><td>{html.escape(left_support)} / {html.escape(right_support)}</td></tr>
                    <tr><th>Erreur moyenne</th><td>{html.escape(error)}</td></tr>
                    <tr><th>Candidats</th><td>{html.escape(candidates)}</td></tr>
                    <tr><th>Erreur</th><td>{html.escape(message)}</td></tr>
                </table>
            </article>
            """
        )

    document = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <title>Contrôle géométrique prioritaire</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #eeeeee;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
            gap: 18px;
        }}
        .card {{
            background: white;
            border: 1px solid #bbbbbb;
            border-radius: 8px;
            padding: 12px;
        }}
        img {{
            width: 100%;
            height: auto;
            display: block;
            background: #222222;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            border: 1px solid #cccccc;
            padding: 5px 7px;
            text-align: left;
        }}
        th {{
            width: 150px;
            background: #eeeeee;
        }}
    </style>
</head>
<body>
    <h1>Contrôle géométrique prioritaire</h1>
    <p>Cas inclus : {len(rows)}</p>
    <main class="grid">
        {''.join(cards)}
    </main>
</body>
</html>
"""

    path.write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()

    batch_root = Path("data/batches") / args.batch
    report_root = (
        batch_root
        / "reports"
        / "frame_detection_v1"
    )
    manifest_path = (
        report_root
        / "frame_detection_manifest.csv"
    )
    package_root = (
        report_root
        / "priority_review_package"
    )
    archive_path = Path(
        f"{args.batch}_frame_detection_review.zip"
    )

    if package_root.exists():
        if not args.overwrite:
            print(
                f"Le paquet existe déjà : {package_root}",
                file=sys.stderr,
            )
            return 1

        shutil.rmtree(package_root)

    if archive_path.exists():
        if not args.overwrite:
            print(
                f"L'archive existe déjà : {archive_path}",
                file=sys.stderr,
            )
            return 1

        archive_path.unlink()

    try:
        fieldnames, rows = read_manifest(manifest_path)
        method_column = method_field(fieldnames)
        selected = select_rows(
            rows,
            method_column,
            args.extra_review,
        )
    except (RuntimeError, csv.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    package_root.mkdir(parents=True, exist_ok=True)

    image_paths = copy_debug_images(
        batch_root,
        selected,
        package_root,
    )

    output_rows: list[dict[str, str]] = []

    for row in selected:
        output_row = dict(row)
        output_row["review_image"] = image_paths.get(
            str(row.get("source_filename") or ""),
            "",
        )
        output_rows.append(output_row)

    output_fields = list(fieldnames)

    if "review_image" not in output_fields:
        output_fields.append("review_image")

    write_csv(
        package_root / "selected_cases.csv",
        output_fields,
        output_rows,
    )

    create_html(
        package_root / "review.html",
        selected,
        method_column,
        image_paths,
    )

    status_counts = Counter(
        str(row.get("status") or "")
        for row in selected
    )
    method_counts = Counter(
        str(row.get(method_column) or "")
        for row in selected
    )

    summary_lines = [
        f"CONTRÔLE GÉOMÉTRIQUE PRIORITAIRE — {args.batch}",
        "=" * 72,
        "",
        f"Cas sélectionnés : {len(selected)}",
        "",
        "Statuts :",
    ]

    for status, count in sorted(status_counts.items()):
        summary_lines.append(f"- {status}: {count}")

    summary_lines.extend(["", "Méthodes :"])

    for method, count in sorted(
        method_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        summary_lines.append(f"- {method}: {count}")

    summary_lines.extend(
        [
            "",
            f"REVIEW contour supplémentaires : {args.extra_review}",
            "",
            "Contenu :",
            "- review.html",
            "- selected_cases.csv",
            "- images/",
            "",
        ]
    )

    summary = "\n".join(summary_lines)
    (package_root / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    with ZipFile(
        archive_path,
        "w",
        ZIP_DEFLATED,
    ) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    path.relative_to(package_root),
                )

    print(summary)
    print(f"Archive : {archive_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
