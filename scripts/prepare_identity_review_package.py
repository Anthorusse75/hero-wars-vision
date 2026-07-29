from __future__ import annotations

import argparse
import base64
import csv
import html
import re
import shutil
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, UnidentifiedImageError


BATCH_PATTERN = re.compile(r"^hero_batch_\d{3}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prépare une archive compacte avec uniquement les cas d'identité "
            "restant à revoir après la réconciliation V2."
        )
    )
    parser.add_argument(
        "--batch",
        required=True,
        help="Nom du lot, par exemple hero_batch_003.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remplace un paquet déjà existant.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise RuntimeError(f"Fichier absent : {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream, delimiter=";")
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not fields:
        raise RuntimeError(f"En-tête CSV absent : {path}")

    return fields, rows


def write_csv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def review_required(row: dict[str, str]) -> bool:
    explicit = str(row.get("review_required") or "").strip()

    if explicit:
        return explicit in {"1", "true", "TRUE", "yes", "YES"}

    decision = str(row.get("decision") or "").strip().upper()

    return decision in {
        "UNRESOLVED_OCR_TEXT",
        "MANUAL_REVIEW",
        "AMBIGUOUS_ALIAS",
        "CONFLICT",
        "AGREEMENT_REVIEW",
        "NEW_ALIAS_CANDIDATE",
    }


def first_existing(
    candidates: list[Path],
) -> Path | None:
    for path in candidates:
        if path.exists():
            return path

    return None


def find_crop(
    batch_root: Path,
    row: dict[str, str],
    crop_type: str,
) -> Path | None:
    screenshot_id = str(row.get("screenshot_id") or "")
    side = str(row.get("side") or "")
    slot = str(row.get("slot") or "")

    avatar_file = str(row.get("avatar_file") or "").strip()
    name_file = str(row.get("name_file") or "").strip()

    directory = (
        batch_root
        / "crops_dynamic_v1"
        / crop_type
    )

    exact_name = (
        avatar_file
        if crop_type == "avatars_inner"
        else name_file
    )

    candidates: list[Path] = []

    if exact_name:
        candidates.append(directory / Path(exact_name).name)

    pattern = f"{screenshot_id}_*__{side}{slot}.png"
    candidates.extend(sorted(directory.glob(pattern)))

    return first_existing(candidates)


def find_raw_screenshot(
    batch_root: Path,
    screenshot_id: str,
) -> Path | None:
    matches = sorted(
        (batch_root / "raw").glob(
            f"{screenshot_id}_*"
        )
    )

    return matches[0] if matches else None


def image_uri(
    path: Path,
    max_size: tuple[int, int],
) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail(
            max_size,
            Image.Resampling.LANCZOS,
        )

        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=90,
        )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


def create_html(
    path: Path,
    rows: list[dict[str, str]],
    batch_root: Path,
) -> None:
    parts = [
        "<!DOCTYPE html>",
        '<html lang="fr">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Revue identités batch</title>",
        """
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .case { border: 1px solid #aaa; margin-bottom: 24px; padding: 14px; }
        .images { display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap; }
        img.avatar { width: 150px; image-rendering: auto; }
        img.name { max-width: 440px; }
        img.full { max-width: 900px; }
        table { border-collapse: collapse; margin-top: 12px; }
        th, td { border: 1px solid #bbb; padding: 5px 8px; text-align: left; }
        th { background: #eee; }
        code { background: #f1f1f1; padding: 2px 4px; }
        </style>
        """,
        "</head>",
        "<body>",
        f"<h1>Cas restant à revoir : {len(rows)}</h1>",
    ]

    for index, row in enumerate(rows, start=1):
        screenshot_id = str(row.get("screenshot_id") or "")
        side = str(row.get("side") or "")
        slot = str(row.get("slot") or "")

        avatar_path = find_crop(
            batch_root,
            row,
            "avatars_inner",
        )
        name_path = find_crop(
            batch_root,
            row,
            "names",
        )
        raw_path = find_raw_screenshot(
            batch_root,
            screenshot_id,
        )

        parts.append(
            '<section class="case">'
        )
        parts.append(
            f"<h2>Cas {index} — "
            f"{html.escape(screenshot_id)} "
            f"{html.escape(side)}{html.escape(slot)}</h2>"
        )
        parts.append('<div class="images">')

        if avatar_path:
            parts.append(
                f'<div><strong>Avatar</strong><br>'
                f'<img class="avatar" src="{image_uri(avatar_path, (300, 300))}"></div>'
            )

        if name_path:
            parts.append(
                f'<div><strong>Nom OCR</strong><br>'
                f'<img class="name" src="{image_uri(name_path, (700, 150))}"></div>'
            )

        parts.append("</div>")

        display_fields = [
            "decision",
            "visual_status",
            "predicted_hero_name",
            "predicted_hero_uid",
            "visual_similarity",
            "visual_margin",
            "ocr_text",
            "ocr_confidence",
            "ocr_status",
            "ocr_matched_hero_name",
            "ocr_match_method",
            "final_hero_name",
            "review_required",
        ]

        parts.append("<table>")

        for field in display_fields:
            if field in row:
                parts.append(
                    "<tr>"
                    f"<th>{html.escape(field)}</th>"
                    f"<td>{html.escape(str(row.get(field) or ''))}</td>"
                    "</tr>"
                )

        parts.append("</table>")

        if raw_path:
            parts.append(
                "<details>"
                "<summary>Afficher la capture complète</summary>"
                f'<img class="full" src="{image_uri(raw_path, (1400, 900))}">'
                "</details>"
            )

        parts.append("</section>")

    parts.extend(
        [
            "</body>",
            "</html>",
        ]
    )

    path.write_text(
        "\n".join(parts),
        encoding="utf-8",
    )


def copy_assets(
    rows: list[dict[str, str]],
    batch_root: Path,
    package_root: Path,
) -> None:
    avatar_dir = package_root / "avatars"
    names_dir = package_root / "names"
    raw_dir = package_root / "screenshots"

    avatar_dir.mkdir(parents=True, exist_ok=True)
    names_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    copied_raw: set[str] = set()

    for row in rows:
        avatar_path = find_crop(
            batch_root,
            row,
            "avatars_inner",
        )
        name_path = find_crop(
            batch_root,
            row,
            "names",
        )

        if avatar_path:
            shutil.copy2(
                avatar_path,
                avatar_dir / avatar_path.name,
            )

        if name_path:
            shutil.copy2(
                name_path,
                names_dir / name_path.name,
            )

        screenshot_id = str(row.get("screenshot_id") or "")

        if screenshot_id in copied_raw:
            continue

        raw_path = find_raw_screenshot(
            batch_root,
            screenshot_id,
        )

        if raw_path:
            shutil.copy2(
                raw_path,
                raw_dir / raw_path.name,
            )
            copied_raw.add(screenshot_id)


def main() -> int:
    args = parse_args()

    if not BATCH_PATTERN.fullmatch(args.batch):
        print(
            "--batch doit avoir la forme hero_batch_003.",
            file=sys.stderr,
        )
        return 2

    batch_root = (
        Path("data/batches")
        / args.batch
    )

    source_csv = (
        batch_root
        / "reports"
        / "reconciliation_v2"
        / "reconciliation_results.csv"
    )

    package_root = (
        batch_root
        / "reports"
        / "identity_review_package"
    )

    zip_path = Path(
        f"{args.batch}_identity_review.zip"
    )

    if package_root.exists():
        if not args.overwrite:
            print(
                f"Le paquet existe déjà : {package_root}",
                file=sys.stderr,
            )
            return 1

        shutil.rmtree(package_root)

    if zip_path.exists():
        if not args.overwrite:
            print(
                f"L'archive existe déjà : {zip_path}",
                file=sys.stderr,
            )
            return 1

        zip_path.unlink()

    try:
        fields, rows = read_csv(source_csv)
    except (RuntimeError, csv.Error) as error:
        print(
            f"Erreur : {error}",
            file=sys.stderr,
        )
        return 1

    review_rows = [
        row
        for row in rows
        if review_required(row)
    ]

    review_rows.sort(
        key=lambda row: (
            int(str(row.get("screenshot_id") or 0)),
            str(row.get("side") or ""),
            int(str(row.get("slot") or 0)),
        )
    )

    package_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        package_root / "review_cases.csv",
        fields,
        review_rows,
    )

    create_html(
        package_root / "review.html",
        review_rows,
        batch_root,
    )

    copy_assets(
        review_rows,
        batch_root,
        package_root,
    )

    summary = "\n".join(
        [
            f"REVUE DES IDENTITÉS — {args.batch}",
            "=" * 72,
            "",
            f"Cas restant à revoir : {len(review_rows)}",
            f"Captures concernées : {len({row.get('screenshot_id', '') for row in review_rows})}",
            "",
            "Contenu :",
            "- review_cases.csv",
            "- review.html",
            "- avatars/",
            "- names/",
            "- screenshots/",
            "",
        ]
    )

    (package_root / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    with ZipFile(
        zip_path,
        "w",
        ZIP_DEFLATED,
    ) as archive:
        for path in sorted(
            package_root.rglob("*")
        ):
            if path.is_file():
                archive.write(
                    path,
                    path.relative_to(package_root),
                )

    print(summary)
    print(
        f"Archive : {zip_path.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
