from __future__ import annotations

import argparse
import csv
import html
import shutil
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REVIEW_STATUSES = {
    "NEW_ALIAS_CANDIDATE",
    "AGREEMENT_REVIEW",
    "UNRESOLVED_OCR_TEXT",
    "MANUAL_REVIEW",
    "AMBIGUOUS_ALIAS",
    "CONFLICT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crée une archive compacte des cas de réconciliation à revoir."
    )
    parser.add_argument("--batch", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sniff_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
    except csv.Error:
        return ";"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise RuntimeError(f"Fichier absent : {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=sniff_delimiter(path))
        fields = list(reader.fieldnames or [])
        rows = [
            {key: value or "" for key, value in row.items()}
            for row in reader
        ]

    if not fields:
        raise RuntimeError(f"En-tête CSV absent : {path}")

    return fields, rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def first_value(row: dict[str, str], candidates: tuple[str, ...]) -> str:
    lower = {key.lower(): value for key, value in row.items()}
    for candidate in candidates:
        value = str(lower.get(candidate.lower(), "")).strip()
        if value:
            return value
    return ""


def detect_status(row: dict[str, str]) -> str:
    for column in (
        "reconciliation_status",
        "final_status",
        "status",
        "decision",
        "result",
        "resolution",
    ):
        value = first_value(row, (column,)).upper()
        if value in REVIEW_STATUSES:
            return value

    for value in row.values():
        normalized = str(value or "").strip().upper()
        if normalized in REVIEW_STATUSES:
            return normalized

    review_required = first_value(
        row,
        ("review_required", "needs_review", "manual_review", "requires_review"),
    ).lower()

    if review_required in {"1", "true", "yes", "oui"}:
        return "MANUAL_REVIEW"

    return ""


def normalize_side(value: str) -> str:
    text = value.strip().upper()
    return {
        "LEFT": "L",
        "RIGHT": "R",
        "GAUCHE": "L",
        "DROITE": "R",
    }.get(text, text)


def safe_name(value: str) -> str:
    result = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    return result.strip("._") or "unknown"


def image_files(roots: list[Path]) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    result: list[Path] = []

    for root in roots:
        if not root.exists():
            continue
        result.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )

    return result


def choose_images(
    files: list[Path],
    screenshot_id: str,
    side: str,
    slot: str,
) -> list[Path]:
    sid = screenshot_id.lower()
    candidates = [
        path for path in files if sid and sid in path.name.lower()
    ]

    if not candidates:
        return []

    side = side.lower()
    slot = slot.lower()
    patterns = {
        f"{side}{slot}",
        f"{side}_{slot}",
        f"{side}-{slot}",
        f"{side}_slot_{slot}",
        f"slot_{slot}_{side}",
    }

    strong = [
        path
        for path in candidates
        if any(pattern in path.name.lower() for pattern in patterns)
    ]

    raw = [path for path in candidates if "raw" in {p.name for p in path.parents}]
    selected: list[Path] = []

    for path in [*raw[:2], *(strong[:20] if strong else candidates[:20])]:
        if path not in selected:
            selected.append(path)

    return selected


def unique_copy(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name

    if not destination.exists():
        shutil.copy2(source, destination)
        return destination

    index = 2
    while True:
        candidate = destination_dir / f"{source.stem}_{index}{source.suffix}"
        if not candidate.exists():
            shutil.copy2(source, candidate)
            return candidate
        index += 1


def create_html(
    path: Path,
    rows: list[dict[str, str]],
    row_images: dict[int, list[str]],
) -> None:
    cards: list[str] = []

    for index, row in enumerate(rows, start=1):
        screenshot_id = first_value(
            row, ("screenshot_id", "capture_id", "image_id", "id")
        )
        side = normalize_side(
            first_value(row, ("side", "team_side", "column", "camp"))
        )
        slot = first_value(row, ("slot", "position", "row", "slot_index"))
        status = detect_status(row)

        images = "".join(
            f'<figure><img src="{html.escape(relative)}"><figcaption>'
            f'{html.escape(Path(relative).name)}</figcaption></figure>'
            for relative in row_images.get(index, [])
        )

        metadata = "".join(
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>"
            for key, value in row.items()
            if str(value).strip()
        )

        cards.append(
            f"""
<section class="card">
<h2>{index:03d} — {html.escape(screenshot_id)} {html.escape(side)}{html.escape(slot)}
— {html.escape(status)}</h2>
<div class="images">{images}</div>
<table>{metadata}</table>
</section>
"""
        )

    document = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Revue de réconciliation</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;background:#f4f4f4}}
.card{{background:#fff;margin:20px 0;padding:18px;border-radius:10px;
box-shadow:0 1px 5px rgba(0,0,0,.15)}}
.images{{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}}
figure{{margin:0}}
img{{max-width:900px;max-height:520px;object-fit:contain;border:1px solid #bbb;background:#111}}
figcaption{{max-width:900px;font-size:12px;overflow-wrap:anywhere}}
table{{border-collapse:collapse;margin-top:14px;width:100%}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}}
th{{width:260px;background:#eee}}
</style>
</head>
<body>
<h1>Revue de réconciliation</h1>
<p>Cas sélectionnés : {len(rows)}</p>
{''.join(cards)}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()

    try:
        batch_root = Path("data/batches") / args.batch
        report_dir = batch_root / "reports" / "reconciliation_v2"
        results_path = report_dir / "reconciliation_results.csv"
        aliases_path = report_dir / "alias_candidates.csv"
        existing_html = report_dir / "reconciliation_review.html"
        archive_path = Path(f"{args.batch}_reconciliation_review.zip")

        if archive_path.exists() and not args.overwrite:
            raise RuntimeError(
                f"L'archive existe déjà : {archive_path}. Utilise --overwrite."
            )

        fields, rows = read_csv(results_path)
        selected: list[dict[str, str]] = []

        for row in rows:
            status = detect_status(row)
            if status:
                copy = dict(row)
                copy["_review_status"] = status
                selected.append(copy)

        if not selected:
            raise RuntimeError("Aucun cas de revue trouvé.")

        if "_review_status" not in fields:
            fields.append("_review_status")

        temp_dir = batch_root / "_reconciliation_review_package"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)

        write_csv(temp_dir / "selected_cases.csv", fields, selected)

        source_dir = temp_dir / "source"
        source_dir.mkdir()
        shutil.copy2(results_path, source_dir / results_path.name)

        for candidate in (
            aliases_path,
            existing_html,
            batch_root / "reports" / "visual_matching_dynamic_v1" / "visual_match_results.csv",
            batch_root / "reports" / "ocr_dynamic_v1" / "hero_names_ocr.csv",
            batch_root / "crops_dynamic_v1" / "crop_manifest.csv",
            batch_root / "batch_manifest.csv",
        ):
            if candidate.is_file():
                shutil.copy2(candidate, source_dir / candidate.name)

        roots = [
            batch_root / "raw",
            batch_root / "crops_dynamic_v1",
            batch_root / "reports" / "visual_matching_dynamic_v1",
            batch_root / "reports" / "ocr_dynamic_v1",
            report_dir,
        ]
        indexed = image_files(roots)
        copied = 0
        row_images: dict[int, list[str]] = {}

        for index, row in enumerate(selected, start=1):
            screenshot_id = first_value(
                row, ("screenshot_id", "capture_id", "image_id", "id")
            )
            side = normalize_side(
                first_value(row, ("side", "team_side", "column", "camp"))
            )
            slot = first_value(row, ("slot", "position", "row", "slot_index"))
            case_dir = (
                temp_dir
                / "images"
                / safe_name(f"{index:03d}_{screenshot_id}_{side}{slot}")
            )

            relatives: list[str] = []
            for source in choose_images(indexed, screenshot_id, side, slot):
                destination = unique_copy(source, case_dir)
                relatives.append(destination.relative_to(temp_dir).as_posix())
                copied += 1

            row_images[index] = relatives

        create_html(temp_dir / "review.html", selected, row_images)

        counts = Counter(detect_status(row) for row in selected)
        readme = [
            "REVUE DE RÉCONCILIATION",
            "=" * 72,
            "",
            f"Lot : {args.batch}",
            f"Cas sélectionnés : {len(selected)}",
            "",
            "Statuts :",
        ]
        readme.extend(f"- {status}: {count}" for status, count in sorted(counts.items()))
        readme.extend(
            [
                "",
                "Ouvrir review.html.",
                "selected_cases.csv contient les données complètes.",
                "",
            ]
        )
        (temp_dir / "README.txt").write_text("\n".join(readme), encoding="utf-8")

        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            for path in sorted(temp_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(temp_dir).as_posix())

        shutil.rmtree(temp_dir)

    except (RuntimeError, OSError, csv.Error, ValueError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    print("PRÉPARATION DE LA REVUE DE RÉCONCILIATION")
    print("=" * 72)
    print()
    print(f"Lot : {args.batch}")
    print(f"Cas sélectionnés : {len(selected)}")
    print()
    print("Statuts :")
    for status, count in sorted(counts.items()):
        print(f"- {status}: {count}")
    print()
    print(f"Images indexées : {len(indexed)}")
    print(f"Images copiées  : {copied}")
    print()
    print(f"Archive : {archive_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
