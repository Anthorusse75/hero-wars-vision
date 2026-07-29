from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


BATCH_NAME = "hero_batch_003"

BATCH_ROOT = Path("data/batches") / BATCH_NAME
RAW_DIR = BATCH_ROOT / "raw"
REPORT_DIR = BATCH_ROOT / "reports" / "frame_detection_v1"
MANIFEST = REPORT_DIR / "frame_detection_manifest.csv"

# Les deux captures 2752 × 2064 ont été mal interprétées par la géométrie
# automatique : leurs lignes 5 étaient découpées dans le décor.
CORRECTIONS = {
    "31352_IMG_4444.png": {
        "source_width": "2752",
        "source_height": "2064",
        "status": "REVIEW",
        "left_center_x": "493.0",
        "right_center_x": "2257.0",
        "row_1_center_y": "327.0",
        "row_2_center_y": "588.0",
        "row_3_center_y": "850.0",
        "row_4_center_y": "1112.0",
        "row_5_center_y": "1373.0",
        "frame_width": "232.0",
        "frame_height": "232.0",
        "row_step": "261.5",
        "mean_alignment_error": "0.0",
        "left_support": "5",
        "right_support": "5",
        "candidate_count": "10",
        "error": "",
    },
    "35430_IMG_4683.png": {
        "source_width": "2752",
        "source_height": "2064",
        "status": "REVIEW",
        "left_center_x": "493.0",
        "right_center_x": "2257.0",
        "row_1_center_y": "327.0",
        "row_2_center_y": "588.0",
        "row_3_center_y": "850.0",
        "row_4_center_y": "1112.0",
        "row_5_center_y": "1373.0",
        "frame_width": "232.0",
        "frame_height": "232.0",
        "row_step": "261.5",
        "mean_alignment_error": "0.0",
        "left_support": "5",
        "right_support": "5",
        "candidate_count": "10",
        "error": "",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Corrige la géométrie des deux captures carrées du batch 003 "
            "dont la cinquième ligne était mal découpée."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Applique réellement les corrections au manifeste.",
    )
    return parser.parse_args()


def read_manifest() -> tuple[list[str], list[dict[str, str]]]:
    if not MANIFEST.exists():
        raise RuntimeError(f"Manifeste absent : {MANIFEST}")

    with MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream, delimiter=";")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError("Le manifeste ne possède pas d'en-tête.")

    return fieldnames, rows


def write_manifest(
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    temporary = MANIFEST.with_suffix(".csv.tmp")

    with temporary.open(
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

    temporary.replace(MANIFEST)


def create_preview(
    source_filename: str,
    values: dict[str, str],
) -> Path:
    source_path = RAW_DIR / source_filename

    with Image.open(source_path) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)

    frame_width = float(values["frame_width"])
    frame_height = float(values["frame_height"])
    left_x = float(values["left_center_x"])
    right_x = float(values["right_center_x"])

    row_centers = [
        float(values[f"row_{index}_center_y"])
        for index in range(1, 6)
    ]

    for side, center_x in (("L", left_x), ("R", right_x)):
        for slot, center_y in enumerate(row_centers, start=1):
            box = (
                round(center_x - frame_width / 2),
                round(center_y - frame_height / 2),
                round(center_x + frame_width / 2),
                round(center_y + frame_height / 2),
            )

            draw.rectangle(
                box,
                outline="lime",
                width=5,
            )
            draw.text(
                (box[0] + 5, box[1] + 5),
                f"{side}{slot}",
                fill="yellow",
                stroke_width=2,
                stroke_fill="black",
            )

    destination = (
        REPORT_DIR
        / f"{Path(source_filename).stem}__manual_geometry.jpg"
    )

    image.save(
        destination,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    return destination


def main() -> int:
    args = parse_args()

    try:
        fieldnames, rows = read_manifest()

        rows_by_filename = {
            row.get("source_filename", ""): row
            for row in rows
        }

        for source_filename, values in CORRECTIONS.items():
            source_path = RAW_DIR / source_filename

            if not source_path.exists():
                raise RuntimeError(f"Capture absente : {source_path}")

            with Image.open(source_path) as image:
                width, height = image.size

            expected = (
                int(values["source_width"]),
                int(values["source_height"]),
            )

            if (width, height) != expected:
                raise RuntimeError(
                    f"Dimensions inattendues pour {source_filename} : "
                    f"{width} × {height}, attendu "
                    f"{expected[0]} × {expected[1]}."
                )

            if source_filename not in rows_by_filename:
                raise RuntimeError(
                    f"Ligne absente du manifeste : {source_filename}"
                )

            missing_fields = [
                field
                for field in values
                if field not in fieldnames
            ]

            if missing_fields:
                raise RuntimeError(
                    "Colonnes absentes du manifeste : "
                    + ", ".join(missing_fields)
                )

    except (RuntimeError, OSError, csv.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    print("CORRECTION DES GÉOMÉTRIES CARRÉES — HERO_BATCH_003")
    print("=" * 72)
    print()

    for source_filename, values in CORRECTIONS.items():
        old_row = rows_by_filename[source_filename]

        print(source_filename)
        print(f"- Ancien statut : {old_row.get('status', '')}")
        print(
            "- Nouveaux centres Y : "
            + ", ".join(
                values[f"row_{index}_center_y"]
                for index in range(1, 6)
            )
        )
        print(
            f"- Centres X : {values['left_center_x']} / "
            f"{values['right_center_x']}"
        )
        print(
            f"- Cadre : {values['frame_width']} × "
            f"{values['frame_height']}"
        )
        print()

    if not args.apply:
        print("MODE SIMULATION : aucun fichier n'a été modifié.")
        print()
        print("Pour appliquer :")
        print(
            "python scripts/patch_batch_003_square_geometry.py --apply"
        )
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = MANIFEST.with_name(
        f"frame_detection_manifest_before_square_fix_{timestamp}.csv"
    )
    shutil.copy2(MANIFEST, backup)

    preview_paths: list[Path] = []

    for source_filename, values in CORRECTIONS.items():
        preview_path = create_preview(source_filename, values)
        values = dict(values)
        values["debug_file"] = preview_path.as_posix()
        rows_by_filename[source_filename].update(values)
        preview_paths.append(preview_path)

    write_manifest(fieldnames, rows)

    print("Corrections appliquées.")
    print(f"Sauvegarde : {backup}")
    print(f"Manifeste : {MANIFEST}")

    for preview_path in preview_paths:
        print(f"Aperçu : {preview_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
