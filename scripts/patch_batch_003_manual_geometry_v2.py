from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


BATCH_NAME = "hero_batch_003"
SOURCE_FILENAME = "24753_BE6E1C28-524F-4858-AB24-3C262B61585F.png"

BATCH_ROOT = Path("data/batches") / BATCH_NAME
RAW_IMAGE = BATCH_ROOT / "raw" / SOURCE_FILENAME
REPORT_DIR = BATCH_ROOT / "reports" / "frame_detection_v1"
MANIFEST = REPORT_DIR / "frame_detection_manifest.csv"
DEBUG_PREVIEW = REPORT_DIR / "24753_manual_geometry_preview.jpg"

REFERENCE_WIDTH = 2048.0
REFERENCE_HEIGHT = 944.0

REFERENCE_GEOMETRY = {
    "left_center_x": 487.5,
    "right_center_x": 1562.5,
    "row_1_center_y": 203.5,
    "row_2_center_y": 363.0,
    "row_3_center_y": 518.5,
    "row_4_center_y": 679.0,
    "row_5_center_y": 837.5,
    "frame_width": 142.0,
    "frame_height": 151.0,
    "row_step": 158.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Corrige la géométrie de la capture 24753 du batch 003 "
            "avec mise à l'échelle automatique."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Applique réellement la correction.",
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


def scaled_geometry(
    width: int,
    height: int,
) -> dict[str, str]:
    scale_x = width / REFERENCE_WIDTH
    scale_y = height / REFERENCE_HEIGHT

    reference_ratio = REFERENCE_WIDTH / REFERENCE_HEIGHT
    actual_ratio = width / height
    ratio_error = abs(actual_ratio - reference_ratio) / reference_ratio

    if ratio_error > 0.01:
        raise RuntimeError(
            "Le ratio d'image diffère trop de la référence : "
            f"{width} × {height}."
        )

    return {
        "source_width": str(width),
        "source_height": str(height),
        "status": "REVIEW",
        "left_center_x": f"{REFERENCE_GEOMETRY['left_center_x'] * scale_x:.3f}",
        "right_center_x": f"{REFERENCE_GEOMETRY['right_center_x'] * scale_x:.3f}",
        "row_1_center_y": f"{REFERENCE_GEOMETRY['row_1_center_y'] * scale_y:.3f}",
        "row_2_center_y": f"{REFERENCE_GEOMETRY['row_2_center_y'] * scale_y:.3f}",
        "row_3_center_y": f"{REFERENCE_GEOMETRY['row_3_center_y'] * scale_y:.3f}",
        "row_4_center_y": f"{REFERENCE_GEOMETRY['row_4_center_y'] * scale_y:.3f}",
        "row_5_center_y": f"{REFERENCE_GEOMETRY['row_5_center_y'] * scale_y:.3f}",
        "frame_width": f"{REFERENCE_GEOMETRY['frame_width'] * scale_x:.3f}",
        "frame_height": f"{REFERENCE_GEOMETRY['frame_height'] * scale_y:.3f}",
        "row_step": f"{REFERENCE_GEOMETRY['row_step'] * scale_y:.3f}",
        "mean_alignment_error": "0.0",
        "left_support": "5",
        "right_support": "5",
        "candidate_count": "10",
        "debug_file": DEBUG_PREVIEW.as_posix(),
        "error": "",
    }


def create_preview(values: dict[str, str]) -> None:
    with Image.open(RAW_IMAGE) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)

    left_x = float(values["left_center_x"])
    right_x = float(values["right_center_x"])
    frame_width = float(values["frame_width"])
    frame_height = float(values["frame_height"])

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
                width=max(3, round(image.width / 700)),
            )
            draw.text(
                (box[0] + 5, box[1] + 5),
                f"{side}{slot}",
                fill="yellow",
                stroke_width=2,
                stroke_fill="black",
            )

    DEBUG_PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        DEBUG_PREVIEW,
        format="JPEG",
        quality=92,
        optimize=True,
    )


def main() -> int:
    args = parse_args()

    if not RAW_IMAGE.exists():
        print(f"Capture absente : {RAW_IMAGE}", file=sys.stderr)
        return 1

    try:
        with Image.open(RAW_IMAGE) as image:
            width, height = image.size

        values = scaled_geometry(width, height)
        fieldnames, rows = read_manifest()

        matches = [
            row
            for row in rows
            if row.get("source_filename") == SOURCE_FILENAME
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Nombre de lignes correspondantes : {len(matches)}, attendu 1."
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

        row = matches[0]
        old_status = row.get("status", "")
        old_error = row.get("error", "")

    except (RuntimeError, OSError, csv.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    print("CORRECTION MANUELLE V2 — HERO_BATCH_003")
    print("=" * 72)
    print(f"Capture : {SOURCE_FILENAME}")
    print(f"Dimensions réelles : {width} × {height}")
    print(f"Ancien statut : {old_status}")
    print(f"Ancienne erreur : {old_error}")
    print()
    print("Géométrie mise à l'échelle :")
    print(
        f"- Centres X : gauche={values['left_center_x']} | "
        f"droite={values['right_center_x']}"
    )
    print(
        "- Centres Y : "
        + ", ".join(
            values[f"row_{index}_center_y"]
            for index in range(1, 6)
        )
    )
    print(
        f"- Cadre : {values['frame_width']} × {values['frame_height']}"
    )
    print("- Supports : 5 à gauche, 5 à droite")
    print()

    if not args.apply:
        print("MODE SIMULATION : aucun fichier n'a été modifié.")
        print()
        print("Pour appliquer :")
        print(
            "python scripts/patch_batch_003_manual_geometry_v2.py --apply"
        )
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = MANIFEST.with_name(
        f"frame_detection_manifest_before_manual_{timestamp}.csv"
    )
    shutil.copy2(MANIFEST, backup)

    row.update(values)
    write_manifest(fieldnames, rows)
    create_preview(values)

    print("Correction appliquée.")
    print(f"Sauvegarde : {backup}")
    print(f"Manifeste : {MANIFEST}")
    print(f"Aperçu : {DEBUG_PREVIEW}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
