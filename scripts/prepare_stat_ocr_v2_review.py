from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, UnidentifiedImageError


BATCH_PATTERN = re.compile(r"^hero_batch_\d{3}$")
METRICS = (
    "power",
    "damage_dealt",
    "damage_taken",
    "healing",
)

POWER_MIN = 1_000
POWER_MAX = 500_000

PAGE_SIZE = 48
COLUMNS = 4
TILE_WIDTH = 430
TILE_HEIGHT = 132
MARGIN = 14
HIGH_SAMPLE_PER_METRIC = 25
RANDOM_SEED = 20260729


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prépare un paquet de contrôle visuel pour l'OCR "
            "des statistiques V2."
        )
    )
    parser.add_argument(
        "--batch",
        required=True,
        help="Nom du lot, par exemple hero_batch_002.",
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
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(f"En-tête CSV absent : {path}")

    return fieldnames, rows


def write_csv(
    path: Path,
    fieldnames: list[str],
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
            fieldnames=fieldnames,
            delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def to_int(value: str) -> int | None:
    text = str(value or "").strip()

    if not text.isdigit():
        return None

    return int(text)


def classify_change(row: dict[str, str]) -> str:
    old_text = str(row.get("old_value") or "")
    new_text = str(row.get("value") or "")
    metric = str(row.get("metric") or "")

    old_value = to_int(old_text)
    new_value = to_int(new_text)

    if (
        metric == "power"
        and old_value is not None
        and new_value is not None
        and not (POWER_MIN <= old_value <= POWER_MAX)
        and POWER_MIN <= new_value <= POWER_MAX
    ):
        return "POWER_RANGE_FIX"

    if old_text and new_text:
        difference = abs(len(old_text) - len(new_text))

        if difference <= 2 and (
            old_text.startswith(new_text)
            or old_text.endswith(new_text)
            or new_text.startswith(old_text)
            or new_text.endswith(old_text)
        ):
            return "PREFIX_SUFFIX_FIX"

    if old_text == new_text:
        return "UNCHANGED"

    return "COMPLEX_CHANGE"


def attempts_summary(row: dict[str, str]) -> str:
    raw = str(row.get("candidate_attempts") or "").strip()

    if not raw:
        return ""

    try:
        attempts = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:120]

    parts: list[str] = []

    for attempt in attempts:
        digits = str(attempt.get("digits") or "")
        confidence = float(attempt.get("confidence") or 0.0)
        variant = str(attempt.get("variant") or "")

        parts.append(
            f"{variant}:{digits or '-'}@{confidence:.2f}"
        )

    return " | ".join(parts)


def load_crop(
    crop_root: Path,
    crop_file: str,
) -> Image.Image:
    path = crop_root / crop_file

    with Image.open(path) as source:
        image = source.convert("RGB")

    image.thumbnail(
        (TILE_WIDTH - 18, 62),
        Image.Resampling.LANCZOS,
    )

    return image


def create_sheets(
    output_dir: Path,
    crop_root: Path,
    rows: list[dict[str, str]],
    prefix: str,
) -> list[Path]:
    if not rows:
        return []

    output_paths: list[Path] = []

    for start in range(0, len(rows), PAGE_SIZE):
        page_rows = rows[start:start + PAGE_SIZE]
        row_count = math.ceil(len(page_rows) / COLUMNS)

        canvas = Image.new(
            "RGB",
            (
                COLUMNS * TILE_WIDTH + 2 * MARGIN,
                row_count * TILE_HEIGHT + 2 * MARGIN + 34,
            ),
            "white",
        )
        draw = ImageDraw.Draw(canvas)

        page_number = start // PAGE_SIZE + 1

        draw.text(
            (MARGIN, MARGIN),
            f"{prefix} — planche {page_number}",
            fill="black",
        )

        for index, row in enumerate(page_rows):
            column = index % COLUMNS
            grid_row = index // COLUMNS

            x = MARGIN + column * TILE_WIDTH
            y = MARGIN + 34 + grid_row * TILE_HEIGHT

            try:
                crop = load_crop(
                    crop_root,
                    str(row.get("crop_file") or ""),
                )
                canvas.paste(
                    crop,
                    (
                        x + (TILE_WIDTH - crop.width) // 2,
                        y,
                    ),
                )
            except (
                OSError,
                UnidentifiedImageError,
                KeyError,
            ):
                draw.text(
                    (x + 4, y + 10),
                    "IMAGE ABSENTE",
                    fill="red",
                )

            line_1 = (
                f"{row.get('screenshot_id', '')} "
                f"{row.get('side', '')}{row.get('slot', '')} | "
                f"{row.get('metric', '')} | "
                f"{row.get('hero_name', '')}"
            )

            line_2 = (
                f"V1={row.get('old_value', '')} → "
                f"V2={row.get('value', '')} | "
                f"{row.get('status', '')} | "
                f"conf={float(row.get('confidence') or 0):.3f} | "
                f"votes={row.get('votes', '')}"
            )

            line_3 = (
                f"{row.get('change_type', '')} | "
                f"{attempts_summary(row)}"
            )

            draw.text(
                (x + 4, y + 67),
                line_1,
                fill="black",
            )
            draw.text(
                (x + 4, y + 84),
                line_2,
                fill="black",
            )
            draw.text(
                (x + 4, y + 101),
                line_3[:120],
                fill="black",
            )

        output_path = (
            output_dir
            / f"{prefix.lower().replace(' ', '_')}_{page_number:02d}.jpg"
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        canvas.save(
            output_path,
            format="JPEG",
            quality=92,
            optimize=True,
        )
        output_paths.append(output_path)

    return output_paths


def main() -> int:
    args = parse_args()

    if not BATCH_PATTERN.fullmatch(args.batch):
        print(
            "--batch doit avoir la forme hero_batch_002.",
            file=sys.stderr,
        )
        return 2

    batch_root = Path("data/batches") / args.batch
    report_root = batch_root / "reports" / "stat_ocr_v2"
    crop_root = batch_root / "stat_crops_v2"

    detailed_csv = report_root / "stat_values_ocr.csv"
    changed_csv = report_root / "changed_values.csv"
    combined_csv = report_root / "battle_stats_ocr.csv"

    package_dir = (
        batch_root
        / "reports"
        / "stat_ocr_v2_review_package"
    )
    zip_path = Path(
        f"{args.batch}_stat_ocr_v2_review.zip"
    )

    if package_dir.exists():
        if not args.overwrite:
            print(
                f"Le paquet existe déjà : {package_dir}",
                file=sys.stderr,
            )
            return 1

        shutil.rmtree(package_dir)

    if zip_path.exists():
        if not args.overwrite:
            print(
                f"L'archive existe déjà : {zip_path}",
                file=sys.stderr,
            )
            return 1

        zip_path.unlink()

    try:
        fieldnames, all_rows = read_csv(detailed_csv)
        _, changed_rows = read_csv(changed_csv)
    except (RuntimeError, csv.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    for row in changed_rows:
        row["change_type"] = classify_change(row)

    review_rows = [
        dict(row)
        for row in all_rows
        if row.get("status") in {
            "MEDIUM",
            "LOW",
            "NO_DETECTION",
            "ERROR",
        }
    ]

    for row in review_rows:
        row["change_type"] = (
            classify_change(row)
            if row.get("changed_from_v1") == "1"
            else "NON_HIGH_UNCHANGED"
        )

    rng = random.Random(RANDOM_SEED)
    high_sample: list[dict[str, str]] = []

    for metric in METRICS:
        candidates = [
            dict(row)
            for row in all_rows
            if row.get("metric") == metric
            and row.get("status") == "HIGH"
            and row.get("changed_from_v1") == "0"
        ]

        rng.shuffle(candidates)

        for row in candidates[:HIGH_SAMPLE_PER_METRIC]:
            row["change_type"] = "UNCHANGED_HIGH_SAMPLE"

        high_sample.extend(
            candidates[:HIGH_SAMPLE_PER_METRIC]
        )

    package_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_fieldnames = list(fieldnames)

    if "change_type" not in review_fieldnames:
        review_fieldnames.append("change_type")

    write_csv(
        package_dir / "changed_values.csv",
        review_fieldnames,
        changed_rows,
    )
    write_csv(
        package_dir / "non_high_values.csv",
        review_fieldnames,
        review_rows,
    )
    write_csv(
        package_dir / "unchanged_high_sample.csv",
        review_fieldnames,
        high_sample,
    )

    shutil.copy2(
        detailed_csv,
        package_dir / "stat_values_ocr.csv",
    )
    shutil.copy2(
        combined_csv,
        package_dir / "battle_stats_ocr.csv",
    )

    changed_sheets = create_sheets(
        package_dir / "sheets",
        crop_root,
        changed_rows,
        "changed values",
    )
    non_high_sheets = create_sheets(
        package_dir / "sheets",
        crop_root,
        review_rows,
        "non high",
    )
    high_sheets = create_sheets(
        package_dir / "sheets",
        crop_root,
        high_sample,
        "unchanged high sample",
    )

    change_counts = Counter(
        row["change_type"]
        for row in changed_rows
    )
    status_counts = Counter(
        row.get("status", "")
        for row in review_rows
    )

    summary_lines = [
        "PAQUET DE CONTRÔLE OCR NUMÉRIQUE V2",
        "=" * 72,
        "",
        f"Valeurs modifiées V1 → V2 : {len(changed_rows)}",
        f"Valeurs V2 non-HIGH : {len(review_rows)}",
        f"Échantillon HIGH inchangé : {len(high_sample)}",
        "",
        "Types de modifications :",
    ]

    for name, count in change_counts.most_common():
        summary_lines.append(
            f"- {name:<22} : {count}"
        )

    summary_lines.extend(
        [
            "",
            "Statuts V2 non-HIGH :",
        ]
    )

    for name in (
        "MEDIUM",
        "LOW",
        "NO_DETECTION",
        "ERROR",
    ):
        summary_lines.append(
            f"- {name:<12} : {status_counts[name]}"
        )

    summary_lines.extend(
        [
            "",
            "Fichiers :",
            "- changed_values.csv",
            "- non_high_values.csv",
            "- unchanged_high_sample.csv",
            "- sheets/changed_values_*.jpg",
            "- sheets/non_high_*.jpg",
            "- sheets/unchanged_high_sample_*.jpg",
            "",
        ]
    )

    summary = "\n".join(summary_lines)

    (package_dir / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    with ZipFile(
        zip_path,
        "w",
        ZIP_DEFLATED,
    ) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    path.relative_to(package_dir),
                )

    print(summary)
    print(
        f"Planches changements : {len(changed_sheets)}"
    )
    print(
        f"Planches non-HIGH : {len(non_high_sheets)}"
    )
    print(
        f"Planches HIGH inchangées : {len(high_sheets)}"
    )
    print(
        f"Archive : {zip_path.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
