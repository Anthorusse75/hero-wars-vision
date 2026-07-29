from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


BATCH_PATTERN = __import__("re").compile(r"^hero_batch_\d{3}$")
METRICS = (
    "power",
    "damage_dealt",
    "damage_taken",
    "healing",
)
REVIEW_STATUSES = {"MEDIUM", "LOW", "NO_DETECTION", "ERROR"}
HIGH_SAMPLE_PER_METRIC = 25
RANDOM_SEED = 20260729

TILE_WIDTH = 310
TILE_HEIGHT = 108
COLUMNS = 6
PAGE_SIZE = 60
MARGIN = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prépare une archive compacte pour contrôler l'OCR numérique "
            "du batch Hero Wars."
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
        help="Remplace un ancien paquet de contrôle.",
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


def choose_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    review_rows = [
        row
        for row in rows
        if row.get("status") in REVIEW_STATUSES
    ]

    rng = random.Random(RANDOM_SEED)
    high_samples: list[dict[str, str]] = []

    for metric in METRICS:
        candidates = [
            row
            for row in rows
            if row.get("metric") == metric
            and row.get("status") == "HIGH"
        ]

        rng.shuffle(candidates)
        high_samples.extend(
            candidates[:HIGH_SAMPLE_PER_METRIC]
        )

    return review_rows, high_samples


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_crop(
    crop_root: Path,
    row: dict[str, str],
) -> Image.Image:
    crop_path = crop_root / row["crop_file"]

    with Image.open(crop_path) as source:
        image = source.convert("RGB")

    image.thumbnail(
        (TILE_WIDTH - 16, 54),
        Image.Resampling.LANCZOS,
    )

    return image


def create_sheets(
    output_dir: Path,
    crop_root: Path,
    rows: list[dict[str, str]],
    title_prefix: str,
) -> list[Path]:
    if not rows:
        return []

    rows = sorted(
        rows,
        key=lambda row: (
            row.get("metric", ""),
            row.get("status", ""),
            int(row.get("screenshot_id") or 0),
            row.get("side", ""),
            int(row.get("slot") or 0),
        ),
    )

    output_paths: list[Path] = []

    for page_start in range(0, len(rows), PAGE_SIZE):
        page_rows = rows[page_start:page_start + PAGE_SIZE]
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

        page_number = page_start // PAGE_SIZE + 1
        draw.text(
            (MARGIN, MARGIN),
            f"{title_prefix} — planche {page_number}",
            fill="black",
        )

        for index, row in enumerate(page_rows):
            column = index % COLUMNS
            grid_row = index // COLUMNS

            x = MARGIN + column * TILE_WIDTH
            y = MARGIN + 34 + grid_row * TILE_HEIGHT

            try:
                crop = load_crop(crop_root, row)
                canvas.paste(
                    crop,
                    (
                        x + (TILE_WIDTH - crop.width) // 2,
                        y,
                    ),
                )
            except (OSError, UnidentifiedImageError, KeyError):
                draw.text(
                    (x + 4, y + 10),
                    "IMAGE ABSENTE",
                    fill="red",
                )

            label_1 = (
                f"{row.get('screenshot_id', '')} "
                f"{row.get('side', '')}{row.get('slot', '')} | "
                f"{row.get('metric', '')}"
            )
            label_2 = (
                f"OCR={row.get('value', '')} | "
                f"conf={safe_float(row.get('confidence', '')):.3f} | "
                f"{row.get('status', '')}"
            )
            label_3 = (
                f"{row.get('hero_name', '')} | "
                f"var={row.get('selected_variant', '')}"
            )

            draw.text((x + 4, y + 59), label_1, fill="black")
            draw.text((x + 4, y + 75), label_2, fill="black")
            draw.text((x + 4, y + 91), label_3, fill="black")

        output_path = (
            output_dir
            / f"{title_prefix.lower().replace(' ', '_')}_{page_number:02d}.jpg"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(
            output_path,
            format="JPEG",
            quality=92,
            optimize=True,
        )
        output_paths.append(output_path)

    return output_paths


def create_summary(
    review_rows: list[dict[str, str]],
    high_rows: list[dict[str, str]],
) -> str:
    counts: dict[str, int] = defaultdict(int)
    metric_counts: dict[tuple[str, str], int] = defaultdict(int)

    for row in review_rows:
        status = row.get("status", "")
        metric = row.get("metric", "")
        counts[status] += 1
        metric_counts[(metric, status)] += 1

    lines = [
        "PAQUET DE CONTRÔLE OCR NUMÉRIQUE",
        "=" * 72,
        "",
        f"Cas non-HIGH inclus : {len(review_rows)}",
        f"Échantillon HIGH inclus : {len(high_rows)}",
        "",
        "Cas non-HIGH par statut :",
    ]

    for status in ("MEDIUM", "LOW", "NO_DETECTION", "ERROR"):
        lines.append(f"- {status:<12} : {counts[status]}")

    lines.extend(
        [
            "",
            "Cas non-HIGH par métrique :",
        ]
    )

    for metric in METRICS:
        parts = [
            f"{status}={metric_counts[(metric, status)]}"
            for status in ("MEDIUM", "LOW", "NO_DETECTION", "ERROR")
        ]
        lines.append(f"- {metric:<14} : " + " | ".join(parts))

    lines.extend(
        [
            "",
            "Contenu :",
            "- review_cases.csv : tous les MEDIUM/LOW/NO_DETECTION/ERROR",
            "- high_sample.csv : 25 valeurs HIGH par métrique",
            "- sheets/review_*.jpg : planches des cas à contrôler",
            "- sheets/high_sample_*.jpg : échantillon de contrôle des HIGH",
            "- stat_values_ocr.csv : résultat OCR complet",
            "- battle_stats_ocr.csv : table consolidée complète",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    if not BATCH_PATTERN.fullmatch(args.batch):
        print(
            "--batch doit avoir la forme hero_batch_002.",
            file=sys.stderr,
        )
        return 2

    batch_root = Path("data/batches") / args.batch
    report_root = batch_root / "reports" / "stat_ocr_v1"
    crop_root = batch_root / "stat_crops_v2"

    detailed_csv = report_root / "stat_values_ocr.csv"
    combined_csv = report_root / "battle_stats_ocr.csv"

    package_dir = batch_root / "reports" / "stat_ocr_review_package"
    zip_path = Path(f"{args.batch}_stat_ocr_review.zip")

    if package_dir.exists():
        if not args.overwrite:
            print(
                f"Le paquet existe déjà : {package_dir}",
                file=sys.stderr,
            )
            print(
                "Relance avec --overwrite pour le remplacer.",
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
            print(
                "Relance avec --overwrite pour la remplacer.",
                file=sys.stderr,
            )
            return 1
        zip_path.unlink()

    try:
        fieldnames, rows = read_csv(detailed_csv)
    except (RuntimeError, csv.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    review_rows, high_rows = choose_rows(rows)

    package_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        package_dir / "review_cases.csv",
        fieldnames,
        review_rows,
    )
    write_csv(
        package_dir / "high_sample.csv",
        fieldnames,
        high_rows,
    )

    shutil.copy2(
        detailed_csv,
        package_dir / "stat_values_ocr.csv",
    )
    shutil.copy2(
        combined_csv,
        package_dir / "battle_stats_ocr.csv",
    )

    review_sheets = create_sheets(
        package_dir / "sheets",
        crop_root,
        review_rows,
        "review",
    )
    high_sheets = create_sheets(
        package_dir / "sheets",
        crop_root,
        high_rows,
        "high sample",
    )

    summary = create_summary(
        review_rows,
        high_rows,
    )
    (package_dir / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    path.relative_to(package_dir),
                )

    print(summary)
    print(f"Planches de revue : {len(review_sheets)}")
    print(f"Planches HIGH : {len(high_sheets)}")
    print(f"Archive : {zip_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
