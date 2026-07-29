from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, UnidentifiedImageError


GEOMETRY_FIELDS = [
    "left_center_x",
    "right_center_x",
    "row_1_center_y",
    "row_2_center_y",
    "row_3_center_y",
    "row_4_center_y",
    "row_5_center_y",
    "frame_width",
    "frame_height",
]

METRICS = (
    "power",
    "damage_dealt",
    "damage_taken",
    "healing",
)

METRIC_Y_OFFSETS = {
    "power": -0.345,
    "damage_dealt": -0.115,
    "damage_taken": 0.115,
    "healing": 0.345,
}

# V2 : les zones V1 étaient trop serrées sur le bord extérieur.
# L'arête proche du centre reste fixe ; seule la largeur vers l'extérieur
# est augmentée afin de récupérer les grands nombres sans toucher l'icône centrale.
VALUE_COLUMN_WIDTH_IN_FRAME_WIDTHS = 1.42
VALUE_COLUMN_GAP_IN_FRAME_WIDTHS = 0.10
VALUE_CROP_HEIGHT_IN_FRAME_HEIGHTS = 0.205

BATCH_PATTERN = re.compile(r"^hero_batch_\d{3}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extrait en V2 les quatre valeurs numériques de chaque "
            "emplacement de statistiques Hero Wars."
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
        help="Remplace une extraction V2 existante.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Manifeste absent : {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def finite_float(
    row: dict[str, str],
    field: str,
) -> float:
    raw = str(row.get(field) or "").strip()

    if not raw:
        raise ValueError(f"Valeur absente : {field}")

    value = float(raw)

    if not math.isfinite(value):
        raise ValueError(f"Valeur invalide : {field}={raw}")

    return value


def clamp_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    pixel_x1 = max(
        0,
        min(image_width - 1, int(round(x1))),
    )
    pixel_y1 = max(
        0,
        min(image_height - 1, int(round(y1))),
    )
    pixel_x2 = max(
        pixel_x1 + 1,
        min(image_width, int(round(x2))),
    )
    pixel_y2 = max(
        pixel_y1 + 1,
        min(image_height, int(round(y2))),
    )

    return pixel_x1, pixel_y1, pixel_x2, pixel_y2


def metric_box(
    side: str,
    metric: str,
    middle_x: float,
    row_center_y: float,
    frame_width: float,
    frame_height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    column_width = (
        frame_width
        * VALUE_COLUMN_WIDTH_IN_FRAME_WIDTHS
    )
    gap = (
        frame_width
        * VALUE_COLUMN_GAP_IN_FRAME_WIDTHS
    )
    crop_height = (
        frame_height
        * VALUE_CROP_HEIGHT_IN_FRAME_HEIGHTS
    )

    center_y = (
        row_center_y
        + frame_height
        * METRIC_Y_OFFSETS[metric]
    )

    # On garde le bord intérieur identique à la V1 et on élargit
    # uniquement vers l'extérieur.
    if side == "L":
        x2 = middle_x - gap
        x1 = x2 - column_width
    else:
        x1 = middle_x + gap
        x2 = x1 + column_width

    return clamp_box(
        (
            x1,
            center_y - crop_height / 2,
            x2,
            center_y + crop_height / 2,
        ),
        image_width,
        image_height,
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "source_filename",
        "screenshot_id",
        "side",
        "slot",
        "metric",
        "crop_file",
        "x1",
        "y1",
        "x2",
        "y2",
        "crop_width",
        "crop_height",
    ]

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
        )
        writer.writeheader()
        writer.writerows(rows)


def screenshot_id_from_name(filename: str) -> str:
    match = re.match(r"^(\d+)", filename)
    return match.group(1) if match else ""


def create_debug_image(
    image: Image.Image,
    boxes: list[
        tuple[
            str,
            int,
            str,
            tuple[int, int, int, int],
        ]
    ],
    destination: Path,
) -> None:
    debug = image.copy()
    draw = ImageDraw.Draw(debug)

    metric_colors = {
        "power": "yellow",
        "damage_dealt": "red",
        "damage_taken": "cyan",
        "healing": "lime",
    }

    for side, slot, metric, box in boxes:
        draw.rectangle(
            box,
            outline=metric_colors[metric],
            width=3,
        )
        draw.text(
            (box[0], max(0, box[1] - 16)),
            f"{side}{slot} {metric}",
            fill=metric_colors[metric],
            stroke_width=2,
            stroke_fill="black",
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    debug.save(
        destination,
        format="JPEG",
        quality=90,
        optimize=True,
    )


def create_contact_sheets(
    crop_rows: list[dict[str, Any]],
    output_root: Path,
) -> list[Path]:
    rows_by_metric: dict[
        str,
        list[dict[str, Any]],
    ] = {
        metric: []
        for metric in METRICS
    }

    for row in crop_rows:
        rows_by_metric[str(row["metric"])].append(row)

    sheet_paths: list[Path] = []

    columns = 10
    tile_width = 220
    tile_height = 82
    margin = 12

    for metric in METRICS:
        metric_rows = rows_by_metric[metric]

        for page_start in range(
            0,
            len(metric_rows),
            100,
        ):
            page_rows = metric_rows[
                page_start:page_start + 100
            ]
            row_count = math.ceil(
                len(page_rows) / columns
            )

            canvas = Image.new(
                "RGB",
                (
                    columns * tile_width + 2 * margin,
                    row_count * tile_height + 2 * margin + 28,
                ),
                "white",
            )
            draw = ImageDraw.Draw(canvas)

            page_number = page_start // 100 + 1

            draw.text(
                (margin, margin),
                f"{metric} — planche {page_number}",
                fill="black",
            )

            for index, row in enumerate(page_rows):
                column = index % columns
                grid_row = index // columns

                x = margin + column * tile_width
                y = margin + 28 + grid_row * tile_height

                crop_path = output_root / str(row["crop_file"])

                with Image.open(crop_path) as source:
                    crop = source.convert("RGB")

                crop.thumbnail(
                    (
                        tile_width - 10,
                        46,
                    ),
                    Image.Resampling.LANCZOS,
                )

                canvas.paste(
                    crop,
                    (
                        x + (tile_width - crop.width) // 2,
                        y,
                    ),
                )

                draw.text(
                    (x + 3, y + 51),
                    (
                        f"{row['screenshot_id']} "
                        f"{row['side']}{row['slot']}"
                    ),
                    fill="black",
                )

            sheet_path = (
                output_root
                / "review"
                / f"{metric}_sheet_{page_number:02d}.jpg"
            )
            sheet_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            canvas.save(
                sheet_path,
                format="JPEG",
                quality=90,
                optimize=True,
            )
            sheet_paths.append(sheet_path)

    return sheet_paths


def main() -> int:
    args = parse_args()

    if not BATCH_PATTERN.fullmatch(args.batch):
        print(
            "--batch doit avoir la forme hero_batch_002.",
            file=sys.stderr,
        )
        return 2

    batch_root = Path("data/batches") / args.batch
    source_dir = batch_root / "raw"
    detection_manifest = (
        batch_root
        / "reports"
        / "frame_detection_v1"
        / "frame_detection_manifest.csv"
    )

    output_root = batch_root / "stat_crops_v2"
    output_manifest = output_root / "stat_crop_manifest.csv"
    debug_dir = output_root / "debug"

    if output_root.exists():
        if not args.overwrite:
            print(
                f"Le résultat existe déjà : {output_root}",
                file=sys.stderr,
            )
            print(
                "Relance avec --overwrite pour le remplacer.",
                file=sys.stderr,
            )
            return 1

        shutil.rmtree(output_root)

    try:
        detection_rows = read_csv(detection_manifest)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    crop_rows: list[dict[str, Any]] = []
    screenshot_errors = 0

    print(f"Captures à traiter : {len(detection_rows)}")
    print(
        "Largeur numérique V2 : "
        f"{VALUE_COLUMN_WIDTH_IN_FRAME_WIDTHS:.2f} × frame_width"
    )
    print()

    for index, row in enumerate(
        detection_rows,
        start=1,
    ):
        source_filename = str(
            row.get("source_filename") or ""
        ).strip()
        source_path = source_dir / source_filename

        try:
            if not source_filename:
                raise RuntimeError("source_filename absent")

            if not source_path.exists():
                raise RuntimeError(
                    f"Image absente : {source_path}"
                )

            values = {
                field: finite_float(row, field)
                for field in GEOMETRY_FIELDS
            }

            with Image.open(source_path) as source:
                image = source.convert("RGB")

            image_width, image_height = image.size

            middle_x = (
                values["left_center_x"]
                + values["right_center_x"]
            ) / 2

            frame_width = values["frame_width"]
            frame_height = values["frame_height"]
            screenshot_id = screenshot_id_from_name(
                source_filename
            )

            debug_boxes: list[
                tuple[
                    str,
                    int,
                    str,
                    tuple[int, int, int, int],
                ]
            ] = []

            for slot in range(1, 6):
                row_center_y = values[
                    f"row_{slot}_center_y"
                ]

                for side in ("L", "R"):
                    for metric in METRICS:
                        box = metric_box(
                            side=side,
                            metric=metric,
                            middle_x=middle_x,
                            row_center_y=row_center_y,
                            frame_width=frame_width,
                            frame_height=frame_height,
                            image_width=image_width,
                            image_height=image_height,
                        )

                        output_name = (
                            f"{Path(source_filename).stem}"
                            f"__{side}{slot}"
                            f"__{metric}.png"
                        )
                        relative_crop = (
                            Path(metric)
                            / output_name
                        )
                        output_path = output_root / relative_crop
                        output_path.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        crop = image.crop(box)
                        crop.save(
                            output_path,
                            format="PNG",
                            optimize=True,
                        )

                        crop_rows.append(
                            {
                                "source_filename": source_filename,
                                "screenshot_id": screenshot_id,
                                "side": side,
                                "slot": slot,
                                "metric": metric,
                                "crop_file": relative_crop.as_posix(),
                                "x1": box[0],
                                "y1": box[1],
                                "x2": box[2],
                                "y2": box[3],
                                "crop_width": crop.width,
                                "crop_height": crop.height,
                            }
                        )

                        debug_boxes.append(
                            (
                                side,
                                slot,
                                metric,
                                box,
                            )
                        )

            create_debug_image(
                image=image,
                boxes=debug_boxes,
                destination=(
                    debug_dir
                    / (
                        f"{Path(source_filename).stem}"
                        "__stats.jpg"
                    )
                ),
            )

            print(
                f"[{index:03}/{len(detection_rows):03}] "
                f"OK | {source_filename}"
            )

        except (
            RuntimeError,
            ValueError,
            OSError,
            UnidentifiedImageError,
        ) as error:
            screenshot_errors += 1
            print(
                f"[ERREUR] {source_filename or '(sans nom)'} : "
                f"{error}",
                file=sys.stderr,
            )

    write_csv(output_manifest, crop_rows)

    sheet_paths = create_contact_sheets(
        crop_rows=crop_rows,
        output_root=output_root,
    )

    expected_crops = (
        (len(detection_rows) - screenshot_errors)
        * 10
        * len(METRICS)
    )

    print()
    print("Résumé :")
    print(
        f"- Captures en erreur : {screenshot_errors}"
    )
    print(
        f"- Découpes créées : {len(crop_rows)}"
    )
    print(
        f"- Découpes attendues : {expected_crops}"
    )
    print(f"- Manifeste : {output_manifest}")
    print(f"- Images de contrôle : {debug_dir}")
    print(f"- Planches créées : {len(sheet_paths)}")

    return 0 if len(crop_rows) == expected_crops else 1


if __name__ == "__main__":
    raise SystemExit(main())
