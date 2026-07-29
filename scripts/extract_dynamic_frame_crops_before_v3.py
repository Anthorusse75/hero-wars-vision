from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageDraw, UnidentifiedImageError


BATCH_ROOT = Path("data/batches/hero_batch_001")
SOURCE_DIR = BATCH_ROOT / "raw"
DETECTION_MANIFEST = (
    BATCH_ROOT
    / "reports/frame_detection_v1/frame_detection_manifest.csv"
)

OUTPUT_ROOT = BATCH_ROOT / "crops_dynamic_v1"
AVATARS_FULL_DIR = OUTPUT_ROOT / "avatars_full"
AVATARS_INNER_DIR = OUTPUT_ROOT / "avatars_inner"
NAMES_DIR = OUTPUT_ROOT / "names"
FALLBACK_DEBUG_DIR = OUTPUT_ROOT / "fallback_debug"
OUTPUT_MANIFEST = OUTPUT_ROOT / "crop_manifest.csv"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

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
    "row_step",
]

# Découpe comparable aux avatars_inner du catalogue initial.
INNER_LEFT = 0.055
INNER_TOP = 0.14
INNER_RIGHT = 0.945
INNER_BOTTOM = 0.91

# Zone du nom, calculée relativement au cadre de l'avatar.
NAME_GAP_IN_FRAME_WIDTHS = 0.10
NAME_WIDTH_IN_FRAME_WIDTHS = 2.10
NAME_TOP_IN_FRAME_HEIGHTS = 0.02
NAME_BOTTOM_IN_FRAME_HEIGHTS = 0.40

# La première zone est volontairement assez large pour accepter les noms longs.
# On ne conserve ensuite que la partie adjacente à l'avatar et le haut de la
# bande, afin d'éliminer les valeurs de puissance et les barres de statistiques.
NAME_ADJACENT_WIDTH_KEEP_RATIO = 0.80
NAME_TOP_BAND_KEEP_RATIO = 0.68


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Manifeste absent : {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        return list(csv.DictReader(csv_file, delimiter=";"))


def as_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "").strip()

    if not value:
        raise ValueError(f"Valeur absente : {field}")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"Valeur non finie : {field}={value}")

    return number


def as_int(row: dict[str, str], field: str) -> int:
    return int(round(as_float(row, field)))


def geometry_is_usable(row: dict[str, str]) -> bool:
    if row.get("status", "").strip().upper() == "FAIL":
        return False

    try:
        width = as_float(row, "source_width")
        height = as_float(row, "source_height")

        values = {
            field: as_float(row, field)
            for field in GEOMETRY_FIELDS
        }

    except (ValueError, TypeError):
        return False

    if width <= 0 or height <= 0:
        return False

    if not (0 < values["left_center_x"] < values["right_center_x"] < width):
        return False

    row_centers = [
        values[f"row_{index}_center_y"]
        for index in range(1, 6)
    ]

    if row_centers != sorted(row_centers):
        return False

    if not all(0 < center < height for center in row_centers):
        return False

    if values["frame_width"] <= 0 or values["frame_height"] <= 0:
        return False

    return True


def normalized_geometry(row: dict[str, str]) -> dict[str, float]:
    width = as_float(row, "source_width")
    height = as_float(row, "source_height")

    result: dict[str, float] = {}

    for field in GEOMETRY_FIELDS:
        value = as_float(row, field)

        if field in {
            "left_center_x",
            "right_center_x",
            "frame_width",
        }:
            result[field] = value / width
        else:
            result[field] = value / height

    return result


def select_fallback_peers(
    target: dict[str, str],
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    target_width = as_int(target, "source_width")
    target_height = as_int(target, "source_height")
    target_ratio = target_width / target_height

    usable_rows = [
        row
        for row in rows
        if row is not target and geometry_is_usable(row)
    ]

    exact_resolution = [
        row
        for row in usable_rows
        if as_int(row, "source_width") == target_width
        and as_int(row, "source_height") == target_height
    ]

    if len(exact_resolution) >= 3:
        return exact_resolution

    same_ratio_family = [
        row
        for row in usable_rows
        if abs(
            as_float(row, "source_width")
            / as_float(row, "source_height")
            - target_ratio
        )
        <= 0.025
    ]

    if len(same_ratio_family) >= 3:
        return sorted(
            same_ratio_family,
            key=lambda row: abs(
                as_float(row, "source_width") - target_width
            )
            + abs(as_float(row, "source_height") - target_height),
        )[:12]

    return sorted(
        usable_rows,
        key=lambda row: abs(
            as_float(row, "source_width")
            / as_float(row, "source_height")
            - target_ratio
        ),
    )[:12]


def fallback_geometry(
    target: dict[str, str],
    rows: list[dict[str, str]],
) -> tuple[dict[str, float], int]:
    peers = select_fallback_peers(target, rows)

    if len(peers) < 3:
        raise RuntimeError(
            f"Pas assez de captures comparables pour {target['source_filename']}"
        )

    normalized_peers = [normalized_geometry(peer) for peer in peers]

    normalized_median = {
        field: median(peer[field] for peer in normalized_peers)
        for field in GEOMETRY_FIELDS
    }

    width = as_float(target, "source_width")
    height = as_float(target, "source_height")

    geometry: dict[str, float] = {}

    for field, normalized_value in normalized_median.items():
        if field in {
            "left_center_x",
            "right_center_x",
            "frame_width",
        }:
            geometry[field] = normalized_value * width
        else:
            geometry[field] = normalized_value * height

    return geometry, len(peers)


def detected_geometry(row: dict[str, str]) -> dict[str, float]:
    return {
        field: as_float(row, field)
        for field in GEOMETRY_FIELDS
    }


def clamp_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    pixel_x1 = max(0, min(image_width - 1, round(x1)))
    pixel_y1 = max(0, min(image_height - 1, round(y1)))
    pixel_x2 = max(pixel_x1 + 1, min(image_width, round(x2)))
    pixel_y2 = max(pixel_y1 + 1, min(image_height, round(y2)))

    return pixel_x1, pixel_y1, pixel_x2, pixel_y2


def frame_box(
    center_x: float,
    center_y: float,
    frame_width: float,
    frame_height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    return clamp_box(
        (
            center_x - frame_width / 2,
            center_y - frame_height / 2,
            center_x + frame_width / 2,
            center_y + frame_height / 2,
        ),
        image_width,
        image_height,
    )


def inner_box(
    full_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = full_box
    width = x2 - x1
    height = y2 - y1

    return clamp_box(
        (
            x1 + width * INNER_LEFT,
            y1 + height * INNER_TOP,
            x1 + width * INNER_RIGHT,
            y1 + height * INNER_BOTTOM,
        ),
        image_width,
        image_height,
    )


def name_box(
    side: str,
    full_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = full_box
    frame_width = x2 - x1
    frame_height = y2 - y1

    gap = frame_width * NAME_GAP_IN_FRAME_WIDTHS
    initial_width = frame_width * NAME_WIDTH_IN_FRAME_WIDTHS

    initial_y1 = y1 + frame_height * NAME_TOP_IN_FRAME_HEIGHTS
    initial_y2 = y1 + frame_height * NAME_BOTTOM_IN_FRAME_HEIGHTS

    # Le nom se trouve toujours du côté de l'avatar.
    # Les parasites se trouvent à l'extrémité opposée :
    # - à droite pour l'équipe de gauche ;
    # - à gauche pour l'équipe de droite.
    kept_width = initial_width * NAME_ADJACENT_WIDTH_KEEP_RATIO

    if side == "L":
        name_x1 = x2 + gap
        name_x2 = name_x1 + kept_width
    elif side == "R":
        name_x2 = x1 - gap
        name_x1 = name_x2 - kept_width
    else:
        raise ValueError(f"Côté invalide : {side}")

    # La partie basse de l'ancienne découpe contenait principalement
    # les barres de statistiques. On conserve uniquement le haut.
    name_y1 = initial_y1
    name_y2 = initial_y1 + (
        initial_y2 - initial_y1
    ) * NAME_TOP_BAND_KEEP_RATIO

    return clamp_box(
        (name_x1, name_y1, name_x2, name_y2),
        image_width,
        image_height,
    )


def save_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    destination: Path,
) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)

    crop = image.crop(box)

    if crop.width <= 0 or crop.height <= 0:
        raise ValueError(f"Découpe vide : {destination}")

    crop.save(destination, format="PNG", optimize=True)

    return crop.size


def save_fallback_debug(
    image: Image.Image,
    source_stem: str,
    boxes: list[
        tuple[
            str,
            int,
            tuple[int, int, int, int],
            tuple[int, int, int, int],
            tuple[int, int, int, int],
        ]
    ],
) -> Path:
    debug = image.copy()
    draw = ImageDraw.Draw(debug)

    for side, slot, full, inner, name in boxes:
        draw.rectangle(full, outline="red", width=5)
        draw.rectangle(inner, outline="orange", width=4)
        draw.rectangle(name, outline="yellow", width=4)
        draw.text(
            (full[0], max(0, full[1] - 22)),
            f"{side}{slot}",
            fill="yellow",
            stroke_width=2,
            stroke_fill="black",
        )

    FALLBACK_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FALLBACK_DEBUG_DIR / f"{source_stem}__fallback.jpg"
    debug.save(output_path, format="JPEG", quality=92, optimize=True)

    return output_path


def main() -> int:
    if OUTPUT_MANIFEST.exists():
        print(
            f"Le résultat existe déjà : {OUTPUT_MANIFEST}",
            file=sys.stderr,
        )
        print("Aucun fichier n'a été modifié.", file=sys.stderr)
        return 1

    try:
        rows = read_manifest(DETECTION_MANIFEST)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    if not rows:
        print("Le manifeste de détection est vide.", file=sys.stderr)
        return 1

    for directory in (
        AVATARS_FULL_DIR,
        AVATARS_INNER_DIR,
        NAMES_DIR,
        FALLBACK_DEBUG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []

    detected_count = 0
    fallback_count = 0
    screenshot_errors = 0

    print(f"Captures à traiter : {len(rows)}")
    print()

    for index, row in enumerate(rows, start=1):
        source_filename = row.get("source_filename", "").strip()
        source_path = SOURCE_DIR / source_filename

        try:
            if not source_filename:
                raise RuntimeError("Nom de fichier source absent")

            if not source_path.exists():
                raise RuntimeError(f"Image source absente : {source_path}")

            with Image.open(source_path) as source:
                image = source.convert("RGB")

            image_width, image_height = image.size

            expected_width = as_int(row, "source_width")
            expected_height = as_int(row, "source_height")

            if (image_width, image_height) != (
                expected_width,
                expected_height,
            ):
                raise RuntimeError(
                    f"Résolution inattendue : {image_width}x{image_height}, "
                    f"attendu {expected_width}x{expected_height}"
                )

            if geometry_is_usable(row):
                geometry = detected_geometry(row)
                geometry_source = "detected"
                fallback_peer_count = 0
                detected_count += 1
            else:
                geometry, fallback_peer_count = fallback_geometry(row, rows)
                geometry_source = "fallback_peer_median"
                fallback_count += 1

            frame_width = geometry["frame_width"]
            frame_height = geometry["frame_height"]

            centers_x = {
                "L": geometry["left_center_x"],
                "R": geometry["right_center_x"],
            }

            centers_y = [
                geometry[f"row_{slot}_center_y"]
                for slot in range(1, 6)
            ]

            fallback_debug_boxes = []

            for side in ("L", "R"):
                for slot, center_y in enumerate(centers_y, start=1):
                    full = frame_box(
                        center_x=centers_x[side],
                        center_y=center_y,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        image_width=image_width,
                        image_height=image_height,
                    )

                    inner = inner_box(
                        full_box=full,
                        image_width=image_width,
                        image_height=image_height,
                    )

                    name = name_box(
                        side=side,
                        full_box=full,
                        image_width=image_width,
                        image_height=image_height,
                    )

                    crop_stem = f"{source_path.stem}__{side}{slot}"

                    full_path = AVATARS_FULL_DIR / f"{crop_stem}.png"
                    inner_path = AVATARS_INNER_DIR / f"{crop_stem}.png"
                    name_path = NAMES_DIR / f"{crop_stem}.png"

                    full_size = save_crop(image, full, full_path)
                    inner_size = save_crop(image, inner, inner_path)
                    name_size = save_crop(image, name, name_path)

                    output_rows.append(
                        {
                            "source_filename": source_filename,
                            "source_width": image_width,
                            "source_height": image_height,
                            "detector_status": row.get("status", ""),
                            "geometry_source": geometry_source,
                            "fallback_peer_count": fallback_peer_count,
                            "side": side,
                            "slot": slot,
                            "avatar_full_file": full_path.as_posix(),
                            "avatar_inner_file": inner_path.as_posix(),
                            "name_file": name_path.as_posix(),
                            "frame_x1": full[0],
                            "frame_y1": full[1],
                            "frame_x2": full[2],
                            "frame_y2": full[3],
                            "avatar_full_width": full_size[0],
                            "avatar_full_height": full_size[1],
                            "avatar_inner_width": inner_size[0],
                            "avatar_inner_height": inner_size[1],
                            "name_width": name_size[0],
                            "name_height": name_size[1],
                        }
                    )

                    if geometry_source == "fallback_peer_median":
                        fallback_debug_boxes.append(
                            (side, slot, full, inner, name)
                        )

            if geometry_source == "fallback_peer_median":
                debug_path = save_fallback_debug(
                    image=image,
                    source_stem=source_path.stem,
                    boxes=fallback_debug_boxes,
                )

                print(
                    f"[{index:03}/{len(rows):03}] FALLBACK | "
                    f"{source_filename} | pairs={fallback_peer_count} | "
                    f"debug={debug_path}"
                )
            else:
                print(
                    f"[{index:03}/{len(rows):03}] "
                    f"{row.get('status', ''):<6} | {source_filename}"
                )

        except (
            RuntimeError,
            ValueError,
            OSError,
            UnidentifiedImageError,
        ) as error:
            print(
                f"[ERREUR] {source_filename or '(sans nom)'} : {error}",
                file=sys.stderr,
            )
            screenshot_errors += 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_filename",
        "source_width",
        "source_height",
        "detector_status",
        "geometry_source",
        "fallback_peer_count",
        "side",
        "slot",
        "avatar_full_file",
        "avatar_inner_file",
        "name_file",
        "frame_x1",
        "frame_y1",
        "frame_x2",
        "frame_y2",
        "avatar_full_width",
        "avatar_full_height",
        "avatar_inner_width",
        "avatar_inner_height",
        "name_width",
        "name_height",
    ]

    with OUTPUT_MANIFEST.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            delimiter=";",
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    expected_slots = (len(rows) - screenshot_errors) * 10

    print()
    print("Résumé :")
    print(f"- Géométries détectées : {detected_count}")
    print(f"- Géométries de secours : {fallback_count}")
    print(f"- Captures en erreur : {screenshot_errors}")
    print(f"- Emplacements extraits : {len(output_rows)}")
    print(f"- Emplacements attendus : {expected_slots}")
    print(f"- Manifeste : {OUTPUT_MANIFEST}")

    if fallback_count:
        print(f"- Contrôle du fallback : {FALLBACK_DEBUG_DIR}")

    if len(output_rows) != expected_slots:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
