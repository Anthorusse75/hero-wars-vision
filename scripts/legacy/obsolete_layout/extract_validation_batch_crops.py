from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from check_batch_crop_layout_v2 import (
    adaptive_normalized_box,
    calculate_effective_layout_height,
)
from crop_hero_samples import (
    ROW_BOUNDS,
    get_slot_boxes,
)


BATCH_ROOT = Path("data/batches/hero_batch_001")
SOURCE_DIR = BATCH_ROOT / "raw"
OUTPUT_ROOT = BATCH_ROOT / "crops"

ROWS_DIR = OUTPUT_ROOT / "rows"
AVATARS_FULL_DIR = OUTPUT_ROOT / "avatars_full"
AVATARS_INNER_DIR = OUTPUT_ROOT / "avatars_inner"
NAMES_DIR = OUTPUT_ROOT / "names"

MANIFEST_PATH = OUTPUT_ROOT / "crop_manifest.csv"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

SCREENSHOT_ID_PATTERN = re.compile(r"^(?P<id>\d+)")


def find_source_images() -> list[Path]:
    if not SOURCE_DIR.exists():
        raise RuntimeError(
            f"Dossier source absent : {SOURCE_DIR}"
        )

    return sorted(
        path
        for path in SOURCE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def extract_screenshot_id(path: Path) -> str:
    match = SCREENSHOT_ID_PATTERN.match(path.stem)

    if match:
        return match.group("id")

    return path.stem


def save_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    destination: Path,
) -> tuple[int, int]:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop = image.crop(box)

    if crop.width <= 0 or crop.height <= 0:
        raise ValueError(
            f"Découpe vide pour {destination.name}: {box}"
        )

    crop.save(
        destination,
        format="PNG",
        optimize=True,
    )

    return crop.size


def main() -> int:
    if MANIFEST_PATH.exists():
        print(
            f"Le manifeste existe déjà : {MANIFEST_PATH}",
            file=sys.stderr,
        )
        print(
            "Aucun fichier n'a été modifié.",
            file=sys.stderr,
        )
        return 1

    try:
        source_images = find_source_images()

    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    if not source_images:
        print(
            f"Aucune capture trouvée dans {SOURCE_DIR}",
            file=sys.stderr,
        )
        return 1

    for directory in (
        ROWS_DIR,
        AVATARS_FULL_DIR,
        AVATARS_INNER_DIR,
        NAMES_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest_rows: list[dict[str, object]] = []

    layout_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()

    successful_screenshots = 0
    errors = 0

    print(f"Captures à traiter : {len(source_images)}")
    print()

    for image_index, source_path in enumerate(
        source_images,
        start=1,
    ):
        try:
            with Image.open(source_path) as source:
                image = source.convert("RGB")

            width, height = image.size
            aspect_ratio = width / height

            effective_height, layout_mode = (
                calculate_effective_layout_height(
                    image_width=width,
                    image_height=height,
                )
            )

            screenshot_id = extract_screenshot_id(
                source_path
            )

            resolution_counts[f"{width}x{height}"] += 1
            layout_counts[layout_mode] += 1

            extracted_slots = 0

            for side in ("left", "right"):
                side_code = (
                    "L"
                    if side == "left"
                    else "R"
                )

                for slot, (
                    row_top,
                    row_bottom,
                ) in enumerate(
                    ROW_BOUNDS,
                    start=1,
                ):
                    normalized_boxes = get_slot_boxes(
                        side=side,
                        row_top=row_top,
                        row_bottom=row_bottom,
                    )

                    pixel_boxes = {
                        box_name: adaptive_normalized_box(
                            image_width=width,
                            image_height=height,
                            effective_layout_height=effective_height,
                            box=normalized_box,
                        )
                        for box_name, normalized_box
                        in normalized_boxes.items()
                    }

                    crop_stem = (
                        f"{source_path.stem}"
                        f"__{side_code}{slot}"
                    )

                    row_path = (
                        ROWS_DIR
                        / f"{crop_stem}.png"
                    )

                    avatar_full_path = (
                        AVATARS_FULL_DIR
                        / f"{crop_stem}.png"
                    )

                    avatar_inner_path = (
                        AVATARS_INNER_DIR
                        / f"{crop_stem}.png"
                    )

                    name_path = (
                        NAMES_DIR
                        / f"{crop_stem}.png"
                    )

                    row_size = save_crop(
                        image=image,
                        box=pixel_boxes["row"],
                        destination=row_path,
                    )

                    avatar_full_size = save_crop(
                        image=image,
                        box=pixel_boxes["avatar_full"],
                        destination=avatar_full_path,
                    )

                    avatar_inner_size = save_crop(
                        image=image,
                        box=pixel_boxes["avatar_inner"],
                        destination=avatar_inner_path,
                    )

                    name_size = save_crop(
                        image=image,
                        box=pixel_boxes["name"],
                        destination=name_path,
                    )

                    manifest_rows.append(
                        {
                            "screenshot_id": screenshot_id,
                            "source_filename": source_path.name,
                            "source_width": width,
                            "source_height": height,
                            "aspect_ratio": f"{aspect_ratio:.6f}",
                            "layout_mode": layout_mode,
                            "effective_layout_height": effective_height,
                            "side": side_code,
                            "slot": slot,
                            "row_file": row_path.as_posix(),
                            "avatar_full_file": (
                                avatar_full_path.as_posix()
                            ),
                            "avatar_inner_file": (
                                avatar_inner_path.as_posix()
                            ),
                            "name_file": name_path.as_posix(),
                            "row_width": row_size[0],
                            "row_height": row_size[1],
                            "avatar_full_width": (
                                avatar_full_size[0]
                            ),
                            "avatar_full_height": (
                                avatar_full_size[1]
                            ),
                            "avatar_inner_width": (
                                avatar_inner_size[0]
                            ),
                            "avatar_inner_height": (
                                avatar_inner_size[1]
                            ),
                            "name_width": name_size[0],
                            "name_height": name_size[1],
                        }
                    )

                    extracted_slots += 1

            if extracted_slots != 10:
                raise RuntimeError(
                    f"{source_path.name}: "
                    f"{extracted_slots} emplacements extraits "
                    "au lieu de 10."
                )

            successful_screenshots += 1

            print(
                f"[{image_index:03}/{len(source_images):03}] "
                f"id={screenshot_id} | "
                f"{width}x{height} | "
                f"{layout_mode} | "
                "10 héros extraits"
            )

        except (
            UnidentifiedImageError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            print(
                f"[ERREUR] {source_path.name} : {error}",
                file=sys.stderr,
            )

            errors += 1

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        fieldnames = [
            "screenshot_id",
            "source_filename",
            "source_width",
            "source_height",
            "aspect_ratio",
            "layout_mode",
            "effective_layout_height",
            "side",
            "slot",
            "row_file",
            "avatar_full_file",
            "avatar_inner_file",
            "name_file",
            "row_width",
            "row_height",
            "avatar_full_width",
            "avatar_full_height",
            "avatar_inner_width",
            "avatar_inner_height",
            "name_width",
            "name_height",
        ]

        writer = csv.DictWriter(
            csv_file,
            delimiter=";",
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    expected_crops = successful_screenshots * 10

    print()
    print("Résumé :")
    print(
        f"- Captures traitées : "
        f"{successful_screenshots}"
    )
    print(f"- Captures en erreur : {errors}")
    print(
        f"- Emplacements extraits : "
        f"{len(manifest_rows)}"
    )
    print(
        f"- Emplacements attendus : "
        f"{expected_crops}"
    )

    print()
    print("Modes de mise en page :")

    for layout_mode, count in layout_counts.most_common():
        print(f"- {layout_mode}: {count}")

    print()
    print("Fichiers créés :")
    print(f"- Lignes : {ROWS_DIR}")
    print(f"- Avatars complets : {AVATARS_FULL_DIR}")
    print(f"- Avatars intérieurs : {AVATARS_INNER_DIR}")
    print(f"- Noms : {NAMES_DIR}")
    print(f"- Manifeste : {MANIFEST_PATH}")

    if len(manifest_rows) != expected_crops:
        print(
            "Le nombre de découpes est incohérent.",
            file=sys.stderr,
        )
        return 1

    return 0 if manifest_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())