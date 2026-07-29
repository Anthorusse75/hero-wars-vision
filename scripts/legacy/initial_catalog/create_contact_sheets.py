from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


SAMPLES_ROOT = Path("data/samples")
OUTPUT_ROOT = Path("data/reports")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

COLUMNS = 4
CARD_WIDTH = 520
IMAGE_HEIGHT = 230
LABEL_HEIGHT = 50
MARGIN = 20


def find_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def create_contact_sheet(combat_type: str) -> Path | None:
    source_directory = SAMPLES_ROOT / combat_type

    if not source_directory.exists():
        print(f"Dossier absent : {source_directory}")
        return None

    image_paths = find_images(source_directory)

    if not image_paths:
        print(f"Aucune image trouvée dans {source_directory}")
        return None

    rows = math.ceil(len(image_paths) / COLUMNS)

    card_height = IMAGE_HEIGHT + LABEL_HEIGHT

    sheet_width = (
        COLUMNS * CARD_WIDTH
        + (COLUMNS + 1) * MARGIN
    )

    sheet_height = (
        rows * card_height
        + (rows + 1) * MARGIN
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    loaded_count = 0

    for index, image_path in enumerate(image_paths):
        column = index % COLUMNS
        row = index // COLUMNS

        card_x = MARGIN + column * (CARD_WIDTH + MARGIN)
        card_y = MARGIN + row * (card_height + MARGIN)

        try:
            with Image.open(image_path) as source_image:
                source_image = source_image.convert("RGB")

                original_width, original_height = source_image.size

                thumbnail = ImageOps.contain(
                    source_image,
                    (
                        CARD_WIDTH - 20,
                        IMAGE_HEIGHT - 20,
                    ),
                )

            thumbnail_x = (
                card_x
                + (CARD_WIDTH - thumbnail.width) // 2
            )

            thumbnail_y = (
                card_y
                + (IMAGE_HEIGHT - thumbnail.height) // 2
            )

            sheet.paste(
                thumbnail,
                (thumbnail_x, thumbnail_y),
            )

            label = (
                f"{image_path.name}\n"
                f"{original_width} × {original_height}"
            )

            draw.multiline_text(
                (
                    card_x + 5,
                    card_y + IMAGE_HEIGHT + 2,
                ),
                label,
                fill="black",
                spacing=2,
            )

            loaded_count += 1

        except (UnidentifiedImageError, OSError) as error:
            draw.text(
                (
                    card_x + 10,
                    card_y + 10,
                ),
                f"Erreur : {image_path.name}\n{error}",
                fill="black",
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_ROOT
        / f"{combat_type}_contact_sheet.jpg"
    )

    sheet.save(
        output_path,
        format="JPEG",
        quality=92,
    )

    print(
        f"[OK] {combat_type} : "
        f"{loaded_count} images → {output_path}"
    )

    return output_path


def main() -> int:
    generated_files = []

    for combat_type in ("hero", "titan"):
        output_path = create_contact_sheet(combat_type)

        if output_path is not None:
            generated_files.append(output_path)

    if not generated_files:
        print(
            "Aucune planche générée.",
            file=sys.stderr,
        )
        return 1

    print()
    print("Fichiers générés :")

    for path in generated_files:
        print(f"- {path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())