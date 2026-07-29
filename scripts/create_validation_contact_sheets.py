from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


SOURCE_DIR = Path("data/batches/hero_batch_001/raw")
OUTPUT_DIR = Path("data/batches/hero_batch_001/reports")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

IMAGES_PER_SHEET = 25
COLUMNS = 5

CARD_WIDTH = 420
PREVIEW_HEIGHT = 190
LABEL_HEIGHT = 42
MARGIN = 14


def find_images() -> list[Path]:
    if not SOURCE_DIR.exists():
        raise RuntimeError(f"Dossier absent : {SOURCE_DIR}")

    return sorted(
        path
        for path in SOURCE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def split_into_chunks(
    image_paths: list[Path],
    chunk_size: int,
) -> list[list[Path]]:
    return [
        image_paths[index:index + chunk_size]
        for index in range(0, len(image_paths), chunk_size)
    ]


def short_filename(path: Path, max_length: int = 48) -> str:
    filename = path.name

    if len(filename) <= max_length:
        return filename

    return filename[:max_length - 3] + "..."


def create_sheet(
    image_paths: list[Path],
    sheet_number: int,
    total_sheets: int,
) -> tuple[Path, int]:
    rows = math.ceil(len(image_paths) / COLUMNS)
    card_height = PREVIEW_HEIGHT + LABEL_HEIGHT

    sheet_width = (
        COLUMNS * CARD_WIDTH
        + (COLUMNS + 1) * MARGIN
    )

    title_height = 45

    sheet_height = (
        title_height
        + rows * card_height
        + (rows + 1) * MARGIN
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    draw.text(
        (MARGIN, 15),
        (
            f"Lot hero_batch_001 — "
            f"planche {sheet_number}/{total_sheets}"
        ),
        fill="black",
    )

    loaded_count = 0

    for index, image_path in enumerate(image_paths):
        column = index % COLUMNS
        row = index // COLUMNS

        card_x = (
            MARGIN
            + column * (CARD_WIDTH + MARGIN)
        )

        card_y = (
            title_height
            + MARGIN
            + row * (card_height + MARGIN)
        )

        draw.rectangle(
            (
                card_x,
                card_y,
                card_x + CARD_WIDTH,
                card_y + card_height,
            ),
            outline="gray",
            width=1,
        )

        try:
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                width, height = source.size

                preview = ImageOps.contain(
                    source,
                    (
                        CARD_WIDTH - 12,
                        PREVIEW_HEIGHT - 12,
                    ),
                    method=Image.Resampling.LANCZOS,
                )

            preview_x = (
                card_x
                + (CARD_WIDTH - preview.width) // 2
            )

            preview_y = (
                card_y
                + (PREVIEW_HEIGHT - preview.height) // 2
            )

            sheet.paste(
                preview,
                (preview_x, preview_y),
            )

            label = (
                f"{short_filename(image_path)}\n"
                f"{width} × {height}"
            )

            draw.multiline_text(
                (
                    card_x + 5,
                    card_y + PREVIEW_HEIGHT + 2,
                ),
                label,
                fill="black",
                spacing=2,
            )

            loaded_count += 1

        except (UnidentifiedImageError, OSError) as error:
            draw.multiline_text(
                (
                    card_x + 8,
                    card_y + 8,
                ),
                f"Erreur :\n{image_path.name}\n{error}",
                fill="black",
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"validation_contact_sheet_{sheet_number:02d}.jpg"
    )

    sheet.save(
        output_path,
        format="JPEG",
        quality=92,
    )

    return output_path, loaded_count


def main() -> int:
    try:
        image_paths = find_images()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    if not image_paths:
        print(
            f"Aucune image trouvée dans {SOURCE_DIR}",
            file=sys.stderr,
        )
        return 1

    chunks = split_into_chunks(
        image_paths,
        IMAGES_PER_SHEET,
    )

    print(f"Images trouvées : {len(image_paths)}")
    print(f"Planches à générer : {len(chunks)}")
    print()

    total_loaded = 0

    for sheet_number, chunk in enumerate(
        chunks,
        start=1,
    ):
        output_path, loaded_count = create_sheet(
            image_paths=chunk,
            sheet_number=sheet_number,
            total_sheets=len(chunks),
        )

        total_loaded += loaded_count

        print(
            f"[OK] Planche {sheet_number} : "
            f"{loaded_count} images → {output_path}"
        )

    print()
    print(f"Images affichées : {total_loaded}")
    print(f"Dossier : {OUTPUT_DIR.resolve()}")

    return 0 if total_loaded else 1


if __name__ == "__main__":
    raise SystemExit(main())