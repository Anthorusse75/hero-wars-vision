from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import (
    Image,
    ImageDraw,
    ImageOps,
    UnidentifiedImageError,
)

from crop_hero_samples import (
    ROW_BOUNDS,
    get_slot_boxes,
    normalized_box,
)


SOURCE_DIR = Path("data/batches/hero_batch_001/raw")

OUTPUT_DIR = Path(
    "data/batches/hero_batch_001/reports/crop_layout_check"
)

DEBUG_DIR = OUTPUT_DIR / "debug"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

IMAGES_PER_SHEET = 25
SHEET_COLUMNS = 5

MAX_DEBUG_WIDTH = 1600


def find_images() -> list[Path]:
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


def create_debug_image(
    source_path: Path,
) -> tuple[Path, str]:
    with Image.open(source_path) as source:
        image = source.convert("RGB")

    width, height = image.size
    ratio = width / height

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for side in ("left", "right"):
        side_code = "L" if side == "left" else "R"

        for slot, (row_top, row_bottom) in enumerate(
            ROW_BOUNDS,
            start=1,
        ):
            boxes = get_slot_boxes(
                side=side,
                row_top=row_top,
                row_bottom=row_bottom,
            )

            pixel_boxes = {
                name: normalized_box(
                    image_width=width,
                    image_height=height,
                    box=box,
                )
                for name, box in boxes.items()
            }

            # Bleu : ligne complète.
            draw.rectangle(
                pixel_boxes["row"],
                outline="deepskyblue",
                width=4,
            )

            # Rouge : avatar avec son cadre.
            draw.rectangle(
                pixel_boxes["avatar_full"],
                outline="red",
                width=5,
            )

            # Orange : intérieur destiné au futur modèle.
            draw.rectangle(
                pixel_boxes["avatar_inner"],
                outline="orange",
                width=4,
            )

            # Jaune : nom destiné à EasyOCR.
            draw.rectangle(
                pixel_boxes["name"],
                outline="yellow",
                width=5,
            )

            avatar_box = pixel_boxes["avatar_full"]

            draw.text(
                (
                    avatar_box[0],
                    max(0, avatar_box[1] - 24),
                ),
                f"{side_code}{slot}",
                fill="yellow",
                stroke_width=2,
                stroke_fill="black",
            )

    # Réduit uniquement le fichier de contrôle pour éviter
    # de générer 100 très grosses images.
    if annotated.width > MAX_DEBUG_WIDTH:
        new_height = round(
            annotated.height
            * MAX_DEBUG_WIDTH
            / annotated.width
        )

        annotated = annotated.resize(
            (MAX_DEBUG_WIDTH, new_height),
            Image.Resampling.LANCZOS,
        )

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_path = (
        DEBUG_DIR
        / f"{source_path.stem}__crop_debug.jpg"
    )

    annotated.save(
        debug_path,
        format="JPEG",
        quality=88,
        optimize=True,
    )

    label = (
        f"{source_path.name}\n"
        f"{width} × {height} — ratio {ratio:.3f}"
    )

    return debug_path, label


def create_contact_sheet(
    items: list[tuple[Path, str]],
    sheet_number: int,
    total_sheets: int,
) -> Path:
    columns = SHEET_COLUMNS
    rows = math.ceil(len(items) / columns)

    card_width = 430
    preview_height = 205
    label_height = 48
    card_height = preview_height + label_height
    margin = 14
    title_height = 48

    sheet_width = (
        columns * card_width
        + (columns + 1) * margin
    )

    sheet_height = (
        title_height
        + rows * card_height
        + (rows + 1) * margin
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    draw.text(
        (margin, 16),
        (
            "Contrôle du découpage actuel — "
            f"planche {sheet_number}/{total_sheets}"
        ),
        fill="black",
    )

    for index, (debug_path, label) in enumerate(items):
        column = index % columns
        row = index // columns

        card_x = margin + column * (
            card_width + margin
        )

        card_y = (
            title_height
            + margin
            + row * (card_height + margin)
        )

        draw.rectangle(
            (
                card_x,
                card_y,
                card_x + card_width,
                card_y + card_height,
            ),
            outline="gray",
            width=1,
        )

        try:
            with Image.open(debug_path) as source:
                source = source.convert("RGB")

                preview = ImageOps.contain(
                    source,
                    (
                        card_width - 12,
                        preview_height - 12,
                    ),
                    method=Image.Resampling.LANCZOS,
                )

            preview_x = (
                card_x
                + (card_width - preview.width) // 2
            )

            preview_y = (
                card_y
                + (preview_height - preview.height) // 2
            )

            sheet.paste(
                preview,
                (preview_x, preview_y),
            )

            draw.multiline_text(
                (
                    card_x + 5,
                    card_y + preview_height + 2,
                ),
                label,
                fill="black",
                spacing=2,
            )

        except (UnidentifiedImageError, OSError) as error:
            draw.multiline_text(
                (card_x + 8, card_y + 8),
                f"Erreur :\n{debug_path.name}\n{error}",
                fill="black",
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"crop_layout_sheet_{sheet_number:02d}.jpg"
    )

    sheet.save(
        output_path,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    return output_path


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

    print(f"Images à contrôler : {len(image_paths)}")
    print()

    debug_items: list[tuple[Path, str]] = []
    errors = 0

    for index, source_path in enumerate(
        image_paths,
        start=1,
    ):
        try:
            debug_path, label = create_debug_image(
                source_path
            )

            debug_items.append(
                (debug_path, label)
            )

            print(
                f"[{index:03}/{len(image_paths):03}] "
                f"{source_path.name}"
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:
            print(
                f"[ERREUR] {source_path.name} : {error}",
                file=sys.stderr,
            )
            errors += 1

    chunks = [
        debug_items[index:index + IMAGES_PER_SHEET]
        for index in range(
            0,
            len(debug_items),
            IMAGES_PER_SHEET,
        )
    ]

    print()
    print("Création des planches...")

    for sheet_number, chunk in enumerate(
        chunks,
        start=1,
    ):
        output_path = create_contact_sheet(
            items=chunk,
            sheet_number=sheet_number,
            total_sheets=len(chunks),
        )

        print(
            f"[OK] Planche {sheet_number} : "
            f"{output_path}"
        )

    print()
    print(f"Images contrôlées : {len(debug_items)}")
    print(f"Erreurs : {errors}")
    print(f"Dossier : {OUTPUT_DIR.resolve()}")

    return 0 if debug_items else 1


if __name__ == "__main__":
    raise SystemExit(main())