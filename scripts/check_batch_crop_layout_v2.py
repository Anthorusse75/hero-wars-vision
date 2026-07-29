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
)


SOURCE_DIR = Path("data/batches/hero_batch_001/raw")

OUTPUT_DIR = Path(
    "data/batches/hero_batch_001/reports/crop_layout_check_v2"
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

# Le jeu est initialement conçu pour une zone proche du ratio 20:9.
REFERENCE_ASPECT_RATIO = 20 / 9

# En dessous de ce ratio, la hauteur supplémentaire de la capture
# ne doit pas être utilisée pour positionner l’interface du jeu.
TABLET_ASPECT_THRESHOLD = 1.70


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


def calculate_effective_layout_height(
    image_width: int,
    image_height: int,
) -> tuple[int, str]:
    """
    Calcule la hauteur réellement occupée par l’interface Hero Wars.

    Sur les captures larges, toute la hauteur correspond à l’interface.

    Sur les captures de tablette ou plus carrées, le jeu conserve
    approximativement une zone 20:9 en haut de l’image. La hauteur
    supplémentaire située dessous ne doit pas décaler les découpes.
    """

    screenshot_ratio = image_width / image_height

    if screenshot_ratio >= TABLET_ASPECT_THRESHOLD:
        return image_height, "full_height"

    effective_height = round(
        image_width / REFERENCE_ASPECT_RATIO
    )

    effective_height = max(
        1,
        min(effective_height, image_height),
    )

    return effective_height, "top_20_9"


def adaptive_normalized_box(
    image_width: int,
    image_height: int,
    effective_layout_height: int,
    box: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """
    Convertit une zone normalisée en coordonnées réelles.

    Les coordonnées horizontales dépendent toujours de la largeur
    totale de l’image.

    Les coordonnées verticales utilisent la hauteur effective de
    l’interface du jeu, et non nécessairement la hauteur complète
    de la capture.
    """

    x1, y1, x2, y2 = box

    pixel_x1 = max(
        0,
        round(x1 * image_width),
    )

    pixel_x2 = min(
        image_width,
        round(x2 * image_width),
    )

    pixel_y1 = max(
        0,
        round(y1 * effective_layout_height),
    )

    pixel_y2 = min(
        image_height,
        round(y2 * effective_layout_height),
    )

    if pixel_x2 <= pixel_x1:
        pixel_x2 = min(
            image_width,
            pixel_x1 + 1,
        )

    if pixel_y2 <= pixel_y1:
        pixel_y2 = min(
            image_height,
            pixel_y1 + 1,
        )

    return (
        pixel_x1,
        pixel_y1,
        pixel_x2,
        pixel_y2,
    )


def create_debug_image(
    source_path: Path,
) -> tuple[Path, str]:
    with Image.open(source_path) as source:
        image = source.convert("RGB")

    width, height = image.size
    ratio = width / height

    effective_height, layout_mode = (
        calculate_effective_layout_height(
            image_width=width,
            image_height=height,
        )
    )

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Ligne verte indiquant la fin de la zone verticale utilisée.
    # Elle n’apparaît que lorsque la hauteur est corrigée.
    if effective_height < height:
        draw.line(
            (
                0,
                effective_height,
                width,
                effective_height,
            ),
            fill="lime",
            width=max(4, width // 500),
        )

        draw.text(
            (
                20,
                max(10, effective_height - 35),
            ),
            "Fin de la zone d'interface 20:9",
            fill="lime",
            stroke_width=2,
            stroke_fill="black",
        )

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
                name: adaptive_normalized_box(
                    image_width=width,
                    image_height=height,
                    effective_layout_height=effective_height,
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

            # Rouge : avatar complet avec cadre.
            draw.rectangle(
                pixel_boxes["avatar_full"],
                outline="red",
                width=5,
            )

            # Orange : intérieur de l’avatar destiné au modèle.
            draw.rectangle(
                pixel_boxes["avatar_inner"],
                outline="orange",
                width=4,
            )

            # Jaune : zone du nom destinée à EasyOCR.
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

    # Réduit uniquement l’image de contrôle.
    # L’image source n’est jamais modifiée.
    if annotated.width > MAX_DEBUG_WIDTH:
        new_height = round(
            annotated.height
            * MAX_DEBUG_WIDTH
            / annotated.width
        )

        annotated = annotated.resize(
            (
                MAX_DEBUG_WIDTH,
                new_height,
            ),
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
        f"{width} × {height} — ratio {ratio:.3f}\n"
        f"layout={layout_mode} — hauteur utile={effective_height}"
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
    label_height = 62
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
        (
            sheet_width,
            sheet_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    draw.text(
        (
            margin,
            16,
        ),
        (
            "Contrôle du découpage adaptatif — "
            f"planche {sheet_number}/{total_sheets}"
        ),
        fill="black",
    )

    for index, (debug_path, label) in enumerate(items):
        column = index % columns
        row = index // columns

        card_x = (
            margin
            + column * (card_width + margin)
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
                (
                    preview_x,
                    preview_y,
                ),
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

        except (
            UnidentifiedImageError,
            OSError,
        ) as error:
            draw.multiline_text(
                (
                    card_x + 8,
                    card_y + 8,
                ),
                (
                    f"Erreur :\n"
                    f"{debug_path.name}\n"
                    f"{error}"
                ),
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
        print(
            error,
            file=sys.stderr,
        )
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

    layout_counts = {
        "full_height": 0,
        "top_20_9": 0,
    }

    for index, source_path in enumerate(
        image_paths,
        start=1,
    ):
        try:
            with Image.open(source_path) as source:
                width, height = source.size

            effective_height, layout_mode = (
                calculate_effective_layout_height(
                    image_width=width,
                    image_height=height,
                )
            )

            layout_counts[layout_mode] += 1

            debug_path, label = create_debug_image(
                source_path
            )

            debug_items.append(
                (
                    debug_path,
                    label,
                )
            )

            print(
                f"[{index:03}/{len(image_paths):03}] "
                f"{source_path.name} | "
                f"mode={layout_mode} | "
                f"hauteur utile={effective_height}"
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
    print()

    print("Modes de mise en page :")
    print(
        f"- hauteur complète : "
        f"{layout_counts['full_height']}"
    )
    print(
        f"- zone supérieure 20:9 : "
        f"{layout_counts['top_20_9']}"
    )

    print()
    print(f"Dossier : {OUTPUT_DIR.resolve()}")

    return 0 if debug_items else 1


if __name__ == "__main__":
    raise SystemExit(main())