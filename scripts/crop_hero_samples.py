from __future__ import annotations

import math
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


SOURCE_DIR = Path("data/samples/hero")
CROPS_DIR = Path("data/crops/hero")
REPORTS_DIR = Path("data/reports")
DEBUG_DIR = REPORTS_DIR / "hero_crop_debug"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

# Positions verticales des cinq personnages.
# Les valeurs sont des proportions de la hauteur totale de l’image.
ROW_BOUNDS = [
    (0.132, 0.286),
    (0.303, 0.456),
    (0.474, 0.625),
    (0.645, 0.796),
    (0.815, 0.966),
]


def normalized_box(
    image_width: int,
    image_height: int,
    box: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """
    Convertit une zone exprimée entre 0 et 1 en coordonnées réelles.
    """

    x1, y1, x2, y2 = box

    pixel_x1 = max(0, round(x1 * image_width))
    pixel_y1 = max(0, round(y1 * image_height))
    pixel_x2 = min(image_width, round(x2 * image_width))
    pixel_y2 = min(image_height, round(y2 * image_height))

    return pixel_x1, pixel_y1, pixel_x2, pixel_y2


def get_slot_boxes(
    side: str,
    row_top: float,
    row_bottom: float,
) -> dict[str, tuple[float, float, float, float]]:
    """
    Retourne les zones d’une ligne de héros.

    avatar_full  : portrait complet avec cadre, niveau et étoile
    avatar_inner : intérieur du portrait, destiné au futur modèle
    name         : nom affiché à côté du portrait
    row          : ligne complète avec les statistiques
    """

    if side == "left":
        return {
            "row": (
                0.195,
                row_top - 0.005,
                0.560,
                row_bottom + 0.005,
            ),
            "avatar_full": (
                0.201,
                row_top,
                0.274,
                row_bottom,
            ),
            "avatar_inner": (
                0.205,
                row_top + 0.022,
                0.271,
                row_bottom - 0.014,
            ),
            "name": (
                0.279,
                row_top + 0.005,
                0.425,
                row_top + 0.058,
            ),
        }

    return {
        "row": (
            0.560,
            row_top - 0.005,
            0.805,
            row_bottom + 0.005,
        ),
        "avatar_full": (
            0.726,
            row_top,
            0.799,
            row_bottom,
        ),
        "avatar_inner": (
            0.729,
            row_top + 0.022,
            0.795,
            row_bottom - 0.014,
        ),
        "name": (
            0.575,
            row_top + 0.005,
            0.724,
            row_top + 0.058,
        ),
    }


def screenshot_id(path: Path) -> str:
    match = re.match(r"^(\d+)", path.stem)

    if match:
        return match.group(1)

    return path.stem


def save_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    cropped_image = image.crop(box)

    # PNG évite d’ajouter une nouvelle compression JPEG.
    cropped_image.save(
        destination,
        format="PNG",
        optimize=True,
    )


def create_contact_sheet(
    items: list[tuple[str, Path]],
    destination: Path,
    title: str,
    columns: int,
    card_width: int,
    card_height: int,
    preview_width: int,
    preview_height: int,
) -> None:
    if not items:
        return

    margin = 12
    title_height = 42
    rows = math.ceil(len(items) / columns)

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
    draw.text((margin, 12), title, fill="black")

    for index, (label, image_path) in enumerate(items):
        column = index % columns
        row = index // columns

        card_x = margin + column * (card_width + margin)
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
            with Image.open(image_path) as source:
                source = source.convert("RGB")

                preview = ImageOps.contain(
                    source,
                    (preview_width, preview_height),
                    method=Image.Resampling.LANCZOS,
                )

            preview_x = (
                card_x
                + (card_width - preview.width) // 2
            )

            preview_y = (
                card_y
                + 5
                + (preview_height - preview.height) // 2
            )

            sheet.paste(
                preview,
                (preview_x, preview_y),
            )

            draw.text(
                (
                    card_x + 5,
                    card_y + preview_height + 9,
                ),
                label,
                fill="black",
            )

        except (UnidentifiedImageError, OSError) as error:
            draw.text(
                (card_x + 5, card_y + 5),
                f"{label}\nErreur : {error}",
                fill="black",
            )

    destination.parent.mkdir(parents=True, exist_ok=True)

    sheet.save(
        destination,
        format="JPEG",
        quality=92,
    )


def main() -> int:
    if not SOURCE_DIR.exists():
        print(
            f"Dossier absent : {SOURCE_DIR}",
            file=sys.stderr,
        )
        return 1

    source_images = sorted(
        path
        for path in SOURCE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not source_images:
        print(
            f"Aucune image dans {SOURCE_DIR}",
            file=sys.stderr,
        )
        return 1

    rows_dir = CROPS_DIR / "rows"
    avatars_full_dir = CROPS_DIR / "avatars_full"
    avatars_inner_dir = CROPS_DIR / "avatars_inner"
    names_dir = CROPS_DIR / "names"

    for directory in (
        rows_dir,
        avatars_full_dir,
        avatars_inner_dir,
        names_dir,
        DEBUG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    row_items: list[tuple[str, Path]] = []
    avatar_items: list[tuple[str, Path]] = []
    name_items: list[tuple[str, Path]] = []
    debug_items: list[tuple[str, Path]] = []

    errors = 0

    for source_path in source_images:
        try:
            with Image.open(source_path) as source:
                image = source.convert("RGB")

            width, height = image.size
            current_id = screenshot_id(source_path)

            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)

            for side in ("left", "right"):
                side_code = "L" if side == "left" else "R"

                for slot, (row_top, row_bottom) in enumerate(
                    ROW_BOUNDS,
                    start=1,
                ):
                    boxes = get_slot_boxes(
                        side,
                        row_top,
                        row_bottom,
                    )

                    pixel_boxes = {
                        name: normalized_box(
                            width,
                            height,
                            box,
                        )
                        for name, box in boxes.items()
                    }

                    crop_name = (
                        f"{source_path.stem}"
                        f"__{side_code}{slot}.png"
                    )

                    row_path = rows_dir / crop_name
                    avatar_full_path = (
                        avatars_full_dir / crop_name
                    )
                    avatar_inner_path = (
                        avatars_inner_dir / crop_name
                    )
                    name_path = names_dir / crop_name

                    save_crop(
                        image,
                        pixel_boxes["row"],
                        row_path,
                    )

                    save_crop(
                        image,
                        pixel_boxes["avatar_full"],
                        avatar_full_path,
                    )

                    save_crop(
                        image,
                        pixel_boxes["avatar_inner"],
                        avatar_inner_path,
                    )

                    save_crop(
                        image,
                        pixel_boxes["name"],
                        name_path,
                    )

                    label = (
                        f"{current_id} "
                        f"{side_code}{slot}"
                    )

                    row_items.append((label, row_path))
                    avatar_items.append(
                        (label, avatar_inner_path)
                    )
                    name_items.append((label, name_path))

                    # Bleu : ligne complète
                    draw.rectangle(
                        pixel_boxes["row"],
                        outline="deepskyblue",
                        width=2,
                    )

                    # Rouge : avatar complet
                    draw.rectangle(
                        pixel_boxes["avatar_full"],
                        outline="red",
                        width=3,
                    )

                    # Orange : intérieur destiné au modèle
                    draw.rectangle(
                        pixel_boxes["avatar_inner"],
                        outline="orange",
                        width=2,
                    )

                    # Jaune : nom destiné à l’OCR
                    draw.rectangle(
                        pixel_boxes["name"],
                        outline="yellow",
                        width=3,
                    )

                    avatar_box = pixel_boxes["avatar_full"]

                    draw.text(
                        (
                            avatar_box[0],
                            max(0, avatar_box[1] - 16),
                        ),
                        f"{side_code}{slot}",
                        fill="yellow",
                    )

            debug_path = (
                DEBUG_DIR
                / f"{source_path.stem}__debug.jpg"
            )

            annotated.save(
                debug_path,
                format="JPEG",
                quality=92,
            )

            debug_items.append(
                (
                    f"{current_id} — {width}x{height}",
                    debug_path,
                )
            )

            print(
                f"[OK] {source_path.name} : "
                "10 lignes extraites"
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

    create_contact_sheet(
        items=debug_items,
        destination=(
            REPORTS_DIR
            / "hero_crop_overview.jpg"
        ),
        title="Contrôle des zones de découpage",
        columns=4,
        card_width=500,
        card_height=270,
        preview_width=480,
        preview_height=220,
    )

    create_contact_sheet(
        items=avatar_items,
        destination=(
            REPORTS_DIR
            / "hero_avatar_sheet.jpg"
        ),
        title="Avatars intérieurs",
        columns=10,
        card_width=145,
        card_height=160,
        preview_width=125,
        preview_height=120,
    )

    create_contact_sheet(
        items=name_items,
        destination=(
            REPORTS_DIR
            / "hero_name_sheet.jpg"
        ),
        title="Zones des noms",
        columns=5,
        card_width=330,
        card_height=105,
        preview_width=310,
        preview_height=62,
    )

    create_contact_sheet(
        items=row_items,
        destination=(
            REPORTS_DIR
            / "hero_row_sheet.jpg"
        ),
        title="Lignes complètes",
        columns=4,
        card_width=500,
        card_height=145,
        preview_width=480,
        preview_height=95,
    )

    print()
    print(f"Captures analysées : {len(source_images)}")
    print(f"Lignes extraites : {len(row_items)}")
    print(f"Avatars extraits : {len(avatar_items)}")
    print(f"Noms extraits : {len(name_items)}")
    print(f"Erreurs : {errors}")
    print()
    print("Planches générées :")
    print("- data/reports/hero_crop_overview.jpg")
    print("- data/reports/hero_avatar_sheet.jpg")
    print("- data/reports/hero_name_sheet.jpg")
    print("- data/reports/hero_row_sheet.jpg")

    return 0 if row_items else 1


if __name__ == "__main__":
    raise SystemExit(main())