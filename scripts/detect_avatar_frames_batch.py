from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


SOURCE_DIR = Path("data/batches/hero_batch_001/raw")
OUTPUT_DIR = Path(
    "data/batches/hero_batch_001/reports/frame_detection_v1"
)
DEBUG_DIR = OUTPUT_DIR / "debug"
MANIFEST_PATH = OUTPUT_DIR / "frame_detection_manifest.csv"

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


@dataclass(frozen=True)
class Candidate:
    x: int
    y: int
    width: int
    height: int
    score: float
    source: str

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True)
class Detection:
    left_center_x: float
    right_center_x: float
    row_centers_y: tuple[float, float, float, float, float]
    frame_width: float
    frame_height: float
    row_step: float
    mean_alignment_error: float
    left_support: int
    right_support: int
    candidate_count: int


def find_images() -> list[Path]:
    if not SOURCE_DIR.exists():
        raise RuntimeError(f"Dossier absent : {SOURCE_DIR}")

    paths = sorted(
        path
        for path in SOURCE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not paths:
        raise RuntimeError(f"Aucune image dans {SOURCE_DIR}")

    return paths


def cluster_1d(
    values: np.ndarray,
    tolerance: float,
) -> list[list[int]]:
    if len(values) == 0:
        return []

    order = np.argsort(values)
    groups: list[list[int]] = [[int(order[0])]]

    for raw_index in order[1:]:
        index = int(raw_index)
        current_group = groups[-1]
        current_mean = float(
            np.mean([values[item] for item in current_group])
        )

        if abs(float(values[index]) - current_mean) <= tolerance:
            current_group.append(index)
        else:
            groups.append([index])

    return groups


def build_candidates(image_bgr: np.ndarray) -> list[Candidate]:
    image_height, image_width = image_bgr.shape[:2]
    minimum_dimension = min(image_width, image_height)

    candidates: list[Candidate] = []

    # 1. Contours géométriques : utiles même si le cadre est gris.
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if not (
            0.045 * minimum_dimension
            <= width
            <= 0.20 * minimum_dimension
        ):
            continue

        if not (
            0.045 * minimum_dimension
            <= height
            <= 0.21 * minimum_dimension
        ):
            continue

        if not 0.65 <= width / height <= 1.35:
            continue

        if area <= 0.015 * width * height:
            continue

        center_x = x + width / 2

        # Exclut le titre, le bas de l'écran et la zone centrale des stats.
        if y < 0.04 * image_height or y > 0.90 * image_height:
            continue

        if 0.38 * image_width < center_x < 0.62 * image_width:
            continue

        # Exclut le bouton de fermeture en haut à droite.
        if x > 0.85 * image_width and y < 0.15 * image_height:
            continue

        candidates.append(
            Candidate(
                x=x,
                y=y,
                width=width,
                height=height,
                score=min(
                    1.0,
                    area / max(0.1 * width * height, 1.0),
                ),
                source="edge",
            )
        )

    # 2. Composants colorés : efficaces sur les cadres rouges/verts/bleus.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    color_mask = (
        (hsv[:, :, 1] > 70)
        & (hsv[:, :, 2] > 50)
    ).astype(np.uint8) * 255

    kernel_size = max(
        3,
        int(round(minimum_dimension / 350)),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )

    color_mask = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        color_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        fill_ratio = area / max(width * height, 1)

        if not (
            0.045 * minimum_dimension
            <= width
            <= 0.21 * minimum_dimension
        ):
            continue

        if not (
            0.045 * minimum_dimension
            <= height
            <= 0.22 * minimum_dimension
        ):
            continue

        if not 0.65 <= width / height <= 1.35:
            continue

        if fill_ratio <= 0.15:
            continue

        center_x = x + width / 2

        if y < 0.04 * image_height or y > 0.90 * image_height:
            continue

        if 0.38 * image_width < center_x < 0.62 * image_width:
            continue

        if x > 0.85 * image_width and y < 0.15 * image_height:
            continue

        candidates.append(
            Candidate(
                x=x,
                y=y,
                width=width,
                height=height,
                score=float(fill_ratio),
                source="color",
            )
        )

    return candidates


def select_column(
    candidates: list[Candidate],
    image_width: int,
    side: str,
) -> dict[str, object] | None:
    center_values = np.asarray(
        [candidate.center_x for candidate in candidates],
        dtype=np.float64,
    )

    groups = cluster_1d(
        center_values,
        tolerance=max(12.0, 0.03 * image_width),
    )

    column_candidates: list[dict[str, object]] = []

    for group in groups:
        group_candidates = [candidates[index] for index in group]

        maximum_width = max(
            candidate.width
            for candidate in group_candidates
        )
        maximum_height = max(
            candidate.height
            for candidate in group_candidates
        )

        # Les grands contours correspondent généralement au cadre extérieur.
        outer_candidates = [
            candidate
            for candidate in group_candidates
            if candidate.width >= 0.72 * maximum_width
            and candidate.height >= 0.72 * maximum_height
        ]

        if not outer_candidates:
            outer_candidates = group_candidates

        center_x = float(
            np.median(
                [candidate.center_x for candidate in outer_candidates]
            )
        )

        if side == "left" and center_x >= image_width / 2:
            continue

        if side == "right" and center_x <= image_width / 2:
            continue

        median_height = float(
            np.median(
                [candidate.height for candidate in outer_candidates]
            )
        )

        y_values = np.asarray(
            [candidate.center_y for candidate in outer_candidates],
            dtype=np.float64,
        )

        y_groups = cluster_1d(
            y_values,
            tolerance=max(8.0, 0.22 * median_height),
        )

        y_span = (
            float(np.ptp(y_values))
            if len(y_values) > 1
            else 0.0
        )

        column_candidates.append(
            {
                "center_x": center_x,
                "outer_candidates": outer_candidates,
                "row_support": len(y_groups),
                "y_span": y_span,
            }
        )

    if not column_candidates:
        return None

    selected = max(
        column_candidates,
        key=lambda item: (
            int(item["row_support"]),
            float(item["y_span"]),
            len(item["outer_candidates"]),
        ),
    )

    if int(selected["row_support"]) < 2:
        return None

    return selected


def fit_five_rows(
    observations: np.ndarray,
    frame_height: float,
    image_height: int,
) -> tuple[np.ndarray, float, float]:
    possible_steps: list[float] = []

    for first_index in range(len(observations)):
        for second_index in range(
            first_index + 1,
            len(observations),
        ):
            difference = (
                observations[second_index]
                - observations[first_index]
            )

            for row_gap in range(1, 5):
                step = difference / row_gap

                if (
                    0.90 * frame_height
                    <= step
                    <= 1.35 * frame_height
                ):
                    possible_steps.append(float(step))

    if not possible_steps:
        possible_steps = [1.08 * frame_height]

    if len(possible_steps) == 1:
        minimum_step = possible_steps[0] * 0.95
        maximum_step = possible_steps[0] * 1.05
    else:
        minimum_step, maximum_step = np.percentile(
            possible_steps,
            [10, 90],
        )

    best_result: tuple[
        float,
        np.ndarray,
        float,
        np.ndarray,
    ] | None = None

    for step in np.linspace(
        float(minimum_step),
        float(maximum_step),
        120,
    ):
        for observation in observations:
            for row_index in range(5):
                first_center = observation - row_index * step
                centers = first_center + step * np.arange(5)

                if first_center - frame_height / 2 < 0:
                    continue

                if centers[-1] + frame_height / 2 > 0.98 * image_height:
                    continue

                distance_matrix = np.abs(
                    observations[:, None] - centers[None, :]
                )

                distances = np.min(
                    distance_matrix,
                    axis=1,
                )

                matched_count = int(
                    np.sum(distances < 0.20 * frame_height)
                )

                score = float(
                    np.sum(
                        np.minimum(
                            distances,
                            0.50 * frame_height,
                        )
                    )
                    - matched_count * 0.25 * frame_height
                )

                if (
                    best_result is None
                    or score < best_result[0]
                ):
                    best_result = (
                        score,
                        centers,
                        float(step),
                        distances,
                    )

    if best_result is None:
        raise RuntimeError(
            "Impossible d'ajuster les cinq lignes."
        )

    _, centers, step, distances = best_result

    mean_alignment_error = float(
        np.mean(
            np.minimum(
                distances,
                frame_height,
            )
        )
        / frame_height
    )

    return centers, step, mean_alignment_error


def detect_avatar_grid(image_bgr: np.ndarray) -> Detection:
    image_height, image_width = image_bgr.shape[:2]
    candidates = build_candidates(image_bgr)

    if not candidates:
        raise RuntimeError("Aucun cadre candidat détecté.")

    left_column = select_column(
        candidates=candidates,
        image_width=image_width,
        side="left",
    )

    right_column = select_column(
        candidates=candidates,
        image_width=image_width,
        side="right",
    )

    if left_column is None and right_column is None:
        raise RuntimeError(
            "Aucune colonne d'avatars détectée."
        )

    selected_candidates: list[Candidate] = []

    if left_column is not None:
        selected_candidates.extend(
            left_column["outer_candidates"]
        )

    if right_column is not None:
        selected_candidates.extend(
            right_column["outer_candidates"]
        )

    maximum_width = max(
        candidate.width
        for candidate in selected_candidates
    )
    maximum_height = max(
        candidate.height
        for candidate in selected_candidates
    )

    selected_candidates = [
        candidate
        for candidate in selected_candidates
        if candidate.width >= 0.72 * maximum_width
        and candidate.height >= 0.72 * maximum_height
    ]

    frame_width = float(
        np.median(
            [candidate.width for candidate in selected_candidates]
        )
    )
    frame_height = float(
        np.median(
            [candidate.height for candidate in selected_candidates]
        )
    )

    if left_column is not None:
        left_center_x = float(left_column["center_x"])
    else:
        right_center = float(right_column["center_x"])
        left_center_x = image_width - right_center

    if right_column is not None:
        right_center_x = float(right_column["center_x"])
    else:
        right_center_x = image_width - left_center_x

    # Force une symétrie parfaite autour du centre de l'écran.
    center_offset = (
        (image_width / 2 - left_center_x)
        + (right_center_x - image_width / 2)
    ) / 2

    left_center_x = image_width / 2 - center_offset
    right_center_x = image_width / 2 + center_offset

    y_values = np.asarray(
        [candidate.center_y for candidate in selected_candidates],
        dtype=np.float64,
    )

    y_groups = cluster_1d(
        y_values,
        tolerance=max(8.0, 0.18 * frame_height),
    )

    y_observations = np.asarray(
        [
            np.median([y_values[index] for index in group])
            for group in y_groups
        ],
        dtype=np.float64,
    )

    row_centers, row_step, mean_alignment_error = (
        fit_five_rows(
            observations=y_observations,
            frame_height=frame_height,
            image_height=image_height,
        )
    )

    return Detection(
        left_center_x=float(left_center_x),
        right_center_x=float(right_center_x),
        row_centers_y=tuple(
            float(value)
            for value in row_centers
        ),
        frame_width=frame_width,
        frame_height=frame_height,
        row_step=float(row_step),
        mean_alignment_error=mean_alignment_error,
        left_support=(
            int(left_column["row_support"])
            if left_column is not None
            else 0
        ),
        right_support=(
            int(right_column["row_support"])
            if right_column is not None
            else 0
        ),
        candidate_count=len(candidates),
    )


def detection_status(detection: Detection) -> str:
    strongest_support = max(
        detection.left_support,
        detection.right_support,
    )

    if (
        strongest_support >= 4
        and detection.mean_alignment_error <= 0.08
    ):
        return "OK"

    if (
        strongest_support >= 2
        and detection.mean_alignment_error <= 0.18
    ):
        return "REVIEW"

    return "FAIL"


def avatar_boxes(
    detection: Detection,
    image_width: int,
    image_height: int,
) -> list[dict[str, object]]:
    # Petite marge pour englober complètement le cadre.
    full_width = detection.frame_width * 1.03
    full_height = detection.frame_height * 1.03

    boxes: list[dict[str, object]] = []

    for side, center_x in (
        ("L", detection.left_center_x),
        ("R", detection.right_center_x),
    ):
        for slot, center_y in enumerate(
            detection.row_centers_y,
            start=1,
        ):
            x1 = max(
                0,
                int(round(center_x - full_width / 2)),
            )
            y1 = max(
                0,
                int(round(center_y - full_height / 2)),
            )
            x2 = min(
                image_width,
                int(round(center_x + full_width / 2)),
            )
            y2 = min(
                image_height,
                int(round(center_y + full_height / 2)),
            )

            width = x2 - x1
            height = y2 - y1

            # Zone interne : retire le niveau, le cadre et les étoiles.
            inner_box = (
                int(round(x1 + 0.07 * width)),
                int(round(y1 + 0.18 * height)),
                int(round(x2 - 0.07 * width)),
                int(round(y2 - 0.12 * height)),
            )

            # Zone du nom, calculée relativement au cadre détecté.
            if side == "L":
                name_x1 = int(round(x2 + 0.10 * width))
                name_x2 = int(round(x2 + 2.40 * width))
            else:
                name_x1 = int(round(x1 - 2.40 * width))
                name_x2 = int(round(x1 - 0.10 * width))

            name_box = (
                max(0, name_x1),
                max(0, int(round(y1 + 0.02 * height))),
                min(image_width, name_x2),
                min(
                    image_height,
                    int(round(y1 + 0.32 * height)),
                ),
            )

            boxes.append(
                {
                    "side": side,
                    "slot": slot,
                    "full": (x1, y1, x2, y2),
                    "inner": inner_box,
                    "name": name_box,
                }
            )

    return boxes


def create_debug_image(
    source_path: Path,
    detection: Detection,
    status: str,
) -> tuple[Path, str]:
    with Image.open(source_path) as source:
        image = source.convert("RGB")

    image_width, image_height = image.size
    draw = ImageDraw.Draw(image)

    line_width = max(2, image_width // 500)

    for item in avatar_boxes(
        detection=detection,
        image_width=image_width,
        image_height=image_height,
    ):
        full_box = item["full"]
        inner_box = item["inner"]
        name_box = item["name"]
        label = f"{item['side']}{item['slot']}"

        draw.rectangle(
            full_box,
            outline="red",
            width=line_width,
        )
        draw.rectangle(
            inner_box,
            outline="orange",
            width=line_width,
        )
        draw.rectangle(
            name_box,
            outline="yellow",
            width=line_width,
        )
        draw.text(
            (
                full_box[0],
                max(0, full_box[1] - 22),
            ),
            label,
            fill="yellow",
            stroke_width=2,
            stroke_fill="black",
        )

    draw.text(
        (20, 20),
        (
            f"status={status} | "
            f"support L/R="
            f"{detection.left_support}/"
            f"{detection.right_support} | "
            f"error={detection.mean_alignment_error:.3f}"
        ),
        fill="lime" if status == "OK" else "yellow",
        stroke_width=2,
        stroke_fill="black",
    )

    if image.width > MAX_DEBUG_WIDTH:
        resized_height = round(
            image.height * MAX_DEBUG_WIDTH / image.width
        )
        image = image.resize(
            (MAX_DEBUG_WIDTH, resized_height),
            Image.Resampling.LANCZOS,
        )

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    debug_path = (
        DEBUG_DIR
        / f"{source_path.stem}__frame_detection.jpg"
    )

    image.save(
        debug_path,
        format="JPEG",
        quality=90,
        optimize=True,
    )

    label = (
        f"{source_path.name}\n"
        f"{image_width} × {image_height} | "
        f"{status} | "
        f"L/R={detection.left_support}/"
        f"{detection.right_support} | "
        f"erreur={detection.mean_alignment_error:.3f}"
    )

    return debug_path, label


def create_contact_sheet(
    items: list[tuple[Path, str]],
    sheet_number: int,
    total_sheets: int,
) -> Path:
    rows = math.ceil(len(items) / SHEET_COLUMNS)

    card_width = 430
    preview_height = 205
    label_height = 58
    card_height = preview_height + label_height
    margin = 14
    title_height = 48

    sheet_width = (
        SHEET_COLUMNS * card_width
        + (SHEET_COLUMNS + 1) * margin
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
            "Détection dynamique des cadres — "
            f"planche {sheet_number}/{total_sheets}"
        ),
        fill="black",
    )

    for index, (debug_path, label) in enumerate(items):
        column = index % SHEET_COLUMNS
        row = index // SHEET_COLUMNS

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
            with Image.open(debug_path) as source:
                preview = ImageOps.contain(
                    source.convert("RGB"),
                    (
                        card_width - 12,
                        preview_height - 12,
                    ),
                    method=Image.Resampling.LANCZOS,
                )

            preview_x = (
                card_x + (card_width - preview.width) // 2
            )
            preview_y = (
                card_y
                + (preview_height - preview.height) // 2
            )

            sheet.paste(preview, (preview_x, preview_y))

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR
        / f"frame_detection_sheet_{sheet_number:02d}.jpg"
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    debug_items: list[tuple[Path, str]] = []
    manifest_rows: list[dict[str, object]] = []

    status_counts = {
        "OK": 0,
        "REVIEW": 0,
        "FAIL": 0,
    }

    print(f"Images à analyser : {len(image_paths)}")
    print()

    for index, source_path in enumerate(image_paths, start=1):
        try:
            image_bgr = cv2.imread(
                str(source_path),
                cv2.IMREAD_COLOR,
            )

            if image_bgr is None:
                raise RuntimeError(
                    "OpenCV ne peut pas ouvrir l'image."
                )

            image_height, image_width = image_bgr.shape[:2]
            detection = detect_avatar_grid(image_bgr)
            status = detection_status(detection)

            debug_path, label = create_debug_image(
                source_path=source_path,
                detection=detection,
                status=status,
            )

            debug_items.append((debug_path, label))
            status_counts[status] += 1

            manifest_rows.append(
                {
                    "source_filename": source_path.name,
                    "source_width": image_width,
                    "source_height": image_height,
                    "status": status,
                    "left_center_x": (
                        f"{detection.left_center_x:.3f}"
                    ),
                    "right_center_x": (
                        f"{detection.right_center_x:.3f}"
                    ),
                    "row_1_center_y": (
                        f"{detection.row_centers_y[0]:.3f}"
                    ),
                    "row_2_center_y": (
                        f"{detection.row_centers_y[1]:.3f}"
                    ),
                    "row_3_center_y": (
                        f"{detection.row_centers_y[2]:.3f}"
                    ),
                    "row_4_center_y": (
                        f"{detection.row_centers_y[3]:.3f}"
                    ),
                    "row_5_center_y": (
                        f"{detection.row_centers_y[4]:.3f}"
                    ),
                    "frame_width": (
                        f"{detection.frame_width:.3f}"
                    ),
                    "frame_height": (
                        f"{detection.frame_height:.3f}"
                    ),
                    "row_step": (
                        f"{detection.row_step:.3f}"
                    ),
                    "mean_alignment_error": (
                        f"{detection.mean_alignment_error:.6f}"
                    ),
                    "left_support": detection.left_support,
                    "right_support": detection.right_support,
                    "candidate_count": detection.candidate_count,
                    "debug_file": debug_path.as_posix(),
                    "error": "",
                }
            )

            print(
                f"[{index:03}/{len(image_paths):03}] "
                f"{status:<6} | "
                f"L/R={detection.left_support}/"
                f"{detection.right_support} | "
                f"erreur="
                f"{detection.mean_alignment_error:.3f} | "
                f"{source_path.name}"
            )

        except (
            RuntimeError,
            ValueError,
            OSError,
            UnidentifiedImageError,
        ) as error:
            status_counts["FAIL"] += 1

            manifest_rows.append(
                {
                    "source_filename": source_path.name,
                    "source_width": "",
                    "source_height": "",
                    "status": "FAIL",
                    "left_center_x": "",
                    "right_center_x": "",
                    "row_1_center_y": "",
                    "row_2_center_y": "",
                    "row_3_center_y": "",
                    "row_4_center_y": "",
                    "row_5_center_y": "",
                    "frame_width": "",
                    "frame_height": "",
                    "row_step": "",
                    "mean_alignment_error": "",
                    "left_support": "",
                    "right_support": "",
                    "candidate_count": "",
                    "debug_file": "",
                    "error": str(error),
                }
            )

            print(
                f"[{index:03}/{len(image_paths):03}] "
                f"FAIL   | {source_path.name} | {error}",
                file=sys.stderr,
            )

    fieldnames = [
        "source_filename",
        "source_width",
        "source_height",
        "status",
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
        "mean_alignment_error",
        "left_support",
        "right_support",
        "candidate_count",
        "debug_file",
        "error",
    ]

    with MANIFEST_PATH.open(
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
        writer.writerows(manifest_rows)

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

    for sheet_number, chunk in enumerate(chunks, start=1):
        output_path = create_contact_sheet(
            items=chunk,
            sheet_number=sheet_number,
            total_sheets=len(chunks),
        )
        print(f"[OK] {output_path}")

    print()
    print("Résumé :")
    print(f"- OK : {status_counts['OK']}")
    print(f"- REVIEW : {status_counts['REVIEW']}")
    print(f"- FAIL : {status_counts['FAIL']}")
    print(f"- Manifeste : {MANIFEST_PATH}")
    print(f"- Rapports : {OUTPUT_DIR}")

    return 0 if debug_items else 1


if __name__ == "__main__":
    raise SystemExit(main())
