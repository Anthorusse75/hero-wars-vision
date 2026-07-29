from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import easyocr
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError


METRICS = (
    "power",
    "damage_dealt",
    "damage_taken",
    "healing",
)

BATCH_PATTERN = re.compile(r"^hero_batch_\d{3}$")
DIGIT_ALLOWLIST = "0123456789 "

# Plafond de sécurité fondé sur les valeurs observées dans ce lot.
# Il sert uniquement à rejeter les faux chiffres ajoutés par les bordures.
POWER_MIN = 1_000
POWER_MAX = 500_000

HIGH_CONFIDENCE = 0.82
MEDIUM_CONFIDENCE = 0.58

# outer_trim retire la barre colorée située du côté extérieur.
# inner_trim retire les petits artefacts proches de l'icône centrale.
SPATIAL_VARIANTS = (
    ("dual_08_06", 0.08, 0.06),
    ("dual_16_09", 0.16, 0.09),
    ("dual_24_12", 0.24, 0.12),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relit les statistiques Hero Wars avec des découpes latérales "
            "qui retirent les barres et les artefacts proches du centre."
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
        help="Remplace un rapport stat_ocr_v2 existant.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force EasyOCR à utiliser le processeur.",
    )

    return parser.parse_args()


def read_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise RuntimeError(f"Fichier absent : {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(
            stream,
            delimiter=";",
        )
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(
            f"En-tête CSV absent : {path}"
        )

    return fieldnames, rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def normalized_digits(text: str) -> str:
    return "".join(
        character
        for character in text
        if character.isdigit()
    )


def parse_detections(
    detections: list[Any],
) -> tuple[str, str, float]:
    ordered: list[
        tuple[float, str, float]
    ] = []

    for detection in detections:
        if len(detection) < 3:
            continue

        box, text, confidence = detection[:3]

        try:
            x_position = min(
                float(point[0])
                for point in box
            )
            confidence_value = float(
                confidence
            )
        except (
            TypeError,
            ValueError,
            IndexError,
        ):
            continue

        ordered.append(
            (
                x_position,
                str(text),
                confidence_value,
            )
        )

    ordered.sort(
        key=lambda item: item[0]
    )

    raw_text = " ".join(
        item[1]
        for item in ordered
    ).strip()

    digits = normalized_digits(
        raw_text
    )

    if not ordered:
        return "", "", 0.0

    weights = [
        max(
            1,
            len(
                normalized_digits(
                    item[1]
                )
            ),
        )
        for item in ordered
    ]

    confidence = sum(
        item[2] * weight
        for item, weight in zip(
            ordered,
            weights,
        )
    ) / sum(weights)

    return raw_text, digits, confidence


def crop_number_band(
    image: Image.Image,
    side: str,
    outer_trim: float,
    inner_trim: float,
) -> Image.Image:
    width, height = image.size

    if side == "L":
        x1 = int(round(width * outer_trim))
        x2 = int(
            round(
                width * (1.0 - inner_trim)
            )
        )
    else:
        x1 = int(round(width * inner_trim))
        x2 = int(
            round(
                width * (1.0 - outer_trim)
            )
        )

    x1 = max(
        0,
        min(width - 2, x1),
    )
    x2 = max(
        x1 + 2,
        min(width, x2),
    )

    return image.crop(
        (x1, 0, x2, height)
    )


def grayscale_variant(
    image: Image.Image,
) -> np.ndarray:
    grayscale = ImageOps.grayscale(
        image
    )
    grayscale = ImageOps.autocontrast(
        grayscale,
        cutoff=1,
    )
    grayscale = ImageEnhance.Contrast(
        grayscale
    ).enhance(1.45)

    resized = grayscale.resize(
        (
            max(1, grayscale.width * 4),
            max(1, grayscale.height * 4),
        ),
        Image.Resampling.LANCZOS,
    )

    return np.asarray(resized)


def threshold_variant(
    image: Image.Image,
) -> np.ndarray:
    grayscale = ImageOps.grayscale(
        image
    )
    grayscale = ImageOps.autocontrast(
        grayscale,
        cutoff=1,
    )

    resized = grayscale.resize(
        (
            max(1, grayscale.width * 4),
            max(1, grayscale.height * 4),
        ),
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(resized)
    threshold = max(
        128,
        int(
            np.percentile(
                array,
                70,
            )
        ),
    )

    return np.where(
        array >= threshold,
        255,
        0,
    ).astype(np.uint8)


def read_variant(
    reader: easyocr.Reader,
    image_array: np.ndarray,
) -> tuple[str, str, float]:
    detections = reader.readtext(
        image_array,
        detail=1,
        paragraph=False,
        decoder="greedy",
        allowlist=DIGIT_ALLOWLIST,
        text_threshold=0.30,
        low_text=0.15,
        link_threshold=0.18,
        contrast_ths=0.05,
        adjust_contrast=0.75,
        add_margin=0.02,
    )

    return parse_detections(
        detections
    )


def plausible_candidate(
    metric: str,
    digits: str,
) -> bool:
    if not digits:
        return False

    if len(digits) > 9:
        return False

    value = int(digits)

    if metric == "power":
        return (
            POWER_MIN
            <= value
            <= POWER_MAX
            and 4 <= len(digits) <= 6
        )

    return 0 <= value <= 99_999_999


def artifact_relation(
    longer: str,
    shorter: str,
    side: str,
) -> bool:
    difference = len(longer) - len(shorter)

    if difference not in (1, 2):
        return False

    # L : l'artefact central est généralement ajouté à droite.
    # R : l'artefact central est généralement ajouté à gauche.
    if side == "L":
        return longer.startswith(
            shorter
        )

    return longer.endswith(
        shorter
    )


def select_candidate(
    metric: str,
    side: str,
    attempts: list[dict[str, Any]],
    old_digits: str,
    old_confidence: float,
) -> dict[str, Any]:
    valid_attempts = [
        attempt
        for attempt in attempts
        if plausible_candidate(
            metric,
            str(attempt["digits"]),
        )
    ]

    old_is_plausible = (
        plausible_candidate(
            metric,
            old_digits,
        )
    )

    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for attempt in valid_attempts:
        groups[
            str(attempt["digits"])
        ].append(attempt)

    if (
        old_is_plausible
        and old_digits not in groups
    ):
        groups[old_digits] = []

    if not groups:
        return {
            "digits": "",
            "value": "",
            "confidence": 0.0,
            "votes": 0,
            "selected_variant": "",
            "old_agreement": 0,
            "score": 0.0,
        }

    scores: dict[str, float] = {}

    for digits, group in groups.items():
        confidences = [
            float(item["confidence"])
            for item in group
        ]

        votes = len(group)
        max_confidence = (
            max(confidences)
            if confidences
            else 0.0
        )
        mean_confidence = (
            sum(confidences)
            / len(confidences)
            if confidences
            else 0.0
        )

        old_agreement = int(
            old_is_plausible
            and digits == old_digits
        )

        score = (
            max_confidence
            + 0.34 * votes
            + 0.08 * mean_confidence
            + 0.12 * old_agreement
        )

        if metric == "power":
            score += 0.12

        scores[digits] = score

    candidates = list(groups)

    for longer in candidates:
        for shorter in candidates:
            if len(longer) <= len(shorter):
                continue

            if artifact_relation(
                longer,
                shorter,
                side,
            ):
                scores[longer] -= 0.45
                scores[shorter] += 0.18

    selected_digits = max(
        scores,
        key=scores.get,
    )
    selected_group = groups[
        selected_digits
    ]

    if selected_group:
        best_attempt = max(
            selected_group,
            key=lambda item: float(
                item["confidence"]
            ),
        )
        confidence = max(
            float(item["confidence"])
            for item in selected_group
        )
        selected_variant = str(
            best_attempt["variant"]
        )
    else:
        confidence = old_confidence
        selected_variant = "old_v1_fallback"

    return {
        "digits": selected_digits,
        "value": int(selected_digits),
        "confidence": confidence,
        "votes": len(selected_group),
        "selected_variant": selected_variant,
        "old_agreement": int(
            selected_digits == old_digits
        ),
        "score": scores[
            selected_digits
        ],
    }


def classify_status(
    result: dict[str, Any],
    slot_status: str,
) -> str:
    if slot_status == "EMPTY":
        return "EMPTY_SLOT"

    if not result["digits"]:
        return "NO_DETECTION"

    votes = int(
        result["votes"]
    )
    confidence = float(
        result["confidence"]
    )
    old_agreement = int(
        result["old_agreement"]
    )

    if (
        votes >= 2
        and confidence >= 0.60
    ):
        return "HIGH"

    if (
        votes >= 1
        and old_agreement
        and confidence >= 0.72
    ):
        return "HIGH"

    if confidence >= HIGH_CONFIDENCE:
        return "HIGH"

    if confidence >= MEDIUM_CONFIDENCE:
        return "MEDIUM"

    return "LOW"


def build_slot_index(
    path: Path,
) -> dict[
    tuple[str, str, str],
    dict[str, str],
]:
    if not path.exists():
        return {}

    _, rows = read_csv(path)

    return {
        (
            row.get(
                "screenshot_id",
                "",
            ),
            row.get("side", ""),
            row.get("slot", ""),
        ): row
        for row in rows
    }


def image_data_uri(
    path: Path,
) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail(
            (440, 110),
            Image.Resampling.LANCZOS,
        )

        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=88,
        )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


def write_review_html(
    path: Path,
    rows: list[dict[str, Any]],
    crop_root: Path,
) -> None:
    review_rows = [
        row
        for row in rows
        if (
            row["status"]
            not in {
                "HIGH",
                "EMPTY_SLOT",
            }
            or row["changed_from_v1"]
            == "1"
        )
    ]

    parts = [
        "<!DOCTYPE html>",
        '<html lang="fr">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Contrôle OCR statistiques V2</title>",
        """
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #bbb; padding: 6px; vertical-align: middle; }
        th { background: #eee; position: sticky; top: 0; }
        img { max-width: 440px; }
        .changed { background: #e6f2ff; }
        .LOW, .NO_DETECTION { background: #ffd9d9; }
        .MEDIUM { background: #fff2bf; }
        </style>
        """,
        "</head>",
        "<body>",
        "<h1>Contrôle OCR statistiques V2</h1>",
        (
            f"<p>Cas affichés : "
            f"{len(review_rows)}</p>"
        ),
        "<table>",
        (
            "<tr><th>Emplacement</th><th>Métrique</th><th>Découpe</th>"
            "<th>V1</th><th>V2</th><th>Confiance</th>"
            "<th>Votes</th><th>Statut</th><th>Variante</th></tr>"
        ),
    ]

    for row in review_rows:
        crop_path = (
            crop_root
            / str(row["crop_file"])
        )

        try:
            uri = image_data_uri(
                crop_path
            )
            image_html = (
                f'<img src="{uri}" alt="">'
            )
        except (
            OSError,
            UnidentifiedImageError,
        ):
            image_html = html.escape(
                str(crop_path)
            )

        classes = [
            str(row["status"])
        ]

        if row["changed_from_v1"] == "1":
            classes.append("changed")

        parts.append(
            f'<tr class="{" ".join(classes)}">'
            f"<td>{html.escape(str(row['screenshot_id']))} "
            f"{html.escape(str(row['side']))}"
            f"{html.escape(str(row['slot']))}</td>"
            f"<td>{html.escape(str(row['metric']))}</td>"
            f"<td>{image_html}</td>"
            f"<td>{html.escape(str(row['old_value']))}</td>"
            f"<td>{html.escape(str(row['value']))}</td>"
            f"<td>{float(row['confidence']):.4f}</td>"
            f"<td>{html.escape(str(row['votes']))}</td>"
            f"<td>{html.escape(str(row['status']))}</td>"
            f"<td>{html.escape(str(row['selected_variant']))}</td>"
            "</tr>"
        )

    parts.extend(
        [
            "</table>",
            "</body>",
            "</html>",
        ]
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "\n".join(parts),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    if not BATCH_PATTERN.fullmatch(
        args.batch
    ):
        print(
            "--batch doit avoir la forme "
            "hero_batch_002.",
            file=sys.stderr,
        )
        return 2

    batch_root = (
        Path("data/batches")
        / args.batch
    )
    crop_root = (
        batch_root
        / "stat_crops_v2"
    )
    crop_manifest = (
        crop_root
        / "stat_crop_manifest.csv"
    )
    old_csv = (
        batch_root
        / "reports"
        / "stat_ocr_v1"
        / "stat_values_ocr.csv"
    )
    slot_manifest = (
        batch_root
        / "validated"
        / "slot_identity_manifest.csv"
    )

    output_root = (
        batch_root
        / "reports"
        / "stat_ocr_v2"
    )
    detailed_csv = (
        output_root
        / "stat_values_ocr.csv"
    )
    combined_csv = (
        output_root
        / "battle_stats_ocr.csv"
    )
    changed_csv = (
        output_root
        / "changed_values.csv"
    )
    review_html = (
        output_root
        / "stat_ocr_review.html"
    )

    if output_root.exists():
        if not args.overwrite:
            print(
                f"Le rapport existe déjà : "
                f"{output_root}",
                file=sys.stderr,
            )
            print(
                "Relance avec --overwrite "
                "pour le remplacer.",
                file=sys.stderr,
            )
            return 1

        shutil.rmtree(
            output_root
        )

    try:
        _, crop_rows = read_csv(
            crop_manifest
        )
        _, old_rows = read_csv(
            old_csv
        )
        slot_index = build_slot_index(
            slot_manifest
        )
    except (
        RuntimeError,
        csv.Error,
    ) as error:
        print(
            f"Erreur : {error}",
            file=sys.stderr,
        )
        return 1

    old_index = {
        (
            row.get(
                "screenshot_id",
                "",
            ),
            row.get("side", ""),
            row.get("slot", ""),
            row.get("metric", ""),
        ): row
        for row in old_rows
    }

    use_gpu = bool(
        torch.cuda.is_available()
        and not args.cpu
    )

    print(f"Batch : {args.batch}")
    print(
        f"Découpes à relire : "
        f"{len(crop_rows)}"
    )
    print(
        "Appareil EasyOCR : "
        f"{'cuda' if use_gpu else 'cpu'}"
    )

    if use_gpu:
        print(
            "GPU : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()
    print(
        "Initialisation d'EasyOCR..."
    )

    reader = easyocr.Reader(
        ["en"],
        gpu=use_gpu,
        verbose=False,
    )

    output_rows: list[
        dict[str, Any]
    ] = []
    status_counts: Counter[str] = (
        Counter()
    )
    metric_counts: dict[
        str,
        Counter[str],
    ] = {
        metric: Counter()
        for metric in METRICS
    }

    changed_counts: Counter[str] = (
        Counter()
    )

    start_time = (
        time.perf_counter()
    )

    for index, row in enumerate(
        crop_rows,
        start=1,
    ):
        screenshot_id = str(
            row.get("screenshot_id") or ""
        )
        side = str(
            row.get("side") or ""
        )
        slot = str(
            row.get("slot") or ""
        )
        metric = str(
            row.get("metric") or ""
        )
        crop_file = str(
            row.get("crop_file") or ""
        )
        crop_path = (
            crop_root / crop_file
        )

        slot_data = slot_index.get(
            (
                screenshot_id,
                side,
                slot,
            ),
            {},
        )
        slot_status = str(
            slot_data.get(
                "slot_status",
                "HERO",
            )
            or "HERO"
        )

        old_row = old_index.get(
            (
                screenshot_id,
                side,
                slot,
                metric,
            ),
            {},
        )
        old_digits = str(
            old_row.get(
                "normalized_digits",
                "",
            )
            or ""
        )
        old_value = str(
            old_row.get(
                "value",
                "",
            )
            or ""
        )

        try:
            old_confidence = float(
                old_row.get(
                    "confidence",
                    0.0,
                )
                or 0.0
            )
        except ValueError:
            old_confidence = 0.0

        attempts: list[
            dict[str, Any]
        ] = []

        try:
            if slot_status == "EMPTY":
                result = {
                    "digits": "",
                    "value": "",
                    "confidence": 1.0,
                    "votes": 0,
                    "selected_variant": "",
                    "old_agreement": 0,
                    "score": 0.0,
                }
            else:
                with Image.open(
                    crop_path
                ) as source:
                    image = source.convert(
                        "RGB"
                    )

                for (
                    variant_name,
                    outer_trim,
                    inner_trim,
                ) in SPATIAL_VARIANTS:
                    band = crop_number_band(
                        image=image,
                        side=side,
                        outer_trim=outer_trim,
                        inner_trim=inner_trim,
                    )

                    (
                        raw_text,
                        digits,
                        confidence,
                    ) = read_variant(
                        reader,
                        grayscale_variant(
                            band
                        ),
                    )

                    attempts.append(
                        {
                            "variant": (
                                variant_name
                            ),
                            "raw_text": raw_text,
                            "digits": digits,
                            "confidence": confidence,
                        }
                    )

                unique_plausible = {
                    str(attempt["digits"])
                    for attempt in attempts
                    if plausible_candidate(
                        metric,
                        str(
                            attempt[
                                "digits"
                            ]
                        ),
                    )
                }

                if len(unique_plausible) != 1:
                    band = crop_number_band(
                        image=image,
                        side=side,
                        outer_trim=0.16,
                        inner_trim=0.09,
                    )

                    (
                        raw_text,
                        digits,
                        confidence,
                    ) = read_variant(
                        reader,
                        threshold_variant(
                            band
                        ),
                    )

                    attempts.append(
                        {
                            "variant": (
                                "dual_16_09_threshold"
                            ),
                            "raw_text": raw_text,
                            "digits": digits,
                            "confidence": confidence,
                        }
                    )

                result = select_candidate(
                    metric=metric,
                    side=side,
                    attempts=attempts,
                    old_digits=old_digits,
                    old_confidence=old_confidence,
                )

            status = classify_status(
                result=result,
                slot_status=slot_status,
            )

        except (
            OSError,
            ValueError,
            TypeError,
            UnidentifiedImageError,
        ) as error:
            result = {
                "digits": "",
                "value": "",
                "confidence": 0.0,
                "votes": 0,
                "selected_variant": "",
                "old_agreement": 0,
                "score": 0.0,
            }
            status = "ERROR"
            print(
                f"[ERREUR] {crop_file} : "
                f"{error}",
                file=sys.stderr,
            )

        new_value = str(
            result["value"]
        )
        changed = int(
            new_value != old_value
            and slot_status != "EMPTY"
        )

        output_row = {
            "screenshot_id": screenshot_id,
            "side": side,
            "slot": slot,
            "slot_status": slot_status,
            "hero_uid": slot_data.get(
                "final_hero_uid",
                "",
            ),
            "hero_name": slot_data.get(
                "final_hero_name",
                "",
            ),
            "metric": metric,
            "crop_file": crop_file,
            "old_value": old_value,
            "old_confidence": (
                f"{old_confidence:.6f}"
            ),
            "normalized_digits": (
                result["digits"]
            ),
            "value": result["value"],
            "confidence": (
                f"{float(result['confidence']):.6f}"
            ),
            "votes": result["votes"],
            "old_agreement": (
                result["old_agreement"]
            ),
            "selected_variant": (
                result["selected_variant"]
            ),
            "candidate_attempts": (
                json.dumps(
                    attempts,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
            "score": (
                f"{float(result['score']):.6f}"
            ),
            "status": status,
            "changed_from_v1": str(
                changed
            ),
        }

        output_rows.append(
            output_row
        )
        status_counts[status] += 1
        metric_counts[
            metric
        ][status] += 1

        if changed:
            changed_counts[
                metric
            ] += 1

        if (
            index % 100 == 0
            or index == len(crop_rows)
        ):
            print(
                f"[{index:04d}/"
                f"{len(crop_rows):04d}] "
                f"HIGH={status_counts['HIGH']} | "
                f"MEDIUM={status_counts['MEDIUM']} | "
                f"LOW={status_counts['LOW']} | "
                f"CHANGED={sum(changed_counts.values())}"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    detailed_fields = [
        "screenshot_id",
        "side",
        "slot",
        "slot_status",
        "hero_uid",
        "hero_name",
        "metric",
        "crop_file",
        "old_value",
        "old_confidence",
        "normalized_digits",
        "value",
        "confidence",
        "votes",
        "old_agreement",
        "selected_variant",
        "candidate_attempts",
        "score",
        "status",
        "changed_from_v1",
    ]

    write_csv(
        detailed_csv,
        output_rows,
        detailed_fields,
    )

    changed_rows = [
        row
        for row in output_rows
        if row["changed_from_v1"]
        == "1"
    ]

    write_csv(
        changed_csv,
        changed_rows,
        detailed_fields,
    )

    by_slot: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    for row in output_rows:
        key = (
            str(row["screenshot_id"]),
            str(row["side"]),
            str(row["slot"]),
        )

        combined = by_slot.setdefault(
            key,
            {
                "screenshot_id": (
                    row["screenshot_id"]
                ),
                "side": row["side"],
                "slot": row["slot"],
                "slot_status": (
                    row["slot_status"]
                ),
                "hero_uid": (
                    row["hero_uid"]
                ),
                "hero_name": (
                    row["hero_name"]
                ),
            },
        )

        metric = str(
            row["metric"]
        )

        combined[metric] = (
            row["value"]
        )
        combined[
            f"{metric}_confidence"
        ] = row["confidence"]
        combined[
            f"{metric}_status"
        ] = row["status"]

    combined_rows = sorted(
        by_slot.values(),
        key=lambda row: (
            int(
                str(
                    row["screenshot_id"]
                )
                or 0
            ),
            str(row["side"]),
            int(
                str(row["slot"])
                or 0
            ),
        ),
    )

    combined_fields = [
        "screenshot_id",
        "side",
        "slot",
        "slot_status",
        "hero_uid",
        "hero_name",
    ]

    for metric in METRICS:
        combined_fields.extend(
            [
                metric,
                f"{metric}_confidence",
                f"{metric}_status",
            ]
        )

    write_csv(
        combined_csv,
        combined_rows,
        combined_fields,
    )

    write_review_html(
        review_html,
        output_rows,
        crop_root,
    )

    print()
    print("Résumé global V2 :")

    for status in (
        "HIGH",
        "MEDIUM",
        "LOW",
        "NO_DETECTION",
        "EMPTY_SLOT",
        "ERROR",
    ):
        print(
            f"- {status:<12} : "
            f"{status_counts[status]}"
        )

    print()
    print(
        "Valeurs modifiées par rapport "
        "à la V1 :"
    )

    for metric in METRICS:
        print(
            f"- {metric:<14} : "
            f"{changed_counts[metric]}"
        )

    print()
    print(
        "Valeurs power hors plage "
        f"{POWER_MIN:,}–{POWER_MAX:,} : "
        f"{sum(1 for row in output_rows if row['metric'] == 'power' and row['status'] not in {'EMPTY_SLOT', 'ERROR'} and not plausible_candidate('power', str(row['normalized_digits'])))}"
    )

    print()
    print(
        f"Durée totale : "
        f"{elapsed:.1f} secondes"
    )
    print(
        f"Moyenne : "
        f"{elapsed / max(1, len(crop_rows)):.3f} "
        "seconde/découpe"
    )
    print()
    print(
        f"Valeurs détaillées : "
        f"{detailed_csv}"
    )
    print(
        f"Valeurs modifiées : "
        f"{changed_csv}"
    )
    print(
        f"Table consolidée : "
        f"{combined_csv}"
    )
    print(
        f"Contrôle visuel : "
        f"{review_html}"
    )

    return (
        0
        if status_counts["ERROR"] == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
