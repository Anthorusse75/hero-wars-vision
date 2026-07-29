from __future__ import annotations

import argparse
import base64
import csv
import html
import math
import re
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

PRIMARY_ACCEPT_CONFIDENCE = 0.82
HIGH_CONFIDENCE = 0.82
MEDIUM_CONFIDENCE = 0.58


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lit par OCR les valeurs numériques extraites des écrans "
            "de statistiques Hero Wars."
        )
    )

    parser.add_argument(
        "--batch",
        required=True,
        help="Nom du lot, par exemple hero_batch_002.",
    )

    parser.add_argument(
        "--input-version",
        default="v2",
        choices=("v1", "v2"),
        help="Version des découpes numériques à utiliser (défaut : v2).",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remplace un rapport OCR existant.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force EasyOCR à utiliser le processeur.",
    )

    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise RuntimeError(f"Fichier absent : {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream, delimiter=";")
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not fieldnames:
        raise RuntimeError(f"En-tête CSV absent : {path}")

    return fieldnames, rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
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
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def normalized_digits(text: str) -> str:
    return "".join(character for character in text if character.isdigit())


def parse_detections(
    detections: list[Any],
) -> tuple[str, str, float]:
    ordered: list[tuple[float, str, float]] = []

    for detection in detections:
        if len(detection) < 3:
            continue

        box, text, confidence = detection[:3]

        try:
            x_position = min(float(point[0]) for point in box)
            confidence_value = float(confidence)
        except (TypeError, ValueError, IndexError):
            continue

        ordered.append(
            (
                x_position,
                str(text),
                confidence_value,
            )
        )

    ordered.sort(key=lambda item: item[0])

    raw_text = " ".join(item[1] for item in ordered).strip()
    digits = normalized_digits(raw_text)

    if not ordered:
        return "", "", 0.0

    digit_weights = [
        max(1, len(normalized_digits(item[1])))
        for item in ordered
    ]
    weighted_confidence = sum(
        item[2] * weight
        for item, weight in zip(ordered, digit_weights)
    ) / sum(digit_weights)

    return raw_text, digits, weighted_confidence


def resize_for_ocr(image: Image.Image, factor: int = 3) -> Image.Image:
    return image.resize(
        (
            max(1, image.width * factor),
            max(1, image.height * factor),
        ),
        Image.Resampling.LANCZOS,
    )


def primary_variant(image: Image.Image) -> np.ndarray:
    grayscale = ImageOps.grayscale(image)
    grayscale = ImageOps.autocontrast(grayscale, cutoff=1)
    grayscale = ImageEnhance.Contrast(grayscale).enhance(1.35)
    return np.asarray(resize_for_ocr(grayscale))


def side_trim_variant(image: Image.Image, side: str) -> np.ndarray:
    width, height = image.size
    trim = int(round(width * 0.10))

    if side == "L":
        cropped = image.crop((trim, 0, width, height))
    else:
        cropped = image.crop((0, 0, width - trim, height))

    grayscale = ImageOps.grayscale(cropped)
    grayscale = ImageOps.autocontrast(grayscale, cutoff=1)
    grayscale = ImageEnhance.Contrast(grayscale).enhance(1.55)

    return np.asarray(resize_for_ocr(grayscale))


def threshold_variant(image: Image.Image) -> np.ndarray:
    grayscale = ImageOps.grayscale(image)
    grayscale = ImageOps.autocontrast(grayscale, cutoff=1)
    resized = resize_for_ocr(grayscale)

    array = np.asarray(resized)
    threshold = max(125, int(np.percentile(array, 72)))
    binary = np.where(array >= threshold, 255, 0).astype(np.uint8)

    return binary


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
        text_threshold=0.35,
        low_text=0.20,
        link_threshold=0.20,
        contrast_ths=0.05,
        adjust_contrast=0.70,
        add_margin=0.03,
    )

    return parse_detections(detections)


def candidate_is_plausible(digits: str) -> bool:
    if not digits:
        return False

    if len(digits) > 9:
        return False

    return True


def choose_candidate(
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_attempts = [
        attempt
        for attempt in attempts
        if candidate_is_plausible(str(attempt["digits"]))
    ]

    if not valid_attempts:
        return {
            "variant": "",
            "raw_text": "",
            "digits": "",
            "value": "",
            "confidence": 0.0,
            "votes": 0,
        }

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for attempt in valid_attempts:
        groups[str(attempt["digits"])].append(attempt)

    ranked: list[tuple[float, str, list[dict[str, Any]]]] = []

    for digits, group in groups.items():
        best_confidence = max(float(item["confidence"]) for item in group)
        mean_confidence = sum(
            float(item["confidence"])
            for item in group
        ) / len(group)

        score = (
            best_confidence
            + 0.055 * (len(group) - 1)
            + 0.005 * min(len(digits), 7)
            + 0.010 * mean_confidence
        )

        ranked.append((score, digits, group))

    ranked.sort(reverse=True, key=lambda item: item[0])
    _, digits, winning_group = ranked[0]

    best_attempt = max(
        winning_group,
        key=lambda item: float(item["confidence"]),
    )

    return {
        "variant": best_attempt["variant"],
        "raw_text": best_attempt["raw_text"],
        "digits": digits,
        "value": int(digits),
        "confidence": max(
            float(item["confidence"])
            for item in winning_group
        ),
        "votes": len(winning_group),
    }


def classify_status(
    result: dict[str, Any],
    slot_status: str,
) -> str:
    if slot_status == "EMPTY":
        return "EMPTY_SLOT"

    if not result["digits"]:
        return "NO_DETECTION"

    confidence = float(result["confidence"])
    votes = int(result["votes"])

    if confidence >= 0.92:
        return "HIGH"

    if confidence >= HIGH_CONFIDENCE and votes >= 1:
        return "HIGH"

    if confidence >= 0.72 and votes >= 2:
        return "HIGH"

    if confidence >= MEDIUM_CONFIDENCE:
        return "MEDIUM"

    return "LOW"


def build_slot_index(
    slot_manifest_path: Path,
) -> dict[tuple[str, str, str], dict[str, str]]:
    if not slot_manifest_path.exists():
        return {}

    _, rows = read_csv(slot_manifest_path)

    return {
        (
            row.get("screenshot_id", ""),
            row.get("side", ""),
            row.get("slot", ""),
        ): row
        for row in rows
    }


def image_data_uri(path: Path) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((420, 100), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=88)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def write_review_html(
    path: Path,
    result_rows: list[dict[str, Any]],
    crop_root: Path,
) -> None:
    review_rows = [
        row
        for row in result_rows
        if row["status"] not in {"HIGH", "EMPTY_SLOT"}
    ]

    parts = [
        "<!DOCTYPE html>",
        '<html lang="fr">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Contrôle OCR des statistiques</title>",
        """
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #bbb; padding: 6px; vertical-align: middle; }
        th { background: #eee; position: sticky; top: 0; }
        img { max-width: 420px; image-rendering: auto; }
        .LOW, .NO_DETECTION { background: #ffd9d9; }
        .MEDIUM { background: #fff2bf; }
        </style>
        """,
        "</head>",
        "<body>",
        "<h1>Contrôle OCR des statistiques</h1>",
        f"<p>Cas à contrôler : {len(review_rows)}</p>",
        "<table>",
        (
            "<tr><th>Emplacement</th><th>Métrique</th><th>Découpe</th>"
            "<th>Valeur</th><th>Confiance</th><th>Statut</th>"
            "<th>Variante</th></tr>"
        ),
    ]

    for row in review_rows:
        crop_path = crop_root / str(row["crop_file"])

        try:
            uri = image_data_uri(crop_path)
            image_html = f'<img src="{uri}" alt="">'
        except (OSError, UnidentifiedImageError):
            image_html = html.escape(str(crop_path))

        parts.append(
            f'<tr class="{html.escape(str(row["status"]))}">'
            f"<td>{html.escape(str(row['screenshot_id']))} "
            f"{html.escape(str(row['side']))}{html.escape(str(row['slot']))}</td>"
            f"<td>{html.escape(str(row['metric']))}</td>"
            f"<td>{image_html}</td>"
            f"<td>{html.escape(str(row['value']))}</td>"
            f"<td>{float(row['confidence']):.4f}</td>"
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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    args = parse_args()

    if not BATCH_PATTERN.fullmatch(args.batch):
        print(
            "--batch doit avoir la forme hero_batch_002.",
            file=sys.stderr,
        )
        return 2

    batch_root = Path("data/batches") / args.batch
    crop_root = batch_root / f"stat_crops_{args.input_version}"
    crop_manifest = crop_root / "stat_crop_manifest.csv"
    slot_manifest = (
        batch_root
        / "validated"
        / "slot_identity_manifest.csv"
    )

    output_root = (
        batch_root
        / "reports"
        / "stat_ocr_v1"
    )
    detailed_csv = output_root / "stat_values_ocr.csv"
    combined_csv = output_root / "battle_stats_ocr.csv"
    review_html = output_root / "stat_ocr_review.html"

    if output_root.exists():
        if not args.overwrite:
            print(
                f"Le rapport existe déjà : {output_root}",
                file=sys.stderr,
            )
            print(
                "Relance avec --overwrite pour le remplacer.",
                file=sys.stderr,
            )
            return 1

        import shutil
        shutil.rmtree(output_root)

    try:
        _, crop_rows = read_csv(crop_manifest)
        slot_index = build_slot_index(slot_manifest)
    except (RuntimeError, csv.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    use_gpu = bool(torch.cuda.is_available() and not args.cpu)

    print(f"Batch : {args.batch}")
    print(f"Découpes à lire : {len(crop_rows)}")
    print(f"Appareil EasyOCR : {'cuda' if use_gpu else 'cpu'}")

    if use_gpu:
        print(f"GPU : {torch.cuda.get_device_name(0)}")

    print()
    print("Initialisation d'EasyOCR...")

    reader = easyocr.Reader(
        ["en"],
        gpu=use_gpu,
        verbose=False,
    )

    output_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    metric_status_counts: dict[str, Counter[str]] = {
        metric: Counter()
        for metric in METRICS
    }

    start_time = time.perf_counter()

    for index, row in enumerate(crop_rows, start=1):
        screenshot_id = str(row.get("screenshot_id") or "")
        side = str(row.get("side") or "")
        slot = str(row.get("slot") or "")
        metric = str(row.get("metric") or "")
        crop_file = str(row.get("crop_file") or "")
        crop_path = crop_root / crop_file

        slot_data = slot_index.get(
            (screenshot_id, side, slot),
            {},
        )
        slot_status = str(
            slot_data.get("slot_status") or "HERO"
        )

        attempts: list[dict[str, Any]] = []

        try:
            if slot_status == "EMPTY":
                result = {
                    "variant": "",
                    "raw_text": "",
                    "digits": "",
                    "value": "",
                    "confidence": 1.0,
                    "votes": 0,
                }
            else:
                with Image.open(crop_path) as source:
                    image = source.convert("RGB")

                raw_text, digits, confidence = read_variant(
                    reader,
                    primary_variant(image),
                )
                attempts.append(
                    {
                        "variant": "primary",
                        "raw_text": raw_text,
                        "digits": digits,
                        "confidence": confidence,
                    }
                )

                primary_good = (
                    candidate_is_plausible(digits)
                    and confidence >= PRIMARY_ACCEPT_CONFIDENCE
                )

                if not primary_good:
                    for variant_name, variant_image in (
                        (
                            "side_trim",
                            side_trim_variant(image, side),
                        ),
                        (
                            "threshold",
                            threshold_variant(image),
                        ),
                    ):
                        raw_text, digits, confidence = read_variant(
                            reader,
                            variant_image,
                        )
                        attempts.append(
                            {
                                "variant": variant_name,
                                "raw_text": raw_text,
                                "digits": digits,
                                "confidence": confidence,
                            }
                        )

                result = choose_candidate(attempts)

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
                "variant": "",
                "raw_text": "",
                "digits": "",
                "value": "",
                "confidence": 0.0,
                "votes": 0,
            }
            status = "ERROR"
            print(
                f"[ERREUR] {crop_file} : {error}",
                file=sys.stderr,
            )

        output_row = {
            "screenshot_id": screenshot_id,
            "side": side,
            "slot": slot,
            "slot_status": slot_status,
            "hero_uid": slot_data.get("final_hero_uid", ""),
            "hero_name": slot_data.get("final_hero_name", ""),
            "metric": metric,
            "crop_file": crop_file,
            "raw_text": result["raw_text"],
            "normalized_digits": result["digits"],
            "value": result["value"],
            "confidence": f"{float(result['confidence']):.6f}",
            "votes": result["votes"],
            "selected_variant": result["variant"],
            "status": status,
        }

        output_rows.append(output_row)
        status_counts[status] += 1
        metric_status_counts.setdefault(metric, Counter())[status] += 1

        if index % 100 == 0 or index == len(crop_rows):
            print(
                f"[{index:04d}/{len(crop_rows):04d}] "
                f"HIGH={status_counts['HIGH']} | "
                f"MEDIUM={status_counts['MEDIUM']} | "
                f"LOW={status_counts['LOW']} | "
                f"NO_DETECTION={status_counts['NO_DETECTION']}"
            )

    elapsed = time.perf_counter() - start_time

    detailed_fields = [
        "screenshot_id",
        "side",
        "slot",
        "slot_status",
        "hero_uid",
        "hero_name",
        "metric",
        "crop_file",
        "raw_text",
        "normalized_digits",
        "value",
        "confidence",
        "votes",
        "selected_variant",
        "status",
    ]

    write_csv(
        detailed_csv,
        output_rows,
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
                "screenshot_id": row["screenshot_id"],
                "side": row["side"],
                "slot": row["slot"],
                "slot_status": row["slot_status"],
                "hero_uid": row["hero_uid"],
                "hero_name": row["hero_name"],
            },
        )

        metric = str(row["metric"])
        combined[metric] = row["value"]
        combined[f"{metric}_confidence"] = row["confidence"]
        combined[f"{metric}_status"] = row["status"]

    combined_rows = sorted(
        by_slot.values(),
        key=lambda row: (
            int(str(row["screenshot_id"]) or 0),
            str(row["side"]),
            int(str(row["slot"]) or 0),
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
    print("Résumé global :")

    for status in (
        "HIGH",
        "MEDIUM",
        "LOW",
        "NO_DETECTION",
        "EMPTY_SLOT",
        "ERROR",
    ):
        print(
            f"- {status:<12} : {status_counts[status]}"
        )

    print()
    print("Résumé par métrique :")

    for metric in METRICS:
        counts = metric_status_counts[metric]
        print(
            f"- {metric:<14} : "
            f"HIGH={counts['HIGH']} | "
            f"MEDIUM={counts['MEDIUM']} | "
            f"LOW={counts['LOW']} | "
            f"NO_DETECTION={counts['NO_DETECTION']} | "
            f"EMPTY={counts['EMPTY_SLOT']} | "
            f"ERROR={counts['ERROR']}"
        )

    print()
    print(f"Durée totale : {elapsed:.1f} secondes")
    print(
        f"Moyenne : {elapsed / max(1, len(crop_rows)):.3f} "
        "seconde/découpe"
    )
    print()
    print(f"Valeurs détaillées : {detailed_csv}")
    print(f"Table consolidée : {combined_csv}")
    print(f"Contrôle visuel : {review_html}")

    return 0 if status_counts["ERROR"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
