from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, UnidentifiedImageError


METRICS = (
    "power",
    "damage_dealt",
    "damage_taken",
    "healing",
)

BATCH_PATTERN = re.compile(r"^hero_batch_\d{3}$")

POWER_MIN = 1_000
POWER_MAX = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Arbitre prudemment les résultats OCR V1/V2 des statistiques "
            "sans relancer EasyOCR."
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
        help="Remplace un résultat V3 déjà existant.",
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
        reader = csv.DictReader(stream, delimiter=";")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

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


def canonical_digits(value: str) -> str:
    digits = "".join(
        character
        for character in str(value or "")
        if character.isdigit()
    )

    if not digits:
        return ""

    return str(int(digits))


def plausible_candidate(
    metric: str,
    digits: str,
) -> bool:
    if not digits or not digits.isdigit():
        return False

    if len(digits) > 9:
        return False

    value = int(digits)

    if metric == "power":
        return (
            POWER_MIN <= value <= POWER_MAX
            and 4 <= len(digits) <= 6
        )

    return 0 <= value <= 99_999_999


def parse_attempts(
    row: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    raw = str(row.get("candidate_attempts") or "").strip()

    if not raw:
        return {}

    try:
        attempts = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for attempt in attempts:
        digits = canonical_digits(
            str(attempt.get("digits") or "")
        )

        if not plausible_candidate(
            str(row.get("metric") or ""),
            digits,
        ):
            continue

        groups[digits].append(attempt)

    return dict(groups)


def candidate_votes(
    groups: dict[str, list[dict[str, Any]]],
    digits: str,
) -> int:
    return len(groups.get(digits, []))


def candidate_max_confidence(
    groups: dict[str, list[dict[str, Any]]],
    digits: str,
) -> float:
    attempts = groups.get(digits, [])

    if not attempts:
        return 0.0

    return max(
        float(attempt.get("confidence") or 0.0)
        for attempt in attempts
    )


def arbitrate_row(
    row: dict[str, str],
) -> tuple[str, str, int]:
    slot_status = str(row.get("slot_status") or "")
    metric = str(row.get("metric") or "")

    if slot_status == "EMPTY":
        return "", "EMPTY_SLOT", 0

    old_value = canonical_digits(
        str(row.get("old_value") or "")
    )
    new_value = canonical_digits(
        str(row.get("value") or "")
    )

    if old_value == new_value and old_value:
        return old_value, "UNCHANGED_V1_V2", 0

    groups = parse_attempts(row)

    old_votes = candidate_votes(groups, old_value)
    new_votes = candidate_votes(groups, new_value)

    old_plausible = plausible_candidate(
        metric,
        old_value,
    )
    new_plausible = plausible_candidate(
        metric,
        new_value,
    )

    # Cas le plus sûr : la V1 produit une puissance impossible,
    # alors que la V2 revient dans la plage observée.
    if (
        metric == "power"
        and not old_plausible
        and new_plausible
    ):
        return new_value, "POWER_RANGE_FIX", 0

    if old_plausible and not new_plausible:
        return old_value, "RETAIN_V1_V2_INVALID", 0

    # Deux variantes spatiales ou plus relisent exactement la V1.
    if old_plausible and old_votes >= 2:
        return old_value, "RETAIN_V1_CONSENSUS", 0

    # Deux variantes ou plus relisent la V2 et aucune ne relit la V1.
    # Pour les substitutions de même longueur, on garde une revue humaine :
    # 0/9, 1/7 et 3/8 restent fréquents.
    if (
        new_plausible
        and new_votes >= 2
        and old_votes == 0
    ):
        if (
            old_plausible
            and len(old_value) == len(new_value)
        ):
            return "", "REVIEW_SAME_LENGTH_SUBSTITUTION", 1

        return new_value, "ACCEPT_V2_CONSENSUS", 0

    # Une valeur V1 plausible reste prioritaire lorsqu'aucun consensus
    # suffisamment fort ne justifie son remplacement.
    return "", "REVIEW_INSUFFICIENT_CONSENSUS", 1


def build_consolidated_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    by_slot: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    for row in rows:
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

        combined[metric] = row["final_value"]
        combined[f"{metric}_arbitration"] = row[
            "arbitration_reason"
        ]
        combined[f"{metric}_review_required"] = row[
            "review_required"
        ]

    rows_sorted = sorted(
        by_slot.values(),
        key=lambda row: (
            int(str(row["screenshot_id"]) or 0),
            str(row["side"]),
            int(str(row["slot"]) or 0),
        ),
    )

    fields = [
        "screenshot_id",
        "side",
        "slot",
        "slot_status",
        "hero_uid",
        "hero_name",
    ]

    for metric in METRICS:
        fields.extend(
            [
                metric,
                f"{metric}_arbitration",
                f"{metric}_review_required",
            ]
        )

    return fields, rows_sorted


def data_uri(path: Path) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail(
            (520, 130),
            Image.Resampling.LANCZOS,
        )

        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=90,
        )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


def write_review_html(
    path: Path,
    rows: list[dict[str, Any]],
    crop_root: Path,
) -> None:
    parts = [
        "<!DOCTYPE html>",
        '<html lang="fr">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Revue OCR statistiques V3</title>",
        """
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #bbb; padding: 7px; vertical-align: middle; }
        th { background: #eee; position: sticky; top: 0; }
        img { max-width: 520px; }
        .same { background: #fff2bf; }
        .consensus { background: #ffdada; }
        </style>
        """,
        "</head>",
        "<body>",
        "<h1>Revue OCR statistiques V3</h1>",
        f"<p>Cas restant à contrôler : {len(rows)}</p>",
        "<table>",
        (
            "<tr><th>Emplacement</th><th>Héros</th><th>Métrique</th>"
            "<th>Découpe</th><th>V1</th><th>V2</th>"
            "<th>Raison</th><th>Votes V1/V2</th></tr>"
        ),
    ]

    for row in rows:
        crop_path = crop_root / str(row["crop_file"])

        try:
            image_html = (
                f'<img src="{data_uri(crop_path)}" alt="">'
            )
        except (OSError, UnidentifiedImageError):
            image_html = html.escape(str(crop_path))

        css_class = (
            "same"
            if row["arbitration_reason"]
            == "REVIEW_SAME_LENGTH_SUBSTITUTION"
            else "consensus"
        )

        parts.append(
            f'<tr class="{css_class}">'
            f"<td>{html.escape(str(row['screenshot_id']))} "
            f"{html.escape(str(row['side']))}"
            f"{html.escape(str(row['slot']))}</td>"
            f"<td>{html.escape(str(row['hero_name']))}</td>"
            f"<td>{html.escape(str(row['metric']))}</td>"
            f"<td>{image_html}</td>"
            f"<td>{html.escape(str(row['old_value']))}</td>"
            f"<td>{html.escape(str(row['value']))}</td>"
            f"<td>{html.escape(str(row['arbitration_reason']))}</td>"
            f"<td>{row['v1_votes']} / {row['v2_votes']}</td>"
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
    path.write_text(
        "\n".join(parts),
        encoding="utf-8",
    )


def copy_review_assets(
    review_rows: list[dict[str, Any]],
    crop_root: Path,
    raw_root: Path,
    package_root: Path,
) -> None:
    crop_destination = package_root / "crops"
    screenshot_destination = package_root / "screenshots"

    crop_destination.mkdir(parents=True, exist_ok=True)
    screenshot_destination.mkdir(parents=True, exist_ok=True)

    screenshot_ids = {
        str(row["screenshot_id"])
        for row in review_rows
    }

    for row in review_rows:
        source = crop_root / str(row["crop_file"])

        if source.exists():
            shutil.copy2(
                source,
                crop_destination / source.name,
            )

    for screenshot_id in screenshot_ids:
        matches = sorted(
            raw_root.glob(f"{screenshot_id}_*")
        )

        for source in matches:
            shutil.copy2(
                source,
                screenshot_destination / source.name,
            )


def main() -> int:
    args = parse_args()

    if not BATCH_PATTERN.fullmatch(args.batch):
        print(
            "--batch doit avoir la forme hero_batch_002.",
            file=sys.stderr,
        )
        return 2

    batch_root = Path("data/batches") / args.batch
    source_csv = (
        batch_root
        / "reports"
        / "stat_ocr_v2"
        / "stat_values_ocr.csv"
    )
    crop_root = batch_root / "stat_crops_v2"
    raw_root = batch_root / "raw"

    output_root = (
        batch_root
        / "reports"
        / "stat_ocr_v3"
    )
    package_root = output_root / "review_package"

    detailed_csv = output_root / "stat_values_arbitrated.csv"
    combined_csv = output_root / "battle_stats_arbitrated.csv"
    review_csv = output_root / "review_cases.csv"
    review_html = output_root / "review.html"
    summary_path = output_root / "summary.txt"

    zip_path = Path(
        f"{args.batch}_stat_ocr_v3_review.zip"
    )

    if output_root.exists():
        if not args.overwrite:
            print(
                f"Le résultat existe déjà : {output_root}",
                file=sys.stderr,
            )
            print(
                "Relance avec --overwrite pour le remplacer.",
                file=sys.stderr,
            )
            return 1

        shutil.rmtree(output_root)

    if zip_path.exists():
        if not args.overwrite:
            print(
                f"L'archive existe déjà : {zip_path}",
                file=sys.stderr,
            )
            return 1

        zip_path.unlink()

    try:
        _, source_rows = read_csv(source_csv)
    except (RuntimeError, csv.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    output_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for row in source_rows:
        final_value, reason, review_required = arbitrate_row(row)
        groups = parse_attempts(row)

        old_value = canonical_digits(
            str(row.get("old_value") or "")
        )
        new_value = canonical_digits(
            str(row.get("value") or "")
        )

        output_row = dict(row)
        output_row["final_value"] = final_value
        output_row["arbitration_reason"] = reason
        output_row["review_required"] = str(review_required)
        output_row["v1_votes"] = str(
            candidate_votes(groups, old_value)
        )
        output_row["v2_votes"] = str(
            candidate_votes(groups, new_value)
        )
        output_row["v1_max_confidence"] = (
            f"{candidate_max_confidence(groups, old_value):.6f}"
        )
        output_row["v2_max_confidence"] = (
            f"{candidate_max_confidence(groups, new_value):.6f}"
        )

        output_rows.append(output_row)
        reason_counts[reason] += 1

    review_rows = [
        row
        for row in output_rows
        if row["review_required"] == "1"
    ]

    detailed_fields = list(output_rows[0].keys())

    write_csv(
        detailed_csv,
        output_rows,
        detailed_fields,
    )
    write_csv(
        review_csv,
        review_rows,
        detailed_fields,
    )

    combined_fields, combined_rows = build_consolidated_rows(
        output_rows
    )

    write_csv(
        combined_csv,
        combined_rows,
        combined_fields,
    )

    write_review_html(
        review_html,
        review_rows,
        crop_root,
    )

    package_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(review_csv, package_root / "review_cases.csv")
    shutil.copy2(review_html, package_root / "review.html")

    copy_review_assets(
        review_rows,
        crop_root,
        raw_root,
        package_root,
    )

    summary_lines = [
        "ARBITRAGE OCR STATISTIQUES V3",
        "=" * 72,
        "",
        f"Valeurs analysées : {len(output_rows)}",
        (
            "Valeurs attribuées automatiquement : "
            f"{sum(1 for row in output_rows if row['review_required'] == '0' and row['arbitration_reason'] != 'EMPTY_SLOT')}"
        ),
        (
            "Emplacements numériques vides ignorés : "
            f"{reason_counts['EMPTY_SLOT']}"
        ),
        f"Valeurs restant à revoir : {len(review_rows)}",
        "",
        "Décisions automatiques :",
    ]

    for reason, count in reason_counts.most_common():
        summary_lines.append(
            f"- {reason:<34} : {count}"
        )

    summary_lines.extend(
        [
            "",
            f"CSV détaillé : {detailed_csv}",
            f"Table consolidée : {combined_csv}",
            f"Cas à revoir : {review_csv}",
            f"Rapport HTML : {review_html}",
            f"Archive : {zip_path}",
            "",
        ]
    )

    summary = "\n".join(summary_lines)
    summary_path.write_text(summary, encoding="utf-8")

    shutil.copy2(summary_path, package_root / "summary.txt")

    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    path.relative_to(package_root),
                )

    print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
