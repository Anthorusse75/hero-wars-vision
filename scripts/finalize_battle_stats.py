from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


METRICS = (
    "power",
    "damage_dealt",
    "damage_taken",
    "healing",
)

EMPTY_SLOT_STATUSES = {
    "EMPTY",
    "EMPTY_SLOT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalise les statistiques OCR d'un lot à partir du résultat "
            "d'arbitrage V3 et d'un CSV de décisions manuelles."
        )
    )
    parser.add_argument(
        "--batch",
        required=True,
        help="Nom du lot, par exemple hero_batch_004.",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=None,
        help=(
            "CSV des décisions manuelles. Par défaut : "
            "data/batches/<batch>/reports/stat_ocr_v3/"
            "manual_decisions.csv"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Écrit réellement les fichiers validés. Sans cette option, "
            "le script effectue uniquement une simulation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Autorise le remplacement des fichiers validés existants "
            "avec sauvegarde préalable."
        ),
    )
    return parser.parse_args()


def validate_batch_name(batch_name: str) -> None:
    if not re.fullmatch(r"hero_batch_\d{3}", batch_name):
        raise RuntimeError(
            "--batch doit avoir la forme hero_batch_004."
        )


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
        rows = [
            {
                key: value or ""
                for key, value in row.items()
            }
            for row in reader
        ]

    if not fieldnames:
        raise RuntimeError(f"En-tête CSV absent : {path}")

    return fieldnames, rows


def write_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
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

    temporary.replace(path)


def row_key(
    row: dict[str, str],
) -> tuple[str, str, str, str]:
    return (
        str(row.get("screenshot_id") or "").strip(),
        str(row.get("side") or "").strip().upper(),
        str(row.get("slot") or "").strip(),
        str(row.get("metric") or "").strip(),
    )


def slot_key(
    row: dict[str, str],
) -> tuple[str, str, str]:
    key = row_key(row)
    return key[0], key[1], key[2]


def is_empty_slot(row: dict[str, str]) -> bool:
    return (
        str(row.get("slot_status") or "")
        .strip()
        .upper()
        in EMPTY_SLOT_STATUSES
    )


def ensure_digits(
    value: str,
    description: str,
) -> str:
    text = str(value or "").strip()

    if not text.isdigit():
        raise RuntimeError(
            f"{description} invalide : {text!r}"
        )

    return str(int(text))


def validate_unique_keys(
    rows: list[dict[str, str]],
    description: str,
) -> None:
    counts = Counter(
        row_key(row)
        for row in rows
    )

    duplicates = [
        key
        for key, count in counts.items()
        if count != 1
    ]

    if duplicates:
        raise RuntimeError(
            f"{description} : clés absentes ou dupliquées :\n"
            + "\n".join(
                "- " + " ".join(key)
                for key in sorted(duplicates)[:50]
            )
        )


def load_manual_decisions(
    path: Path,
) -> tuple[
    list[str],
    list[dict[str, str]],
    dict[tuple[str, str, str, str], dict[str, str]],
]:
    fields, rows = read_csv(path)

    required_fields = {
        "screenshot_id",
        "side",
        "slot",
        "metric",
        "validated_value",
    }

    missing = sorted(
        required_fields.difference(fields)
    )

    if missing:
        raise RuntimeError(
            "Colonnes absentes du CSV de décisions : "
            + ", ".join(missing)
        )

    validate_unique_keys(
        rows,
        "CSV de décisions manuelles",
    )

    indexed: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ] = {}

    for row in rows:
        metric = row.get("metric", "")

        if metric not in METRICS:
            raise RuntimeError(
                f"Métrique inconnue dans les décisions : {metric!r}"
            )

        row["validated_value"] = ensure_digits(
            row.get("validated_value", ""),
            "Valeur validée",
        )

        indexed[row_key(row)] = row

    return fields, rows, indexed


def validate_review_coverage(
    review_rows: list[dict[str, str]],
    decision_index: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ],
) -> None:
    review_keys = {
        row_key(row)
        for row in review_rows
    }
    decision_keys = set(decision_index)

    missing = sorted(
        review_keys.difference(decision_keys)
    )
    extra = sorted(
        decision_keys.difference(review_keys)
    )

    if not missing and not extra:
        return

    messages: list[str] = []

    if missing:
        messages.append(
            "Cas V3 sans décision manuelle :\n"
            + "\n".join(
                "- " + " ".join(key)
                for key in missing[:50]
            )
        )

    if extra:
        messages.append(
            "Décisions sans cas correspondant dans review_cases.csv :\n"
            + "\n".join(
                "- " + " ".join(key)
                for key in extra[:50]
            )
        )

    raise RuntimeError("\n\n".join(messages))


def build_final_rows(
    source_rows: list[dict[str, str]],
    decision_index: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
]:
    final_rows: list[dict[str, str]] = []
    applied_decisions: list[dict[str, str]] = []
    applied_keys: set[
        tuple[str, str, str, str]
    ] = set()

    for source_row in source_rows:
        row = dict(source_row)
        key = row_key(row)
        decision = decision_index.get(key)

        if decision is None:
            row["manual_reviewed"] = "0"
            final_rows.append(row)
            continue

        validated_value = decision[
            "validated_value"
        ]
        applied_keys.add(key)

        row["final_value"] = validated_value
        row["arbitration_reason"] = (
            "MANUAL_VISUAL_REVIEW"
        )
        row["review_required"] = "0"
        row["manual_reviewed"] = "1"

        applied_decisions.append(
            {
                "screenshot_id": key[0],
                "side": key[1],
                "slot": key[2],
                "slot_status": row.get(
                    "slot_status",
                    "",
                ),
                "hero_uid": row.get(
                    "hero_uid",
                    "",
                ),
                "hero_name": row.get(
                    "hero_name",
                    "",
                ),
                "metric": key[3],
                "v1_value": row.get(
                    "old_value",
                    decision.get("v1_value", ""),
                ),
                "v2_value": row.get(
                    "value",
                    decision.get("v2_value", ""),
                ),
                "validated_value": validated_value,
                "decision_source": decision.get(
                    "decision_source",
                    "human_visual_review",
                ),
                "notes": decision.get(
                    "notes",
                    "",
                ),
            }
        )

        final_rows.append(row)

    unapplied = sorted(
        set(decision_index).difference(applied_keys)
    )

    if unapplied:
        raise RuntimeError(
            "Décisions non appliquées au fichier V3 :\n"
            + "\n".join(
                "- " + " ".join(key)
                for key in unapplied[:50]
            )
        )

    return final_rows, applied_decisions


def build_consolidated(
    final_rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    by_slot: dict[
        tuple[str, str, str],
        dict[str, str],
    ] = {}
    seen_metrics: set[
        tuple[str, str, str, str]
    ] = set()

    for row in final_rows:
        metric = row.get("metric", "")

        if metric not in METRICS:
            raise RuntimeError(
                f"Métrique inconnue : {metric!r}"
            )

        current_slot = slot_key(row)
        metric_key = (*current_slot, metric)

        if metric_key in seen_metrics:
            raise RuntimeError(
                "Métrique dupliquée : "
                + " ".join(metric_key)
            )

        seen_metrics.add(metric_key)

        combined = by_slot.setdefault(
            current_slot,
            {
                "screenshot_id": current_slot[0],
                "side": current_slot[1],
                "slot": current_slot[2],
                "slot_status": row.get(
                    "slot_status",
                    "",
                ),
                "hero_uid": row.get(
                    "hero_uid",
                    "",
                ),
                "hero_name": row.get(
                    "hero_name",
                    "",
                ),
            },
        )

        combined[metric] = row.get(
            "final_value",
            "",
        )
        combined[f"{metric}_source"] = row.get(
            "arbitration_reason",
            "",
        )
        combined[
            f"{metric}_manual_reviewed"
        ] = row.get(
            "manual_reviewed",
            "0",
        )

    consolidated_rows = sorted(
        by_slot.values(),
        key=lambda row: (
            int(row["screenshot_id"]),
            row["side"],
            int(row["slot"]),
        ),
    )

    fieldnames = [
        "screenshot_id",
        "side",
        "slot",
        "slot_status",
        "hero_uid",
        "hero_name",
    ]

    for metric in METRICS:
        fieldnames.extend(
            [
                metric,
                f"{metric}_source",
                f"{metric}_manual_reviewed",
            ]
        )

    return fieldnames, consolidated_rows


def validate_final_state(
    final_rows: list[dict[str, str]],
    consolidated_rows: list[dict[str, str]],
    manual_count: int,
) -> dict[str, int]:
    if len(final_rows) % len(METRICS) != 0:
        raise RuntimeError(
            "Le nombre de lignes numériques n'est pas divisible par 4."
        )

    remaining_review = sum(
        str(row.get("review_required") or "0")
        .strip()
        == "1"
        for row in final_rows
    )

    if remaining_review:
        raise RuntimeError(
            f"Il reste {remaining_review} valeurs à revoir."
        )

    empty_numeric_rows = sum(
        is_empty_slot(row)
        for row in final_rows
    )

    hero_numeric_rows = [
        row
        for row in final_rows
        if not is_empty_slot(row)
    ]

    invalid_hero_values = [
        row_key(row)
        for row in hero_numeric_rows
        if not str(
            row.get("final_value") or ""
        ).isdigit()
    ]

    if invalid_hero_values:
        raise RuntimeError(
            "Valeurs finales absentes ou invalides :\n"
            + "\n".join(
                "- " + " ".join(key)
                for key in invalid_hero_values[:50]
            )
        )

    incomplete_slots: list[str] = []

    for row in consolidated_rows:
        missing_metrics = [
            metric
            for metric in METRICS
            if metric not in row
        ]

        if missing_metrics:
            incomplete_slots.append(
                f"{row['screenshot_id']} "
                f"{row['side']}{row['slot']} : "
                + ", ".join(missing_metrics)
            )
            continue

        if is_empty_slot(row):
            non_empty_values = [
                metric
                for metric in METRICS
                if str(row.get(metric) or "").strip()
            ]

            if non_empty_values:
                incomplete_slots.append(
                    f"{row['screenshot_id']} "
                    f"{row['side']}{row['slot']} : "
                    "emplacement vide avec valeurs numériques"
                )
        else:
            invalid_metrics = [
                metric
                for metric in METRICS
                if not str(
                    row.get(metric) or ""
                ).isdigit()
            ]

            if invalid_metrics:
                incomplete_slots.append(
                    f"{row['screenshot_id']} "
                    f"{row['side']}{row['slot']} : "
                    + ", ".join(invalid_metrics)
                )

    if incomplete_slots:
        raise RuntimeError(
            "Emplacements consolidés invalides :\n"
            + "\n".join(incomplete_slots[:50])
        )

    expected_slots = (
        len(final_rows) // len(METRICS)
    )

    if len(consolidated_rows) != expected_slots:
        raise RuntimeError(
            f"{expected_slots} emplacements attendus, "
            f"{len(consolidated_rows)} obtenus."
        )

    hero_slots = sum(
        not is_empty_slot(row)
        for row in consolidated_rows
    )
    empty_slots = sum(
        is_empty_slot(row)
        for row in consolidated_rows
    )

    if empty_numeric_rows != empty_slots * len(METRICS):
        raise RuntimeError(
            "Le nombre de lignes numériques vides ne correspond pas "
            "au nombre d'emplacements vides."
        )

    actual_manual_count = sum(
        row.get("manual_reviewed") == "1"
        for row in final_rows
    )

    if actual_manual_count != manual_count:
        raise RuntimeError(
            f"{manual_count} décisions manuelles attendues, "
            f"{actual_manual_count} appliquées."
        )

    return {
        "numeric_rows": len(final_rows),
        "hero_numeric_rows": len(
            hero_numeric_rows
        ),
        "empty_numeric_rows": (
            empty_numeric_rows
        ),
        "slots": len(consolidated_rows),
        "hero_slots": hero_slots,
        "empty_slots": empty_slots,
        "manual_values": actual_manual_count,
        "automatic_values": (
            len(hero_numeric_rows)
            - actual_manual_count
        ),
    }


def backup_existing_outputs(
    paths: list[Path],
    output_dir: Path,
) -> Path | None:
    existing = [
        path
        for path in paths
        if path.exists()
    ]

    if not existing:
        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    backup_dir = (
        output_dir
        / "backups"
        / f"battle_stats_{timestamp}"
    )
    backup_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    for path in existing:
        shutil.copy2(
            path,
            backup_dir / path.name,
        )

    return backup_dir


def main() -> int:
    args = parse_args()

    try:
        validate_batch_name(args.batch)

        batch_root = (
            Path("data/batches")
            / args.batch
        )
        source_dir = (
            batch_root
            / "reports"
            / "stat_ocr_v3"
        )
        source_values = (
            source_dir
            / "stat_values_arbitrated.csv"
        )
        source_review = (
            source_dir
            / "review_cases.csv"
        )
        decisions_path = (
            args.decisions
            if args.decisions is not None
            else source_dir
            / "manual_decisions.csv"
        )

        output_dir = (
            batch_root
            / "validated"
        )
        final_values_path = (
            output_dir
            / "stat_values_final.csv"
        )
        final_stats_path = (
            output_dir
            / "battle_stats_final.csv"
        )
        manual_output_path = (
            output_dir
            / "manual_stat_decisions.csv"
        )
        summary_path = (
            output_dir
            / "battle_stats_validation_summary.txt"
        )

        source_fields, source_rows = read_csv(
            source_values
        )
        _, review_rows = read_csv(
            source_review
        )
        (
            _,
            decision_rows,
            decision_index,
        ) = load_manual_decisions(
            decisions_path
        )

        validate_unique_keys(
            source_rows,
            "Résultats V3",
        )
        validate_unique_keys(
            review_rows,
            "Cas V3 à revoir",
        )
        validate_review_coverage(
            review_rows,
            decision_index,
        )

        final_rows, applied_decisions = (
            build_final_rows(
                source_rows,
                decision_index,
            )
        )

        final_fields = list(source_fields)

        if "manual_reviewed" not in final_fields:
            final_fields.append(
                "manual_reviewed"
            )

        (
            consolidated_fields,
            consolidated_rows,
        ) = build_consolidated(final_rows)

        validation = validate_final_state(
            final_rows,
            consolidated_rows,
            len(decision_rows),
        )

    except (
        RuntimeError,
        csv.Error,
        OSError,
        ValueError,
    ) as error:
        print(
            f"Erreur : {error}",
            file=sys.stderr,
        )
        return 1

    reason_counts = Counter(
        row.get("arbitration_reason", "")
        for row in final_rows
    )

    print(
        "FINALISATION DES STATISTIQUES — "
        f"{args.batch.upper()}"
    )
    print("=" * 72)
    print()
    print(
        "Valeurs numériques analysées : "
        f"{validation['numeric_rows']}"
    )
    print(
        "Valeurs de héros validées    : "
        f"{validation['hero_numeric_rows']}"
    )
    print(
        "Valeurs vides ignorées       : "
        f"{validation['empty_numeric_rows']}"
    )
    print(
        "Valeurs revues manuellement  : "
        f"{validation['manual_values']}"
    )
    print(
        "Valeurs validées automatiquement : "
        f"{validation['automatic_values']}"
    )
    print(
        "Emplacements de héros        : "
        f"{validation['hero_slots']}"
    )
    print(
        "Emplacements vides           : "
        f"{validation['empty_slots']}"
    )
    print("Cas restant à revoir         : 0")
    print()
    print("Origine des valeurs finales :")

    for reason, count in reason_counts.most_common():
        print(
            f"- {reason:<34} : {count}"
        )

    summary_lines = [
        "STATISTIQUES VALIDÉES — "
        f"{args.batch.upper()}",
        "=" * 72,
        "",
        "Valeurs numériques analysées : "
        f"{validation['numeric_rows']}",
        "Valeurs appartenant à des héros : "
        f"{validation['hero_numeric_rows']}",
        "Valeurs d'emplacement vide ignorées : "
        f"{validation['empty_numeric_rows']}",
        "Valeurs revues manuellement : "
        f"{validation['manual_values']}",
        "Valeurs validées automatiquement : "
        f"{validation['automatic_values']}",
        "Emplacements contenant un héros : "
        f"{validation['hero_slots']}",
        "Emplacements vides : "
        f"{validation['empty_slots']}",
        "Cas restant à revoir : 0",
        "",
        "Origine des valeurs finales :",
    ]

    for reason, count in reason_counts.most_common():
        summary_lines.append(
            f"- {reason}: {count}"
        )

    summary_lines.extend(
        [
            "",
            "Important : seules les valeurs signalées par l'arbitrage "
            "V3 ont été vérifiées manuellement.",
            "Les autres valeurs proviennent de l'arbitrage automatique.",
            "",
        ]
    )

    if not args.apply:
        print()
        print(
            "MODE SIMULATION : aucun fichier "
            "n'a été modifié."
        )
        print()
        print("Pour appliquer :")
        print(
            "python scripts/finalize_battle_stats.py "
            f"--batch {args.batch} --apply"
        )
        return 0

    output_paths = [
        final_values_path,
        final_stats_path,
        manual_output_path,
        summary_path,
    ]

    existing_outputs = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing_outputs and not args.overwrite:
        print(
            "Des fichiers validés existent déjà. "
            "Relance avec --overwrite pour les remplacer.",
            file=sys.stderr,
        )
        return 2

    backup_dir = backup_existing_outputs(
        output_paths,
        output_dir,
    )

    manual_output_fields = [
        "screenshot_id",
        "side",
        "slot",
        "slot_status",
        "hero_uid",
        "hero_name",
        "metric",
        "v1_value",
        "v2_value",
        "validated_value",
        "decision_source",
        "notes",
    ]

    write_csv_atomic(
        final_values_path,
        final_fields,
        final_rows,
    )
    write_csv_atomic(
        final_stats_path,
        consolidated_fields,
        consolidated_rows,
    )
    write_csv_atomic(
        manual_output_path,
        manual_output_fields,
        applied_decisions,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("Finalisation appliquée.")
    print(
        f"Valeurs détaillées : {final_values_path}"
    )
    print(
        f"Table par héros     : {final_stats_path}"
    )
    print(
        f"Décisions manuelles : {manual_output_path}"
    )
    print(
        f"Résumé              : {summary_path}"
    )

    if backup_dir is not None:
        print(
            f"Sauvegarde précédente : {backup_dir}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
