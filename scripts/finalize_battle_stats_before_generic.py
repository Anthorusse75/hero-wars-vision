from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


BATCH_NAME = "hero_batch_002"

BATCH_ROOT = Path("data/batches") / BATCH_NAME
SOURCE_DIR = BATCH_ROOT / "reports" / "stat_ocr_v3"
SOURCE_VALUES = SOURCE_DIR / "stat_values_arbitrated.csv"
SOURCE_REVIEW = SOURCE_DIR / "review_cases.csv"

OUTPUT_DIR = BATCH_ROOT / "validated"
FINAL_VALUES = OUTPUT_DIR / "stat_values_final.csv"
FINAL_STATS = OUTPUT_DIR / "battle_stats_final.csv"
MANUAL_DECISIONS = OUTPUT_DIR / "manual_stat_decisions.csv"
SUMMARY_PATH = OUTPUT_DIR / "battle_stats_validation_summary.txt"

METRICS = (
    "power",
    "damage_dealt",
    "damage_taken",
    "healing",
)

# Valeurs lues visuellement dans les découpes et, lorsque nécessaire,
# confirmées sur la capture complète.
#
# Clé : (screenshot_id, side, slot, metric)
# Valeur : entier validé
MANUAL_VALUES: dict[tuple[str, str, str, str], int] = {
    ("24701", "L", "1", "damage_dealt"): 1_139_081,
    ("24701", "R", "1", "damage_dealt"): 1_283_674,
    ("24701", "R", "2", "damage_taken"): 1_113_917,
    ("24809", "L", "2", "power"): 104_542,
    ("28468", "L", "1", "power"): 177_079,
    ("28468", "L", "2", "power"): 165_819,
    ("28468", "R", "4", "healing"): 682_257,
    ("28468", "R", "5", "damage_dealt"): 8_137,
    ("29563", "R", "5", "damage_dealt"): 7_532,
    ("30809", "L", "4", "power"): 169_791,
    ("31047", "R", "1", "power"): 219_282,
    ("31047", "R", "1", "damage_taken"): 637_897,
    ("31047", "L", "2", "power"): 232_322,
    ("31047", "L", "2", "damage_dealt"): 1_004_412,
    ("31047", "R", "2", "power"): 203_911,
    ("31047", "R", "3", "damage_dealt"): 148_140,
    ("31047", "R", "3", "damage_taken"): 676_940,
    ("31047", "L", "4", "power"): 232_230,
    ("31047", "L", "4", "damage_dealt"): 228_702,
    ("31047", "R", "4", "damage_dealt"): 88_928,
    ("31047", "L", "5", "damage_dealt"): 7_316,
    ("31047", "R", "5", "healing"): 85_323,
    ("33031", "R", "4", "damage_taken"): 690_245,
    ("33474", "R", "1", "power"): 241_896,
    ("34279", "R", "5", "damage_taken"): 642_200,
    ("35041", "L", "4", "damage_taken"): 97_337,
    ("35041", "R", "4", "damage_dealt"): 160_757,
    ("35041", "R", "4", "damage_taken"): 316_354,
    ("35041", "L", "5", "damage_dealt"): 3_931,
    ("35041", "L", "5", "damage_taken"): 486_447,
    ("35041", "L", "5", "healing"): 175_618,
    ("35041", "R", "5", "power"): 95_457,
    ("35041", "R", "5", "damage_dealt"): 33_304,
    ("35041", "R", "5", "damage_taken"): 1_227_876,
    ("35632", "L", "4", "damage_dealt"): 17_569,
    ("35632", "R", "4", "damage_dealt"): 23_188,
    ("35632", "R", "4", "damage_taken"): 641_536,
    ("35660", "R", "4", "damage_dealt"): 0,
    ("35660", "R", "5", "power"): 179_605,
    ("36257", "R", "2", "damage_taken"): 4_890_617,
    ("36964", "L", "2", "power"): 162_149,
    ("36964", "R", "4", "healing"): 808_189,
    ("37112", "R", "4", "power"): 106_619,
    ("37112", "R", "5", "damage_dealt"): 0,
    ("37790", "R", "2", "damage_dealt"): 684_447,
    ("38170", "R", "3", "damage_taken"): 1_117_365,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Applique les 46 décisions humaines finales aux statistiques "
            "du batch 002 et produit les CSV validés."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Écrit réellement les fichiers validés. Sans cette option, "
            "le script effectue uniquement les contrôles."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Autorise le remplacement des fichiers validés existants.",
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


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        str(row.get("screenshot_id") or ""),
        str(row.get("side") or ""),
        str(row.get("slot") or ""),
        str(row.get("metric") or ""),
    )


def int_text(value: Any) -> str:
    text = str(value if value is not None else "").strip()

    if not text:
        return ""

    if not text.isdigit():
        raise RuntimeError(f"Valeur numérique invalide : {text!r}")

    return str(int(text))


def build_final_rows(
    source_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    final_rows: list[dict[str, str]] = []
    decision_rows: list[dict[str, str]] = []
    applied_keys: set[tuple[str, str, str, str]] = set()

    for source_row in source_rows:
        row = dict(source_row)
        key = row_key(row)

        if key in MANUAL_VALUES:
            validated_value = str(MANUAL_VALUES[key])
            applied_keys.add(key)

            decision_rows.append(
                {
                    "screenshot_id": key[0],
                    "side": key[1],
                    "slot": key[2],
                    "hero_uid": row.get("hero_uid", ""),
                    "hero_name": row.get("hero_name", ""),
                    "metric": key[3],
                    "v1_value": row.get("old_value", ""),
                    "v2_value": row.get("value", ""),
                    "validated_value": validated_value,
                    "decision_source": "human_visual_review",
                    "notes": (
                        "Valeur lue sur la découpe V2 et, lorsque la découpe "
                        "était ambiguë, confirmée sur la capture complète."
                    ),
                }
            )

            row["final_value"] = validated_value
            row["arbitration_reason"] = "MANUAL_VISUAL_REVIEW"
            row["review_required"] = "0"
            row["manual_reviewed"] = "1"
        else:
            row["manual_reviewed"] = "0"

        final_rows.append(row)

    missing_decisions = set(MANUAL_VALUES).difference(applied_keys)

    if missing_decisions:
        details = "\n".join(
            "- " + " ".join(key)
            for key in sorted(missing_decisions)
        )
        raise RuntimeError(
            "Certaines décisions ne correspondent à aucune ligne V3 :\n"
            + details
        )

    return final_rows, decision_rows


def validate_review_coverage(
    review_rows: list[dict[str, str]],
) -> None:
    review_keys = {row_key(row) for row in review_rows}
    decision_keys = set(MANUAL_VALUES)

    missing = review_keys.difference(decision_keys)
    extra = decision_keys.difference(review_keys)

    if missing or extra:
        messages: list[str] = []

        if missing:
            messages.append(
                "Cas V3 sans décision :\n"
                + "\n".join(
                    "- " + " ".join(key)
                    for key in sorted(missing)
                )
            )

        if extra:
            messages.append(
                "Décisions absentes du fichier review_cases.csv :\n"
                + "\n".join(
                    "- " + " ".join(key)
                    for key in sorted(extra)
                )
            )

        raise RuntimeError("\n\n".join(messages))


def build_consolidated(
    final_rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    by_slot: dict[tuple[str, str, str], dict[str, str]] = {}

    for row in final_rows:
        key = (
            row.get("screenshot_id", ""),
            row.get("side", ""),
            row.get("slot", ""),
        )

        combined = by_slot.setdefault(
            key,
            {
                "screenshot_id": key[0],
                "side": key[1],
                "slot": key[2],
                "slot_status": row.get("slot_status", ""),
                "hero_uid": row.get("hero_uid", ""),
                "hero_name": row.get("hero_name", ""),
            },
        )

        metric = row.get("metric", "")

        if metric not in METRICS:
            raise RuntimeError(f"Métrique inconnue : {metric!r}")

        combined[metric] = row.get("final_value", "")
        combined[f"{metric}_source"] = row.get(
            "arbitration_reason",
            "",
        )
        combined[f"{metric}_manual_reviewed"] = row.get(
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


def validate_final_rows(
    final_rows: list[dict[str, str]],
    consolidated_rows: list[dict[str, str]],
) -> dict[str, int]:
    review_remaining = sum(
        str(row.get("review_required") or "0") == "1"
        for row in final_rows
    )

    empty_numeric_rows = sum(
        row.get("slot_status") == "EMPTY"
        for row in final_rows
    )

    hero_numeric_rows = [
        row
        for row in final_rows
        if row.get("slot_status") != "EMPTY"
    ]

    invalid_values = [
        row_key(row)
        for row in hero_numeric_rows
        if not str(row.get("final_value") or "").isdigit()
    ]

    if review_remaining:
        raise RuntimeError(
            f"Il reste {review_remaining} valeurs à revoir."
        )

    if invalid_values:
        raise RuntimeError(
            "Valeurs finales absentes ou invalides :\n"
            + "\n".join(
                "- " + " ".join(key)
                for key in invalid_values[:30]
            )
        )

    slots_with_missing_metric: list[str] = []

    for row in consolidated_rows:
        if row.get("slot_status") == "EMPTY":
            continue

        missing = [
            metric
            for metric in METRICS
            if not str(row.get(metric) or "").isdigit()
        ]

        if missing:
            slots_with_missing_metric.append(
                f"{row['screenshot_id']} {row['side']}{row['slot']} : "
                + ", ".join(missing)
            )

    if slots_with_missing_metric:
        raise RuntimeError(
            "Emplacements incomplets :\n"
            + "\n".join(slots_with_missing_metric[:30])
        )

    return {
        "numeric_rows": len(final_rows),
        "hero_numeric_rows": len(hero_numeric_rows),
        "empty_numeric_rows": empty_numeric_rows,
        "slots": len(consolidated_rows),
        "hero_slots": sum(
            row.get("slot_status") != "EMPTY"
            for row in consolidated_rows
        ),
        "empty_slots": sum(
            row.get("slot_status") == "EMPTY"
            for row in consolidated_rows
        ),
        "manual_values": sum(
            row.get("manual_reviewed") == "1"
            for row in final_rows
        ),
    }


def backup_existing_outputs() -> Path | None:
    existing = [
        path
        for path in (
            FINAL_VALUES,
            FINAL_STATS,
            MANUAL_DECISIONS,
            SUMMARY_PATH,
        )
        if path.exists()
    ]

    if not existing:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = (
        BATCH_ROOT
        / "validated"
        / "backups"
        / f"battle_stats_{timestamp}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)

    for path in existing:
        shutil.copy2(path, backup_dir / path.name)

    return backup_dir


def main() -> int:
    args = parse_args()

    try:
        source_fields, source_rows = read_csv(SOURCE_VALUES)
        _, review_rows = read_csv(SOURCE_REVIEW)

        validate_review_coverage(review_rows)

        final_rows, decision_rows = build_final_rows(source_rows)

        final_fields = list(source_fields)

        if "manual_reviewed" not in final_fields:
            final_fields.append("manual_reviewed")

        consolidated_fields, consolidated_rows = build_consolidated(
            final_rows
        )
        validation = validate_final_rows(
            final_rows,
            consolidated_rows,
        )

    except (RuntimeError, csv.Error, ValueError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    reason_counts = Counter(
        row.get("arbitration_reason", "")
        for row in final_rows
    )

    print("FINALISATION DES STATISTIQUES — HERO_BATCH_002")
    print("=" * 72)
    print()
    print(f"Valeurs numériques analysées : {validation['numeric_rows']}")
    print(f"Valeurs de héros validées    : {validation['hero_numeric_rows']}")
    print(f"Valeurs vides ignorées       : {validation['empty_numeric_rows']}")
    print(f"Valeurs revues manuellement  : {validation['manual_values']}")
    print(f"Emplacements de héros        : {validation['hero_slots']}")
    print(f"Emplacements vides           : {validation['empty_slots']}")
    print("Cas restant à revoir         : 0")
    print()
    print("Origine des valeurs finales :")

    for reason, count in reason_counts.most_common():
        print(f"- {reason:<34} : {count}")

    summary = "\n".join(
        [
            "STATISTIQUES VALIDÉES — HERO_BATCH_002",
            "=" * 72,
            "",
            f"Valeurs numériques analysées : {validation['numeric_rows']}",
            f"Valeurs appartenant à des héros : {validation['hero_numeric_rows']}",
            f"Valeurs d'emplacement vide ignorées : {validation['empty_numeric_rows']}",
            f"Valeurs revues manuellement : {validation['manual_values']}",
            f"Emplacements contenant un héros : {validation['hero_slots']}",
            f"Emplacements vides : {validation['empty_slots']}",
            "Cas restant à revoir : 0",
            "",
            "Important : 46 valeurs ambiguës ont été vérifiées visuellement.",
            "Les 3 950 autres valeurs proviennent de l'arbitrage automatique V3.",
            "Le lot complet n'a pas été annoté manuellement valeur par valeur.",
            "",
        ]
    )

    if not args.apply:
        print()
        print("MODE SIMULATION : aucun fichier n'a été modifié.")
        print()
        print("Pour appliquer :")
        print(
            "python scripts/finalize_battle_stats.py --apply"
        )
        return 0

    existing_outputs = [
        path
        for path in (
            FINAL_VALUES,
            FINAL_STATS,
            MANUAL_DECISIONS,
            SUMMARY_PATH,
        )
        if path.exists()
    ]

    if existing_outputs and not args.overwrite:
        print(
            "Des fichiers validés existent déjà. "
            "Relance avec --overwrite pour les remplacer.",
            file=sys.stderr,
        )
        return 2

    backup_dir = backup_existing_outputs()

    decision_fields = [
        "screenshot_id",
        "side",
        "slot",
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
        FINAL_VALUES,
        final_fields,
        final_rows,
    )
    write_csv_atomic(
        FINAL_STATS,
        consolidated_fields,
        consolidated_rows,
    )
    write_csv_atomic(
        MANUAL_DECISIONS,
        decision_fields,
        decision_rows,
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print()
    print("Finalisation appliquée.")
    print(f"Valeurs détaillées : {FINAL_VALUES}")
    print(f"Table par héros     : {FINAL_STATS}")
    print(f"Décisions manuelles : {MANUAL_DECISIONS}")
    print(f"Résumé              : {SUMMARY_PATH}")

    if backup_dir is not None:
        print(f"Sauvegarde précédente : {backup_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
