from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


BATCH_NAME = "hero_batch_003"

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

EMPTY_SLOT_STATUSES = {
    "EMPTY",
    "EMPTY_SLOT",
}

EXPECTED_NUMERIC_ROWS = 4_000
EXPECTED_HERO_NUMERIC_ROWS = 3_996
EXPECTED_EMPTY_NUMERIC_ROWS = 4
EXPECTED_SLOTS = 1_000
EXPECTED_HERO_SLOTS = 999
EXPECTED_EMPTY_SLOTS = 1
EXPECTED_MANUAL_VALUES = 57


# Valeurs vérifiées visuellement dans les 57 découpes ambiguës.
# Lorsque la découpe contenait plusieurs lignes ou était mal positionnée,
# la valeur a été confirmée sur la capture complète.
#
# Clé : (screenshot_id, side, slot, metric)
# Valeur : entier validé
MANUAL_VALUES: dict[tuple[str, str, str, str], int] = {
    ("24068", "R", "1", "damage_taken"): 643_110,
    ("24068", "R", "1", "healing"): 62_226,
    ("24068", "R", "4", "damage_dealt"): 8_563,
    ("24068", "R", "5", "healing"): 61_053,
    ("24509", "R", "1", "damage_dealt"): 1_452_574,
    ("24753", "R", "1", "damage_taken"): 607_305,
    ("24818", "L", "2", "power"): 104_542,
    ("28002", "L", "3", "power"): 165_382,
    ("28153", "L", "1", "damage_dealt"): 2_142_041,
    ("28153", "L", "1", "damage_taken"): 2_485_694,
    ("28153", "R", "1", "damage_taken"): 1_022_259,
    ("28153", "L", "2", "healing"): 2_504_891,
    ("28153", "L", "4", "healing"): 1_408_679,
    ("28153", "L", "5", "healing"): 4_870_679,
    ("28153", "R", "5", "damage_taken"): 1_257_361,
    ("2820", "R", "4", "damage_dealt"): 8_947,
    ("28314", "R", "2", "damage_dealt"): 69_307,
    ("28353", "R", "5", "power"): 135_812,
    ("28353", "R", "5", "damage_dealt"): 20_974,
    ("28353", "R", "5", "damage_taken"): 403_561,
    ("29714", "L", "3", "power"): 135_582,
    ("29803", "L", "1", "damage_dealt"): 5_271_786,
    ("29803", "L", "1", "damage_taken"): 15_531,
    ("29803", "R", "1", "damage_dealt"): 3_298_035,
    ("29803", "L", "2", "damage_dealt"): 74_018,
    ("29803", "R", "2", "damage_dealt"): 628_637,
    ("29803", "L", "3", "power"): 199_549,
    ("30497", "L", "1", "power"): 216_531,
    ("30614", "R", "1", "healing"): 449_040,
    ("30614", "R", "3", "damage_dealt"): 68_523,
    ("30614", "R", "4", "healing"): 68_474,
    ("30614", "R", "5", "damage_taken"): 630_911,
    ("30614", "R", "5", "healing"): 679_714,
    ("31260", "L", "5", "power"): 164_352,
    ("31352", "R", "3", "damage_taken"): 1_247_851,
    ("31438", "R", "3", "power"): 85_089,
    ("31748", "L", "5", "power"): 179_282,
    ("32650", "L", "1", "power"): 192_079,
    ("32934", "R", "1", "healing"): 627_625,
    ("33446", "L", "1", "power"): 208_399,
    ("35428", "L", "1", "damage_dealt"): 3_423_659,
    ("35428", "R", "2", "damage_taken"): 1_384_935,
    ("35428", "R", "4", "damage_taken"): 1_263_661,
    ("35428", "R", "5", "healing"): 1_928_674,
    ("35885", "L", "5", "power"): 188_262,
    ("36322", "L", "1", "power"): 192_842,
    ("36322", "L", "4", "power"): 171_389,
    ("36322", "R", "4", "damage_dealt"): 60_541,
    ("37606", "R", "2", "damage_dealt"): 39_154,
    ("37606", "R", "3", "damage_dealt"): 6_725,
    ("37982", "L", "1", "power"): 272_285,
    ("37982", "L", "2", "damage_dealt"): 1_486_433,
    ("37982", "L", "2", "healing"): 817_383,
    ("37982", "R", "4", "healing"): 626_193,
    ("37982", "L", "5", "healing"): 2_039_055,
    ("37982", "R", "5", "healing"): 708_919,
    ("6900", "L", "5", "power"): 143_492,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Applique les 57 décisions humaines finales aux statistiques "
            "du batch 003 et produit les CSV validés."
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


def is_empty_slot(row: dict[str, str]) -> bool:
    return (
        str(row.get("slot_status") or "")
        .strip()
        .upper()
        in EMPTY_SLOT_STATUSES
    )


def int_text(value: Any) -> str:
    text = str(value if value is not None else "").strip()

    if not text:
        return ""

    if not text.isdigit():
        raise RuntimeError(f"Valeur numérique invalide : {text!r}")

    return str(int(text))


def validate_source_uniqueness(
    source_rows: list[dict[str, str]],
) -> None:
    counts = Counter(row_key(row) for row in source_rows)
    duplicates = [
        key
        for key, count in counts.items()
        if count != 1
    ]

    if duplicates:
        raise RuntimeError(
            "Lignes numériques absentes ou dupliquées :\n"
            + "\n".join(
                "- " + " ".join(key)
                for key in sorted(duplicates)[:30]
            )
        )


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
                        "Valeur lue sur la découpe et, lorsque plusieurs "
                        "lignes apparaissaient dans celle-ci, confirmée sur "
                        "la capture complète."
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
    seen_metrics: set[tuple[str, str, str, str]] = set()

    for row in final_rows:
        slot_key = (
            row.get("screenshot_id", ""),
            row.get("side", ""),
            row.get("slot", ""),
        )

        metric = row.get("metric", "")
        metric_key = (*slot_key, metric)

        if metric not in METRICS:
            raise RuntimeError(f"Métrique inconnue : {metric!r}")

        if metric_key in seen_metrics:
            raise RuntimeError(
                "Métrique dupliquée : "
                + " ".join(metric_key)
            )

        seen_metrics.add(metric_key)

        combined = by_slot.setdefault(
            slot_key,
            {
                "screenshot_id": slot_key[0],
                "side": slot_key[1],
                "slot": slot_key[2],
                "slot_status": row.get("slot_status", ""),
                "hero_uid": row.get("hero_uid", ""),
                "hero_name": row.get("hero_name", ""),
            },
        )

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
        is_empty_slot(row)
        for row in final_rows
    )

    hero_numeric_rows = [
        row
        for row in final_rows
        if not is_empty_slot(row)
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
        if is_empty_slot(row):
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

    validation = {
        "numeric_rows": len(final_rows),
        "hero_numeric_rows": len(hero_numeric_rows),
        "empty_numeric_rows": empty_numeric_rows,
        "slots": len(consolidated_rows),
        "hero_slots": sum(
            not is_empty_slot(row)
            for row in consolidated_rows
        ),
        "empty_slots": sum(
            is_empty_slot(row)
            for row in consolidated_rows
        ),
        "manual_values": sum(
            row.get("manual_reviewed") == "1"
            for row in final_rows
        ),
    }

    expected = {
        "numeric_rows": EXPECTED_NUMERIC_ROWS,
        "hero_numeric_rows": EXPECTED_HERO_NUMERIC_ROWS,
        "empty_numeric_rows": EXPECTED_EMPTY_NUMERIC_ROWS,
        "slots": EXPECTED_SLOTS,
        "hero_slots": EXPECTED_HERO_SLOTS,
        "empty_slots": EXPECTED_EMPTY_SLOTS,
        "manual_values": EXPECTED_MANUAL_VALUES,
    }

    differences = [
        f"{field}: attendu {expected[field]}, obtenu {validation[field]}"
        for field in expected
        if validation[field] != expected[field]
    ]

    if differences:
        raise RuntimeError(
            "Comptages finaux inattendus :\n"
            + "\n".join("- " + item for item in differences)
        )

    return validation


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

        if len(MANUAL_VALUES) != EXPECTED_MANUAL_VALUES:
            raise RuntimeError(
                "Le nombre de décisions manuelles intégré au script "
                f"est {len(MANUAL_VALUES)}, attendu "
                f"{EXPECTED_MANUAL_VALUES}."
            )

        validate_source_uniqueness(source_rows)
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

    print("FINALISATION DES STATISTIQUES — HERO_BATCH_003")
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
            "STATISTIQUES VALIDÉES — HERO_BATCH_003",
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
            "Important : 57 valeurs ambiguës ont été vérifiées visuellement.",
            "Les 3 939 autres valeurs de héros proviennent de l'arbitrage "
            "automatique V3.",
            "Les 4 valeurs de l'emplacement vide sont conservées comme vides.",
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
            "python scripts/finalize_battle_stats_batch_003.py --apply"
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
