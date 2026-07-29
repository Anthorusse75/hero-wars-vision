from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


BATCH_NAME = "hero_batch_005"

BATCH_ROOT = Path("data/batches") / BATCH_NAME
RECONCILIATION_CSV = (
    BATCH_ROOT
    / "reports"
    / "reconciliation_v2"
    / "reconciliation_results.csv"
)
HEROES_CSV = Path("data/catalog/heroes.csv")

VALIDATED_DIR = BATCH_ROOT / "validated"
MANUAL_DECISIONS_CSV = VALIDATED_DIR / "manual_decisions.csv"
SLOT_IDENTITY_MANIFEST_CSV = VALIDATED_DIR / "slot_identity_manifest.csv"
EVALUATION_SUMMARY = VALIDATED_DIR / "independent_evaluation_summary.txt"

EXPECTED_SLOT_COUNT = 10_000
EXPECTED_HERO_COUNT = 9_999
EXPECTED_EMPTY_COUNT = 1

MANUAL_DECISIONS = {
    ("27605", "L", "5"): {
        "slot_status": "HERO",
        "hero_uid": "HW_HERO_0033",
        "hero_name": "Maya",
        "reason": (
            "L'avatar correspond à Maya et le texte Mava est une "
            "erreur OCR d'un caractère."
        ),
    },
    ("28113", "L", "5"): {
        "slot_status": "HERO",
        "hero_uid": "HW_HERO_0049",
        "hero_name": "Tempus",
        "reason": (
            "L'avatar correspond à Tempus. Tamnuc est une mauvaise "
            "lecture OCR et ne doit pas devenir un alias."
        ),
    },
    ("29154", "R", "3"): {
        "slot_status": "HERO",
        "hero_uid": "HW_HERO_0033",
        "hero_name": "Maya",
        "reason": (
            "L'avatar et le nom affiché correspondent à Maya."
        ),
    },
    ("32382", "R", "4"): {
        "slot_status": "HERO",
        "hero_uid": "HW_HERO_0003",
        "hero_name": "Astaroth",
        "reason": (
            "L'avatar et le début du nom lisible correspondent à Astaroth."
        ),
    },
    ("33475", "L", "5"): {
        "slot_status": "EMPTY",
        "hero_uid": "",
        "hero_name": "",
        "reason": (
            "La cinquième position de l'équipe gauche est vide sur "
            "la capture complète."
        ),
    },
    ("33667", "R", "5"): {
        "slot_status": "HERO",
        "hero_uid": "HW_HERO_0034",
        "hero_name": "Miu",
        "reason": (
            "L'avatar correspond à Miu ; Min est une erreur OCR."
        ),
    },
    ("36276", "L", "5"): {
        "slot_status": "HERO",
        "hero_uid": "HW_HERO_0017",
        "hero_name": "Guus",
        "reason": (
            "L'avatar correspond à Guus malgré l'absence de nom OCR."
        ),
    },
    ("36276", "R", "3"): {
        "slot_status": "HERO",
        "hero_uid": "HW_HERO_0042",
        "hero_name": "Polaris",
        "reason": (
            "L'avatar correspond à Polaris ; Dalaric est une lecture OCR "
            "erronée."
        ),
    },
    ("36276", "R", "4"): {
        "slot_status": "HERO",
        "hero_uid": "HW_HERO_0011",
        "hero_name": "Dorian",
        "reason": (
            "L'avatar correspond à Dorian malgré l'absence de nom OCR."
        ),
    },
    ("36276", "R", "5"): {
        "slot_status": "HERO",
        "hero_uid": "HW_HERO_0049",
        "hero_name": "Tempus",
        "reason": (
            "L'avatar correspond à Tempus malgré l'absence de nom OCR."
        ),
    },
    ("37792", "R", "2"): {
        "slot_status": "HERO",
        "hero_uid": "HW_HERO_0033",
        "hero_name": "Maya",
        "reason": (
            "L'avatar correspond à Maya et le texte Mava est une "
            "erreur OCR d'un caractère."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalise les identités restantes du batch hero_batch_005, "
            "y compris un emplacement vide."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Écrit réellement les fichiers. Sans cette option, "
            "le script effectue uniquement une simulation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Autorise le remplacement de fichiers validés déjà présents, "
            "avec sauvegarde préalable."
        ),
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
        fields = list(reader.fieldnames or [])
        rows = [
            {
                key: value or ""
                for key, value in row.items()
            }
            for row in reader
        ]

    if not fields:
        raise RuntimeError(f"En-tête CSV absent : {path}")

    return fields, rows


def write_csv_atomic(
    path: Path,
    fields: list[str],
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
            fieldnames=fields,
            delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def review_required(row: dict[str, str]) -> bool:
    return str(
        row.get("review_required") or ""
    ).strip().casefold() in {
        "1",
        "true",
        "yes",
        "oui",
    }


def make_key(
    row: dict[str, str],
) -> tuple[str, str, str]:
    return (
        str(row.get("screenshot_id") or "").strip(),
        str(row.get("side") or "").strip().upper(),
        str(row.get("slot") or "").strip(),
    )


def validate_catalog(
    hero_rows: list[dict[str, str]],
) -> None:
    by_uid = {
        row.get("hero_uid", ""): row.get("reference_name", "")
        for row in hero_rows
    }

    for decision in MANUAL_DECISIONS.values():
        if decision["slot_status"] == "EMPTY":
            continue

        uid = decision["hero_uid"]
        expected_name = decision["hero_name"]
        actual_name = by_uid.get(uid)

        if actual_name is None:
            raise RuntimeError(
                f"Héros absent du catalogue : {uid} ({expected_name})."
            )

        if actual_name != expected_name:
            raise RuntimeError(
                f"{uid} est nommé {actual_name!r}, "
                f"attendu {expected_name!r}."
            )


def validate_review_cases(
    rows: list[dict[str, str]],
) -> None:
    found = {
        make_key(row)
        for row in rows
        if review_required(row)
    }
    expected = set(MANUAL_DECISIONS)

    if found != expected:
        missing = sorted(expected.difference(found))
        unexpected = sorted(found.difference(expected))

        details: list[str] = [
            "Les cas restant à revoir ne correspondent pas exactement "
            "aux onze décisions validées."
        ]

        if missing:
            details.append(
                "Décisions attendues mais absentes : "
                + ", ".join(
                    f"{key[0]} {key[1]}{key[2]}"
                    for key in missing
                )
            )

        if unexpected:
            details.append(
                "Cas supplémentaires trouvés : "
                + ", ".join(
                    f"{key[0]} {key[1]}{key[2]}"
                    for key in unexpected
                )
            )

        raise RuntimeError("\n".join(details))


def patch_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows_by_key = {
        make_key(row): row
        for row in rows
    }

    if len(rows_by_key) != len(rows):
        raise RuntimeError(
            "Le fichier de réconciliation contient des emplacements dupliqués."
        )

    for row in rows:
        if not str(row.get("slot_status") or "").strip():
            row["slot_status"] = "HERO"

    manual_rows: list[dict[str, str]] = []

    for key, decision in MANUAL_DECISIONS.items():
        row = rows_by_key.get(key)

        if row is None:
            raise RuntimeError(
                "Ligne de réconciliation absente : "
                f"{key[0]} {key[1]}{key[2]}"
            )

        slot_status = decision["slot_status"]
        hero_uid = decision["hero_uid"]
        hero_name = decision["hero_name"]

        row["slot_status"] = slot_status
        row["review_required"] = "0"

        if slot_status == "EMPTY":
            row["decision"] = "EMPTY_SLOT"
            row["final_hero_uid"] = ""
            row["final_hero_name"] = ""

            for field in (
                "ocr_matched_hero_uid",
                "ocr_matched_hero_name",
                "ocr_matched_alias",
                "ocr_alias_key",
                "ocr_cleaned_alias_key",
            ):
                if field in row:
                    row[field] = ""

            if "ocr_match_method" in row:
                row["ocr_match_method"] = "MANUAL_EMPTY_SLOT"
            if "ocr_match_score" in row:
                row["ocr_match_score"] = "1.0"

            output_decision = "EMPTY_SLOT"
        else:
            row["decision"] = "MANUAL_VISUAL_REVIEW"
            row["final_hero_uid"] = hero_uid
            row["final_hero_name"] = hero_name

            if "ocr_matched_hero_uid" in row:
                row["ocr_matched_hero_uid"] = hero_uid
            if "ocr_matched_hero_name" in row:
                row["ocr_matched_hero_name"] = hero_name
            if "ocr_matched_alias" in row:
                row["ocr_matched_alias"] = hero_name
            if "ocr_match_method" in row:
                row["ocr_match_method"] = "MANUAL_VISUAL_REVIEW"
            if "ocr_match_score" in row:
                row["ocr_match_score"] = "1.0"
            if "ocr_cleaned_alias_key" in row:
                row["ocr_cleaned_alias_key"] = hero_name.casefold()

            output_decision = "MANUAL_VISUAL_REVIEW"

        manual_rows.append(
            {
                "screenshot_id": key[0],
                "side": key[1],
                "slot": key[2],
                "slot_status": slot_status,
                "decision": output_decision,
                "hero_uid": hero_uid,
                "hero_name": hero_name,
                "reason": decision["reason"],
            }
        )

    return manual_rows


def validate_final_state(
    rows: list[dict[str, str]],
) -> tuple[int, int, int]:
    remaining_review = sum(
        review_required(row)
        for row in rows
    )

    empty_count = sum(
        str(row.get("slot_status") or "").strip().upper()
        == "EMPTY"
        for row in rows
    )

    identified_count = sum(
        bool(str(row.get("final_hero_uid") or "").strip())
        for row in rows
    )

    unidentified_non_empty = [
        make_key(row)
        for row in rows
        if (
            str(row.get("slot_status") or "").strip().upper()
            != "EMPTY"
            and not str(row.get("final_hero_uid") or "").strip()
        )
    ]

    empty_with_identity = [
        make_key(row)
        for row in rows
        if (
            str(row.get("slot_status") or "").strip().upper()
            == "EMPTY"
            and (
                str(row.get("final_hero_uid") or "").strip()
                or str(row.get("final_hero_name") or "").strip()
            )
        )
    ]

    problems: list[str] = []

    if len(rows) != EXPECTED_SLOT_COUNT:
        problems.append(
            f"{EXPECTED_SLOT_COUNT} lignes attendues, "
            f"{len(rows)} trouvées."
        )

    if identified_count != EXPECTED_HERO_COUNT:
        problems.append(
            f"{EXPECTED_HERO_COUNT} héros identifiés attendus, "
            f"{identified_count} trouvés."
        )

    if empty_count != EXPECTED_EMPTY_COUNT:
        problems.append(
            f"{EXPECTED_EMPTY_COUNT} emplacement vide attendu, "
            f"{empty_count} trouvé(s)."
        )

    if remaining_review:
        problems.append(
            f"{remaining_review} cas restent à revoir."
        )

    if unidentified_non_empty:
        problems.append(
            f"{len(unidentified_non_empty)} emplacement(s) non vide(s) "
            "restent sans identité."
        )

    if empty_with_identity:
        problems.append(
            f"{len(empty_with_identity)} emplacement(s) vide(s) "
            "possèdent encore une identité finale."
        )

    if problems:
        raise RuntimeError(
            "État final invalide :\n"
            + "\n".join(
                "- " + problem
                for problem in problems
            )
        )

    return identified_count, empty_count, remaining_review


def backup_existing(
    paths: list[Path],
) -> Path | None:
    existing = [
        path
        for path in paths
        if path.exists()
    ]

    if not existing:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = (
        VALIDATED_DIR
        / "backups"
        / f"identity_finalization_{timestamp}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)

    for path in existing:
        shutil.copy2(path, backup_dir / path.name)

    return backup_dir


def main() -> int:
    args = parse_args()

    output_paths = [
        MANUAL_DECISIONS_CSV,
        SLOT_IDENTITY_MANIFEST_CSV,
        EVALUATION_SUMMARY,
    ]

    existing_outputs = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing_outputs and not args.overwrite:
        print(
            "Des fichiers d'identité validés existent déjà. "
            "Relance avec --overwrite pour les remplacer.",
            file=sys.stderr,
        )
        return 2

    try:
        reconciliation_fields, rows = read_csv(RECONCILIATION_CSV)
        _, hero_rows = read_csv(HEROES_CSV)

        if "slot_status" not in reconciliation_fields:
            reconciliation_fields.append("slot_status")

        validate_catalog(hero_rows)
        validate_review_cases(rows)
        manual_rows = patch_rows(rows)

        (
            identified_count,
            empty_count,
            remaining_review,
        ) = validate_final_state(rows)

    except (RuntimeError, OSError, csv.Error, ValueError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    decision_counts = Counter(
        str(row.get("decision") or "")
        for row in rows
    )

    print("FINALISATION DES IDENTITÉS — HERO_BATCH_005")
    print("=" * 72)
    print()
    print(f"Emplacements analysés : {len(rows)}")
    print(f"Héros identifiés       : {identified_count}")
    print(f"Emplacements vides     : {empty_count}")
    print(f"Décisions manuelles    : {len(manual_rows)}")
    print(f"Cas restant à revoir   : {remaining_review}")
    print()
    print("Décisions manuelles :")

    for row in manual_rows:
        if row["slot_status"] == "EMPTY":
            label = "EMPLACEMENT VIDE"
        else:
            label = (
                f"{row['hero_name']} ({row['hero_uid']})"
            )

        print(
            f"- {row['screenshot_id']} "
            f"{row['side']}{row['slot']} : {label}"
        )

    summary_lines = [
        "ÉVALUATION DES IDENTITÉS — HERO_BATCH_005",
        "=" * 72,
        "",
        f"Emplacements analysés : {len(rows)}",
        f"Héros identifiés : {identified_count}",
        f"Emplacements vides : {empty_count}",
        f"Décisions manuelles finales : {len(manual_rows)}",
        f"Cas restant à revoir : {remaining_review}",
        "",
        "Origine des décisions finales :",
    ]

    for decision, count in sorted(
        decision_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        summary_lines.append(
            f"- {decision or 'SANS_DECISION'} : {count}"
        )

    summary_lines.extend(
        [
            "",
            "Décisions manuelles finales :",
        ]
    )

    for row in manual_rows:
        if row["slot_status"] == "EMPTY":
            label = "EMPLACEMENT VIDE"
        else:
            label = (
                f"{row['hero_name']} ({row['hero_uid']})"
            )

        summary_lines.append(
            f"- {row['screenshot_id']} "
            f"{row['side']}{row['slot']} : {label}"
        )

    summary_lines.extend(
        [
            "",
            "Remarque : les onze cas restants ont été vérifiés "
            "sur les avatars, les noms et les captures complètes.",
            "L'emplacement 33475 L5 est explicitement marqué EMPTY "
            "afin que les quatre statistiques correspondantes soient "
            "ignorées par les étapes OCR.",
            "",
        ]
    )

    if not args.apply:
        print()
        print("MODE SIMULATION : aucun fichier n'a été modifié.")
        print()
        print("Pour appliquer :")
        print(
            "python scripts/finalize_hero_batch_005.py --apply"
        )
        return 0

    backup_dir = backup_existing(
        [RECONCILIATION_CSV, *output_paths]
    )

    write_csv_atomic(
        RECONCILIATION_CSV,
        reconciliation_fields,
        rows,
    )

    write_csv_atomic(
        SLOT_IDENTITY_MANIFEST_CSV,
        reconciliation_fields,
        rows,
    )

    manual_fields = [
        "screenshot_id",
        "side",
        "slot",
        "slot_status",
        "decision",
        "hero_uid",
        "hero_name",
        "reason",
    ]

    write_csv_atomic(
        MANUAL_DECISIONS_CSV,
        manual_fields,
        manual_rows,
    )

    EVALUATION_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    EVALUATION_SUMMARY.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("Finalisation appliquée.")
    print(f"Manifeste final : {SLOT_IDENTITY_MANIFEST_CSV}")
    print(f"Décisions manuelles : {MANUAL_DECISIONS_CSV}")
    print(f"Résumé : {EVALUATION_SUMMARY}")

    if backup_dir is not None:
        print(f"Sauvegarde : {backup_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
