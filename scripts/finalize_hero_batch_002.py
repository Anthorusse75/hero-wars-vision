from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any


BATCH_NAME = "hero_batch_002"
SCREENSHOT_ID = "2819"
CHABBA_SIDE = "R"
CHABBA_SLOT = "4"
EMPTY_SIDE = "R"
EMPTY_SLOT = "5"
CHABBA_NAME = "Chabba"

CATALOG_DIR = Path("data/catalog")
HEROES_CSV = CATALOG_DIR / "heroes.csv"
APPEARANCES_CSV = CATALOG_DIR / "hero_appearances.csv"
ALIASES_CSV = CATALOG_DIR / "hero_name_aliases.csv"
AVATAR_MANIFEST_CSV = CATALOG_DIR / "hero_avatar_manifest.csv"

BATCH_ROOT = Path("data/batches") / BATCH_NAME
RECONCILIATION_CSV = (
    BATCH_ROOT
    / "reports"
    / "reconciliation_v2"
    / "reconciliation_results.csv"
)

SOURCE_AVATAR = (
    BATCH_ROOT
    / "crops_dynamic_v1"
    / "avatars_inner"
    / "2819_Screenshot_20250709-175304__R4.png"
)

CANONICAL_AVATAR_DIR = Path("data/crops/hero/avatars_inner")
DESTINATION_AVATAR = CANONICAL_AVATAR_DIR / SOURCE_AVATAR.name

VALIDATED_DIR = BATCH_ROOT / "validated"
MANUAL_DECISIONS_CSV = (
    VALIDATED_DIR / "manual_decisions.csv"
)
SLOT_IDENTITY_MANIFEST_CSV = (
    VALIDATED_DIR / "slot_identity_manifest.csv"
)
EVALUATION_SUMMARY = (
    VALIDATED_DIR / "independent_evaluation_summary.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalise hero_batch_002 : ajoute Chabba au catalogue "
            "et marque l'emplacement R5 de la capture 2819 comme vide."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Applique réellement les modifications. "
            "Sans cette option, seule une simulation est effectuée."
        ),
    )
    return parser.parse_args()


def normalize_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)

    return "".join(
        character
        for character in value.casefold()
        if character.isalnum()
        and not unicodedata.combining(character)
    )


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    characters: list[str] = []

    for character in value.casefold():
        if unicodedata.combining(character):
            continue

        if character.isalnum():
            characters.append(character)
        else:
            characters.append("_")

    slug = re.sub(r"_+", "_", "".join(characters))
    return slug.strip("_") or "hero"


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


def write_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

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


def next_hero_uid(
    heroes: list[dict[str, str]],
) -> str:
    maximum = 0

    for row in heroes:
        match = re.fullmatch(
            r"HW_HERO_(\d+)",
            row.get("hero_uid", ""),
        )

        if match:
            maximum = max(
                maximum,
                int(match.group(1)),
            )

    return f"HW_HERO_{maximum + 1:04d}"


def unique_appearance_id(
    appearances: list[dict[str, str]],
    base_slug: str,
) -> str:
    existing = {
        row.get("appearance_id", "")
        for row in appearances
    }

    index = 1

    while True:
        candidate = (
            f"{base_slug}__appearance_{index:02d}"
        )

        if candidate not in existing:
            return candidate

        index += 1


def find_reconciliation_row(
    rows: list[dict[str, str]],
    side: str,
    slot: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get("screenshot_id") == SCREENSHOT_ID
        and row.get("side") == side
        and row.get("slot") == slot
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Ligne de réconciliation introuvable ou ambiguë : "
            f"{SCREENSHOT_ID} {side}{slot}"
        )

    return matches[0]


def ensure_chabba_hero(
    heroes: list[dict[str, str]],
) -> tuple[str, bool]:
    matches = [
        row
        for row in heroes
        if normalize_key(
            row.get("reference_name", "")
        )
        == normalize_key(CHABBA_NAME)
    ]

    if len(matches) > 1:
        raise RuntimeError(
            "Plusieurs héros Chabba existent déjà."
        )

    if matches:
        hero_uid = matches[0]["hero_uid"]
        return hero_uid, False

    hero_uid = next_hero_uid(heroes)

    heroes.append(
        {
            "hero_uid": hero_uid,
            "reference_name": CHABBA_NAME,
            "provisional_key": slugify(CHABBA_NAME),
            "reviewed": "1",
            "notes": (
                "Ajouté après revue humaine de "
                "hero_batch_002, capture 2819 R4."
            ),
        }
    )

    return hero_uid, True


def ensure_chabba_appearance(
    appearances: list[dict[str, str]],
    hero_uid: str,
) -> tuple[str, bool]:
    hero_appearances = [
        row
        for row in appearances
        if row.get("hero_uid") == hero_uid
    ]

    for row in hero_appearances:
        sample_files = {
            value.strip()
            for value in row.get(
                "sample_files",
                "",
            ).split("|")
            if value.strip()
        }

        if SOURCE_AVATAR.name in sample_files:
            return row["appearance_id"], False

    appearance_id = unique_appearance_id(
        appearances,
        slugify(CHABBA_NAME),
    )

    appearances.append(
        {
            "appearance_id": appearance_id,
            "hero_uid": hero_uid,
            "technical_cluster_id": "",
            "avatar_count": "1",
            "sample_files": SOURCE_AVATAR.name,
            "appearance_type": "unknown_variant",
            "reviewed": "1",
            "notes": (
                "Première apparence validée depuis "
                "hero_batch_002, capture 2819 R4."
            ),
        }
    )

    return appearance_id, True


def ensure_chabba_alias(
    aliases: list[dict[str, str]],
    hero_uid: str,
) -> bool:
    matches = [
        row
        for row in aliases
        if normalize_key(row.get("alias", ""))
        == normalize_key(CHABBA_NAME)
    ]

    conflicting = [
        row
        for row in matches
        if row.get("hero_uid") != hero_uid
    ]

    if conflicting:
        raise RuntimeError(
            "L'alias Chabba appartient déjà "
            "à un autre héros."
        )

    if matches:
        row = matches[0]
        row["reviewed"] = "1"
        row["occurrences"] = str(
            max(
                int(row.get("occurrences") or 0),
                1,
            )
        )
        return False

    aliases.append(
        {
            "hero_uid": hero_uid,
            "alias": CHABBA_NAME,
            "language": "fr",
            "source": "manual_review_batch_002",
            "occurrences": "1",
            "reviewed": "1",
        }
    )

    return True


def ensure_avatar_manifest_entry(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    hero_uid: str,
    appearance_id: str,
    reconciliation_row: dict[str, str],
) -> bool:
    required_fields = {
        "avatar_file",
        "hero_uid",
        "appearance_id",
        "screenshot_id",
        "side",
        "slot",
    }

    missing = required_fields.difference(
        fieldnames
    )

    if missing:
        raise RuntimeError(
            "Colonnes absentes du manifeste maître : "
            + ", ".join(sorted(missing))
        )

    matches = [
        row
        for row in rows
        if row.get("avatar_file")
        == SOURCE_AVATAR.name
    ]

    if matches:
        row = matches[0]
        row["hero_uid"] = hero_uid
        row["appearance_id"] = appearance_id
        row["reviewed"] = "1"
        return False

    new_row = {
        field: ""
        for field in fieldnames
    }

    values = {
        "avatar_file": SOURCE_AVATAR.name,
        "hero_uid": hero_uid,
        "appearance_id": appearance_id,
        "ocr_text": reconciliation_row.get(
            "ocr_text",
            CHABBA_NAME,
        ),
        "ocr_confidence": reconciliation_row.get(
            "ocr_confidence",
            "",
        ),
        "ocr_status": reconciliation_row.get(
            "ocr_status",
            "HIGH",
        ),
        "screenshot_id": SCREENSHOT_ID,
        "side": CHABBA_SIDE,
        "slot": CHABBA_SLOT,
        "label_source": (
            "manual_visual_and_ocr_batch_002"
        ),
        "reviewed": "1",
    }

    for key, value in values.items():
        if key in new_row:
            new_row[key] = value

    rows.append(new_row)
    return True


def patch_reconciliation(
    rows: list[dict[str, str]],
    hero_uid: str,
    appearance_id: str,
) -> None:
    chabba = find_reconciliation_row(
        rows,
        CHABBA_SIDE,
        CHABBA_SLOT,
    )
    empty = find_reconciliation_row(
        rows,
        EMPTY_SIDE,
        EMPTY_SLOT,
    )

    chabba["decision"] = "CURATED_NEW_HERO"
    chabba["final_hero_uid"] = hero_uid
    chabba["final_hero_name"] = CHABBA_NAME
    chabba["review_required"] = "0"

    if "ocr_matched_hero_uid" in chabba:
        chabba["ocr_matched_hero_uid"] = hero_uid
    if "ocr_matched_hero_name" in chabba:
        chabba["ocr_matched_hero_name"] = CHABBA_NAME
    if "ocr_matched_alias" in chabba:
        chabba["ocr_matched_alias"] = CHABBA_NAME
    if "ocr_match_method" in chabba:
        chabba["ocr_match_method"] = (
            "MANUAL_CURATION_NEW_HERO"
        )
    if "ocr_match_score" in chabba:
        chabba["ocr_match_score"] = "1.0"

    empty["decision"] = "EMPTY_SLOT"
    empty["final_hero_uid"] = ""
    empty["final_hero_name"] = ""
    empty["review_required"] = "0"

    if "appearance_id" in chabba:
        chabba["appearance_id"] = appearance_id


def build_manual_decisions(
    hero_uid: str,
    appearance_id: str,
) -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "screenshot_id",
        "side",
        "slot",
        "slot_status",
        "validated_hero_uid",
        "validated_hero_name",
        "appearance_id",
        "decision_source",
        "notes",
    ]

    rows = [
        {
            "screenshot_id": SCREENSHOT_ID,
            "side": CHABBA_SIDE,
            "slot": CHABBA_SLOT,
            "slot_status": "HERO",
            "validated_hero_uid": hero_uid,
            "validated_hero_name": CHABBA_NAME,
            "appearance_id": appearance_id,
            "decision_source": "human_review",
            "notes": (
                "Portrait et nom Chabba visibles "
                "sur la capture complète."
            ),
        },
        {
            "screenshot_id": SCREENSHOT_ID,
            "side": EMPTY_SIDE,
            "slot": EMPTY_SLOT,
            "slot_status": "EMPTY",
            "validated_hero_uid": "",
            "validated_hero_name": "",
            "appearance_id": "",
            "decision_source": "human_review",
            "notes": (
                "Aucun cinquième héros présent "
                "dans l'équipe de droite."
            ),
        },
    ]

    return fields, rows


def build_slot_manifest(
    reconciliation_rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "screenshot_id",
        "side",
        "slot",
        "avatar_file",
        "slot_status",
        "final_hero_uid",
        "final_hero_name",
        "decision",
        "review_required",
        "curation_source",
    ]

    rows: list[dict[str, str]] = []

    for row in reconciliation_rows:
        is_empty = (
            row.get("screenshot_id") == SCREENSHOT_ID
            and row.get("side") == EMPTY_SIDE
            and row.get("slot") == EMPTY_SLOT
        )

        if is_empty:
            slot_status = "EMPTY"
            curation_source = "human_review"
        elif row.get("final_hero_uid"):
            slot_status = "HERO"
            curation_source = (
                "human_review"
                if row.get("decision")
                == "CURATED_NEW_HERO"
                else "visual_ocr_pipeline"
            )
        else:
            slot_status = "UNRESOLVED"
            curation_source = ""

        rows.append(
            {
                "screenshot_id": row.get(
                    "screenshot_id",
                    "",
                ),
                "side": row.get("side", ""),
                "slot": row.get("slot", ""),
                "avatar_file": row.get(
                    "avatar_file",
                    "",
                ),
                "slot_status": slot_status,
                "final_hero_uid": row.get(
                    "final_hero_uid",
                    "",
                ),
                "final_hero_name": row.get(
                    "final_hero_name",
                    "",
                ),
                "decision": row.get(
                    "decision",
                    "",
                ),
                "review_required": row.get(
                    "review_required",
                    "",
                ),
                "curation_source": curation_source,
            }
        )

    return fields, rows


def build_evaluation_summary(
    reconciliation_rows: list[dict[str, str]],
) -> str:
    total_slots = len(reconciliation_rows)
    empty_slots = sum(
        1
        for row in reconciliation_rows
        if row.get("decision") == "EMPTY_SLOT"
    )
    hero_slots = total_slots - empty_slots
    identified_hero_slots = sum(
        1
        for row in reconciliation_rows
        if row.get("final_hero_uid")
    )
    review_count = sum(
        int(row.get("review_required") or 0)
        for row in reconciliation_rows
    )

    coverage = (
        identified_hero_slots
        / hero_slots
        * 100
        if hero_slots
        else 0.0
    )

    return "\n".join(
        [
            "ÉVALUATION INDÉPENDANTE — HERO_BATCH_002",
            "=" * 72,
            "",
            f"Emplacements extraits : {total_slots}",
            f"Emplacements contenant un héros : {hero_slots}",
            f"Emplacements vides : {empty_slots}",
            (
                "Identités finales attribuées aux héros : "
                f"{identified_hero_slots}"
            ),
            (
                "Couverture d'identification après curation : "
                f"{coverage:.2f} %"
            ),
            f"Cas restant à revoir : {review_count}",
            "",
            "Cas découvert pendant l'évaluation :",
            "- 2819 R4 : Chabba, absent du catalogue initial.",
            "- 2819 R5 : emplacement vide, et non un héros inconnu.",
            "",
            "Important : cette valeur mesure la couverture d'identification.",
            "Elle ne constitue pas encore une précision intégralement vérifiée",
            "par annotation humaine de chacun des 999 héros.",
            "",
        ]
    )


def backup_files(
    paths: list[Path],
) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    backup_dir = (
        CATALOG_DIR
        / "backups"
        / f"finalize_batch_002_{timestamp}"
    )
    backup_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    for path in paths:
        if path.exists():
            relative_name = path.as_posix().replace(
                "/",
                "__",
            )
            shutil.copy2(
                path,
                backup_dir / relative_name,
            )

    return backup_dir


def main() -> int:
    args = parse_args()

    required_paths = [
        HEROES_CSV,
        APPEARANCES_CSV,
        ALIASES_CSV,
        AVATAR_MANIFEST_CSV,
        RECONCILIATION_CSV,
        SOURCE_AVATAR,
    ]

    missing = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        print(
            "Fichiers absents :",
            file=sys.stderr,
        )

        for path in missing:
            print(
                f"- {path}",
                file=sys.stderr,
            )

        return 1

    try:
        (
            hero_fields,
            heroes,
        ) = read_csv(HEROES_CSV)
        (
            appearance_fields,
            appearances,
        ) = read_csv(APPEARANCES_CSV)
        (
            alias_fields,
            aliases,
        ) = read_csv(ALIASES_CSV)
        (
            avatar_fields,
            avatar_rows,
        ) = read_csv(AVATAR_MANIFEST_CSV)
        (
            reconciliation_fields,
            reconciliation_rows,
        ) = read_csv(RECONCILIATION_CSV)

        hero_uid, hero_created = (
            ensure_chabba_hero(heroes)
        )
        appearance_id, appearance_created = (
            ensure_chabba_appearance(
                appearances,
                hero_uid,
            )
        )
        alias_created = ensure_chabba_alias(
            aliases,
            hero_uid,
        )

        chabba_reconciliation = (
            find_reconciliation_row(
                reconciliation_rows,
                CHABBA_SIDE,
                CHABBA_SLOT,
            )
        )

        avatar_created = (
            ensure_avatar_manifest_entry(
                avatar_rows,
                avatar_fields,
                hero_uid,
                appearance_id,
                chabba_reconciliation,
            )
        )

        patch_reconciliation(
            reconciliation_rows,
            hero_uid,
            appearance_id,
        )

        (
            manual_fields,
            manual_rows,
        ) = build_manual_decisions(
            hero_uid,
            appearance_id,
        )

        (
            slot_fields,
            slot_rows,
        ) = build_slot_manifest(
            reconciliation_rows
        )

        summary = build_evaluation_summary(
            reconciliation_rows
        )

    except (
        RuntimeError,
        ValueError,
        csv.Error,
    ) as error:
        print(
            f"Erreur : {error}",
            file=sys.stderr,
        )
        return 1

    print("Finalisation de hero_batch_002")
    print()
    print(
        f"Héros Chabba : {hero_uid} "
        f"({'nouveau' if hero_created else 'déjà présent'})"
    )
    print(
        f"Apparence : {appearance_id} "
        f"({'nouvelle' if appearance_created else 'déjà présente'})"
    )
    print(
        f"Alias Chabba : "
        f"{'ajouté' if alias_created else 'déjà présent'}"
    )
    print(
        f"Avatar de référence : "
        f"{'à ajouter' if avatar_created else 'déjà présent'}"
    )
    print(
        "2819 R5 : EMPTY_SLOT"
    )
    print()
    print(summary)

    if not args.apply:
        print(
            "MODE SIMULATION : aucun fichier "
            "n'a été modifié."
        )
        print()
        print("Pour appliquer :")
        print(
            "python scripts/finalize_hero_batch_002.py --apply"
        )
        return 0

    backup_dir = backup_files(
        [
            HEROES_CSV,
            APPEARANCES_CSV,
            ALIASES_CSV,
            AVATAR_MANIFEST_CSV,
            RECONCILIATION_CSV,
        ]
    )

    write_csv_atomic(
        HEROES_CSV,
        hero_fields,
        heroes,
    )
    write_csv_atomic(
        APPEARANCES_CSV,
        appearance_fields,
        appearances,
    )
    write_csv_atomic(
        ALIASES_CSV,
        alias_fields,
        aliases,
    )
    write_csv_atomic(
        AVATAR_MANIFEST_CSV,
        avatar_fields,
        avatar_rows,
    )
    write_csv_atomic(
        RECONCILIATION_CSV,
        reconciliation_fields,
        reconciliation_rows,
    )

    write_csv_atomic(
        MANUAL_DECISIONS_CSV,
        manual_fields,
        manual_rows,
    )
    write_csv_atomic(
        SLOT_IDENTITY_MANIFEST_CSV,
        slot_fields,
        slot_rows,
    )

    EVALUATION_SUMMARY.write_text(
        summary,
        encoding="utf-8",
    )

    CANONICAL_AVATAR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DESTINATION_AVATAR.exists():
        shutil.copy2(
            SOURCE_AVATAR,
            DESTINATION_AVATAR,
        )

    print()
    print("Finalisation appliquée.")
    print(f"Sauvegarde : {backup_dir}")
    print(
        f"Décisions : {MANUAL_DECISIONS_CSV}"
    )
    print(
        f"Manifeste des emplacements : "
        f"{SLOT_IDENTITY_MANIFEST_CSV}"
    )
    print(
        f"Résumé d'évaluation : "
        f"{EVALUATION_SUMMARY}"
    )
    print(
        f"Référence copiée : "
        f"{DESTINATION_AVATAR}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
