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


BATCH_NAME = "hero_batch_003"

ELMIR_SCREENSHOT_ID = "6900"
ELMIR_SIDE = "R"
ELMIR_SLOT = "3"
ELMIR_NAME = "Elmir"

EMPTY_SCREENSHOT_ID = "2820"
EMPTY_SIDE = "R"
EMPTY_SLOT = "5"

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
    / "6900_Screenshot_20250819_223339_Hero_Wars__R3.png"
)

CANONICAL_AVATAR_DIR = Path("data/crops/hero/avatars_inner")
DESTINATION_AVATAR = CANONICAL_AVATAR_DIR / SOURCE_AVATAR.name

VALIDATED_DIR = BATCH_ROOT / "validated"
MANUAL_DECISIONS_CSV = VALIDATED_DIR / "manual_decisions.csv"
SLOT_IDENTITY_MANIFEST_CSV = VALIDATED_DIR / "slot_identity_manifest.csv"
EVALUATION_SUMMARY = VALIDATED_DIR / "independent_evaluation_summary.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalise hero_batch_003 : ajoute Elmir au catalogue et marque "
            "2820 R5 comme emplacement vide."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Applique réellement les modifications. Sans cette option, "
            "seule une simulation est effectuée."
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
        candidate = f"{base_slug}__appearance_{index:02d}"

        if candidate not in existing:
            return candidate

        index += 1


def find_reconciliation_row(
    rows: list[dict[str, str]],
    screenshot_id: str,
    side: str,
    slot: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get("screenshot_id") == screenshot_id
        and row.get("side") == side
        and row.get("slot") == slot
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Ligne de réconciliation introuvable ou ambiguë : "
            f"{screenshot_id} {side}{slot}"
        )

    return matches[0]


def ensure_elmir_hero(
    heroes: list[dict[str, str]],
) -> tuple[str, bool]:
    matches = [
        row
        for row in heroes
        if normalize_key(
            row.get("reference_name", "")
        )
        == normalize_key(ELMIR_NAME)
    ]

    if len(matches) > 1:
        raise RuntimeError(
            "Plusieurs héros Elmir existent déjà."
        )

    if matches:
        return matches[0]["hero_uid"], False

    hero_uid = next_hero_uid(heroes)

    heroes.append(
        {
            "hero_uid": hero_uid,
            "reference_name": ELMIR_NAME,
            "provisional_key": slugify(ELMIR_NAME),
            "reviewed": "1",
            "notes": (
                "Ajouté après revue humaine de "
                "hero_batch_003, capture 6900 R3."
            ),
        }
    )

    return hero_uid, True


def ensure_elmir_appearance(
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
        slugify(ELMIR_NAME),
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
                "hero_batch_003, capture 6900 R3."
            ),
        }
    )

    return appearance_id, True


def ensure_elmir_alias(
    aliases: list[dict[str, str]],
    hero_uid: str,
) -> bool:
    matches = [
        row
        for row in aliases
        if normalize_key(
            row.get("alias", "")
        )
        == normalize_key(ELMIR_NAME)
    ]

    conflicting = [
        row
        for row in matches
        if row.get("hero_uid") != hero_uid
    ]

    if conflicting:
        raise RuntimeError(
            "L'alias Elmir appartient déjà à un autre héros."
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
            "alias": ELMIR_NAME,
            "language": "en",
            "source": "manual_review_batch_003",
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

    missing = required_fields.difference(fieldnames)

    if missing:
        raise RuntimeError(
            "Colonnes absentes du manifeste maître : "
            + ", ".join(sorted(missing))
        )

    matches = [
        row
        for row in rows
        if row.get("avatar_file") == SOURCE_AVATAR.name
    ]

    if matches:
        row = matches[0]
        row["hero_uid"] = hero_uid
        row["appearance_id"] = appearance_id

        if "reviewed" in row:
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
            ELMIR_NAME,
        ),
        "ocr_confidence": reconciliation_row.get(
            "ocr_confidence",
            "",
        ),
        "ocr_status": reconciliation_row.get(
            "ocr_status",
            "HIGH",
        ),
        "screenshot_id": ELMIR_SCREENSHOT_ID,
        "side": ELMIR_SIDE,
        "slot": ELMIR_SLOT,
        "label_source": "manual_visual_and_ocr_batch_003",
        "reviewed": "1",
    }

    for key, value in values.items():
        if key in new_row:
            new_row[key] = value

    rows.append(new_row)
    return True


def review_required(
    row: dict[str, str],
) -> bool:
    return str(
        row.get("review_required") or ""
    ).strip().casefold() in {
        "1",
        "true",
        "yes",
        "oui",
    }


def validate_remaining_reviews(
    rows: list[dict[str, str]],
) -> None:
    expected = {
        (
            ELMIR_SCREENSHOT_ID,
            ELMIR_SIDE,
            ELMIR_SLOT,
        ),
        (
            EMPTY_SCREENSHOT_ID,
            EMPTY_SIDE,
            EMPTY_SLOT,
        ),
    }

    found = {
        (
            row.get("screenshot_id", ""),
            row.get("side", ""),
            row.get("slot", ""),
        )
        for row in rows
        if review_required(row)
    }

    if found != expected:
        raise RuntimeError(
            "Les cas restant à revoir ne correspondent pas "
            "exactement à Elmir et à l'emplacement vide.\n"
            f"Attendus : {sorted(expected)}\n"
            f"Trouvés : {sorted(found)}"
        )


def patch_reconciliation(
    rows: list[dict[str, str]],
    hero_uid: str,
) -> None:
    elmir = find_reconciliation_row(
        rows,
        ELMIR_SCREENSHOT_ID,
        ELMIR_SIDE,
        ELMIR_SLOT,
    )

    empty = find_reconciliation_row(
        rows,
        EMPTY_SCREENSHOT_ID,
        EMPTY_SIDE,
        EMPTY_SLOT,
    )

    elmir["decision"] = "CURATED_NEW_HERO"
    elmir["final_hero_uid"] = hero_uid
    elmir["final_hero_name"] = ELMIR_NAME
    elmir["review_required"] = "0"

    if "ocr_matched_hero_uid" in elmir:
        elmir["ocr_matched_hero_uid"] = hero_uid
    if "ocr_matched_hero_name" in elmir:
        elmir["ocr_matched_hero_name"] = ELMIR_NAME
    if "ocr_matched_alias" in elmir:
        elmir["ocr_matched_alias"] = ELMIR_NAME
    if "ocr_match_method" in elmir:
        elmir["ocr_match_method"] = "MANUAL_CURATION_NEW_HERO"
    if "ocr_match_score" in elmir:
        elmir["ocr_match_score"] = "1.0"
    if "ocr_cleaned_alias_key" in elmir:
        elmir["ocr_cleaned_alias_key"] = normalize_key(ELMIR_NAME)

    empty["decision"] = "EMPTY_SLOT"
    empty["final_hero_uid"] = ""
    empty["final_hero_name"] = ""
    empty["review_required"] = "0"


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
            "screenshot_id": ELMIR_SCREENSHOT_ID,
            "side": ELMIR_SIDE,
            "slot": ELMIR_SLOT,
            "slot_status": "HERO",
            "validated_hero_uid": hero_uid,
            "validated_hero_name": ELMIR_NAME,
            "appearance_id": appearance_id,
            "decision_source": "human_review",
            "notes": (
                "Portrait et nom Elmir visibles "
                "sur la capture complète."
            ),
        },
        {
            "screenshot_id": EMPTY_SCREENSHOT_ID,
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
            row.get("screenshot_id") == EMPTY_SCREENSHOT_ID
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
                if row.get("decision") == "CURATED_NEW_HERO"
                else "visual_ocr_pipeline"
            )
        else:
            slot_status = "UNRESOLVED"
            curation_source = ""

        rows.append(
            {
                "screenshot_id": row.get("screenshot_id", ""),
                "side": row.get("side", ""),
                "slot": row.get("slot", ""),
                "avatar_file": row.get("avatar_file", ""),
                "slot_status": slot_status,
                "final_hero_uid": row.get("final_hero_uid", ""),
                "final_hero_name": row.get("final_hero_name", ""),
                "decision": row.get("decision", ""),
                "review_required": row.get("review_required", ""),
                "curation_source": curation_source,
            }
        )

    return fields, rows


def build_evaluation_summary(
    reconciliation_rows: list[dict[str, str]],
    hero_uid: str,
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
        identified_hero_slots / hero_slots * 100
        if hero_slots
        else 0.0
    )

    return "\n".join(
        [
            "ÉVALUATION INDÉPENDANTE — HERO_BATCH_003",
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
            "Cas découverts pendant l'évaluation :",
            (
                f"- 6900 R3 : Elmir ({hero_uid}), "
                "absent du catalogue initial."
            ),
            "- 2820 R5 : emplacement vide, et non un héros inconnu.",
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = (
        CATALOG_DIR
        / "backups"
        / f"finalize_batch_003_{timestamp}"
    )
    backup_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    for path in paths:
        if path.exists():
            shutil.copy2(
                path,
                backup_dir / path.name,
            )

    return backup_dir


def main() -> int:
    args = parse_args()

    try:
        hero_fields, heroes = read_csv(HEROES_CSV)
        appearance_fields, appearances = read_csv(APPEARANCES_CSV)
        alias_fields, aliases = read_csv(ALIASES_CSV)
        avatar_fields, avatar_rows = read_csv(AVATAR_MANIFEST_CSV)
        reconciliation_fields, reconciliation_rows = read_csv(
            RECONCILIATION_CSV
        )

        if not SOURCE_AVATAR.exists():
            raise RuntimeError(
                f"Avatar source absent : {SOURCE_AVATAR}"
            )

        if len(reconciliation_rows) != 1000:
            raise RuntimeError(
                "Le manifeste de réconciliation doit contenir "
                f"1000 lignes, trouvé : {len(reconciliation_rows)}"
            )

        validate_remaining_reviews(reconciliation_rows)

        elmir_row = find_reconciliation_row(
            reconciliation_rows,
            ELMIR_SCREENSHOT_ID,
            ELMIR_SIDE,
            ELMIR_SLOT,
        )

        if normalize_key(elmir_row.get("ocr_text", "")) != normalize_key(
            ELMIR_NAME
        ):
            raise RuntimeError(
                "Le texte OCR du cas 6900 R3 n'est pas Elmir : "
                f"{elmir_row.get('ocr_text', '')!r}"
            )

        hero_uid, hero_added = ensure_elmir_hero(heroes)
        appearance_id, appearance_added = ensure_elmir_appearance(
            appearances,
            hero_uid,
        )
        alias_added = ensure_elmir_alias(
            aliases,
            hero_uid,
        )
        avatar_added = ensure_avatar_manifest_entry(
            avatar_rows,
            avatar_fields,
            hero_uid,
            appearance_id,
            elmir_row,
        )

        patch_reconciliation(
            reconciliation_rows,
            hero_uid,
        )

        remaining_review = sum(
            1
            for row in reconciliation_rows
            if review_required(row)
        )
        empty_count = sum(
            1
            for row in reconciliation_rows
            if row.get("decision") == "EMPTY_SLOT"
        )
        identified_count = sum(
            1
            for row in reconciliation_rows
            if row.get("final_hero_uid")
        )

        if remaining_review != 0:
            raise RuntimeError(
                f"{remaining_review} cas restent en revue."
            )

        if empty_count != 1:
            raise RuntimeError(
                f"1 emplacement vide attendu, trouvé : {empty_count}."
            )

        if identified_count != 999:
            raise RuntimeError(
                f"999 héros identifiés attendus, trouvé : {identified_count}."
            )

        manual_fields, manual_rows = build_manual_decisions(
            hero_uid,
            appearance_id,
        )
        slot_fields, slot_rows = build_slot_manifest(
            reconciliation_rows
        )
        summary = build_evaluation_summary(
            reconciliation_rows,
            hero_uid,
        )

    except (RuntimeError, OSError, csv.Error, ValueError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    print("FINALISATION — HERO_BATCH_003")
    print("=" * 72)
    print(f"Héros Elmir : {hero_uid}")
    print(f"Apparence : {appearance_id}")
    print(f"Héros ajouté : {'oui' if hero_added else 'non, déjà présent'}")
    print(
        "Apparence ajoutée : "
        f"{'oui' if appearance_added else 'non, déjà présente'}"
    )
    print(f"Alias ajouté : {'oui' if alias_added else 'non, déjà présent'}")
    print(
        "Référence avatar ajoutée : "
        f"{'oui' if avatar_added else 'non, déjà présente'}"
    )
    print()
    print(f"Emplacements analysés : {len(reconciliation_rows)}")
    print(f"Héros identifiés : {identified_count}")
    print(f"Emplacements vides : {empty_count}")
    print(f"Cas restant à revoir : {remaining_review}")
    print()

    if not args.apply:
        print("MODE SIMULATION : aucun fichier n'a été modifié.")
        print()
        print("Pour appliquer :")
        print(
            "python scripts/finalize_hero_batch_003_v2.py --apply"
        )
        return 0

    backup_dir = backup_files(
        [
            HEROES_CSV,
            APPEARANCES_CSV,
            ALIASES_CSV,
            AVATAR_MANIFEST_CSV,
            RECONCILIATION_CSV,
            MANUAL_DECISIONS_CSV,
            SLOT_IDENTITY_MANIFEST_CSV,
            EVALUATION_SUMMARY,
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

    CANONICAL_AVATAR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DESTINATION_AVATAR.exists():
        shutil.copy2(
            SOURCE_AVATAR,
            DESTINATION_AVATAR,
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

    EVALUATION_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    EVALUATION_SUMMARY.write_text(
        summary,
        encoding="utf-8",
    )

    print("Finalisation appliquée.")
    print(f"Sauvegarde : {backup_dir}")
    print(f"Avatar copié : {DESTINATION_AVATAR}")
    print(f"Manifeste final : {SLOT_IDENTITY_MANIFEST_CSV}")
    print(f"Décisions manuelles : {MANUAL_DECISIONS_CSV}")
    print(f"Résumé : {EVALUATION_SUMMARY}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
