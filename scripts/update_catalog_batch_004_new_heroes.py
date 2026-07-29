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


BATCH_NAME = "hero_batch_004"

CATALOG_DIR = Path("data/catalog")
HEROES_CSV = CATALOG_DIR / "heroes.csv"
APPEARANCES_CSV = CATALOG_DIR / "hero_appearances.csv"
ALIASES_CSV = CATALOG_DIR / "hero_name_aliases.csv"
AVATAR_MANIFEST_CSV = CATALOG_DIR / "hero_avatar_manifest.csv"

BATCH_ROOT = Path("data/batches") / BATCH_NAME
SOURCE_AVATAR_DIR = BATCH_ROOT / "crops_dynamic_v1" / "avatars_inner"
CANONICAL_AVATAR_DIR = Path("data/crops/hero/avatars_inner")

NEW_HEROES = (
    {
        "reference_name": "Markus",
        "aliases": (
            ("Markus", "en", 1),
            ("Marcus", "fr", 3),
        ),
        "samples": (
            ("6628", "R", "4", "6628_Screenshot_20250818_185646_Hero_Wars__R4.png"),
            ("29699", "R", "4", "29699_IMG_2291__R4.png"),
            ("33354", "R", "4", "33354_Screenshot_20260512_191313_com_nexters_herowars_MessagingUnityPlayerActivity__R4.png"),
            ("37688", "R", "3", "37688_Screenshot_2026-07-02-07-57-52-488_com.nexters.herowars__R3.png"),
        ),
    },
    {
        "reference_name": "Ninja Turtles",
        "aliases": (
            ("Ninja Turtles", "en", 1),
            ("Tortues Ninja", "fr", 1),
        ),
        "samples": (
            ("24011", "R", "1", "24011_Screenshot_20260205_224503_Hero_Wars__R1.png"),
            ("27428", "L", "1", "27428_Screenshot_20260303_182757_Hero_Wars__L1.png"),
        ),
    },
    {
        "reference_name": "Fox",
        "aliases": (
            ("Fox", "en", 1),
        ),
        "samples": (
            ("25511", "R", "4", "25511_E582EEDB-B61B-42D0-B3BD-9D18EF70401F__R4.png"),
        ),
    },
)

EXISTING_ALIAS_ADDITIONS = (
    {
        "expected_hero_uid": "HW_HERO_0075",
        "expected_reference_name": "Trompelamort",
        "alias": "Daredevil",
        "language": "en",
        "occurrences": 1,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ajoute au catalogue les héros découverts dans hero_batch_004 "
            "ainsi que l'alias Daredevil."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Applique réellement les modifications. Sans cette option, "
            "le script effectue uniquement une simulation."
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

        characters.append(
            character
            if character.isalnum()
            else "_"
        )

    return re.sub(r"_+", "_", "".join(characters)).strip("_") or "hero"


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
        raise RuntimeError(f"En-tête absent : {path}")

    return fields, rows


def write_csv_atomic(
    path: Path,
    fields: list[str],
    rows: list[dict[str, Any]],
) -> None:
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
    reference_name: str,
) -> str:
    existing = {
        row.get("appearance_id", "")
        for row in appearances
    }
    slug = slugify(reference_name)
    index = 1

    while True:
        candidate = f"{slug}__appearance_{index:02d}"

        if candidate not in existing:
            return candidate

        index += 1


def ensure_hero(
    heroes: list[dict[str, str]],
    reference_name: str,
) -> tuple[str, bool]:
    matches = [
        row
        for row in heroes
        if normalize_key(row.get("reference_name", ""))
        == normalize_key(reference_name)
    ]

    if len(matches) > 1:
        raise RuntimeError(
            f"Plusieurs héros correspondent à {reference_name!r}."
        )

    if matches:
        return matches[0]["hero_uid"], False

    hero_uid = next_hero_uid(heroes)

    heroes.append(
        {
            "hero_uid": hero_uid,
            "reference_name": reference_name,
            "provisional_key": slugify(reference_name),
            "reviewed": "1",
            "notes": (
                "Ajouté après revue humaine de hero_batch_004."
            ),
        }
    )

    return hero_uid, True


def ensure_appearance(
    appearances: list[dict[str, str]],
    hero_uid: str,
    reference_name: str,
    sample_names: list[str],
) -> tuple[str, bool]:
    normalized_samples = set(sample_names)

    for row in appearances:
        if row.get("hero_uid") != hero_uid:
            continue

        existing_samples = {
            value.strip()
            for value in row.get("sample_files", "").split("|")
            if value.strip()
        }

        if existing_samples.intersection(normalized_samples):
            merged = sorted(existing_samples.union(normalized_samples))
            row["sample_files"] = "|".join(merged)
            row["avatar_count"] = str(len(merged))
            row["reviewed"] = "1"
            return row["appearance_id"], False

    appearance_id = unique_appearance_id(
        appearances,
        reference_name,
    )

    appearances.append(
        {
            "appearance_id": appearance_id,
            "hero_uid": hero_uid,
            "technical_cluster_id": "",
            "avatar_count": str(len(sample_names)),
            "sample_files": "|".join(sample_names),
            "appearance_type": "unknown_variant",
            "reviewed": "1",
            "notes": (
                "Première apparence validée depuis hero_batch_004."
            ),
        }
    )

    return appearance_id, True


def ensure_alias(
    aliases: list[dict[str, str]],
    hero_uid: str,
    alias: str,
    language: str,
    occurrences: int,
    source: str,
) -> bool:
    alias_key = normalize_key(alias)

    same_alias = [
        row
        for row in aliases
        if normalize_key(row.get("alias", "")) == alias_key
    ]

    conflicts = [
        row
        for row in same_alias
        if row.get("hero_uid") != hero_uid
    ]

    if conflicts:
        raise RuntimeError(
            f"L'alias {alias!r} appartient déjà à un autre héros."
        )

    if same_alias:
        row = same_alias[0]
        row["reviewed"] = "1"
        row["occurrences"] = str(
            max(
                int(row.get("occurrences") or 0),
                occurrences,
            )
        )

        if not row.get("language"):
            row["language"] = language
        if not row.get("source"):
            row["source"] = source

        return False

    aliases.append(
        {
            "hero_uid": hero_uid,
            "alias": alias,
            "language": language,
            "source": source,
            "occurrences": str(occurrences),
            "reviewed": "1",
        }
    )

    return True


def ensure_avatar_manifest_entry(
    rows: list[dict[str, str]],
    fields: list[str],
    hero_uid: str,
    appearance_id: str,
    screenshot_id: str,
    side: str,
    slot: str,
    avatar_filename: str,
    reference_name: str,
) -> bool:
    required = {
        "avatar_file",
        "hero_uid",
        "appearance_id",
        "screenshot_id",
        "side",
        "slot",
    }

    missing = required.difference(fields)

    if missing:
        raise RuntimeError(
            "Colonnes absentes du manifeste maître : "
            + ", ".join(sorted(missing))
        )

    matches = [
        row
        for row in rows
        if row.get("avatar_file") == avatar_filename
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
        for field in fields
    }

    values = {
        "avatar_file": avatar_filename,
        "hero_uid": hero_uid,
        "appearance_id": appearance_id,
        "ocr_text": reference_name,
        "ocr_confidence": "1.0",
        "ocr_status": "HIGH",
        "screenshot_id": screenshot_id,
        "side": side,
        "slot": slot,
        "label_source": "manual_review_batch_004",
        "reviewed": "1",
    }

    for key, value in values.items():
        if key in new_row:
            new_row[key] = value

    rows.append(new_row)
    return True


def find_existing_hero_uid(
    heroes: list[dict[str, str]],
    expected_uid: str,
    expected_name: str,
) -> str:
    by_uid = [
        row
        for row in heroes
        if row.get("hero_uid") == expected_uid
    ]

    if len(by_uid) != 1:
        raise RuntimeError(
            f"Héros attendu absent : {expected_uid}."
        )

    actual_name = by_uid[0].get("reference_name", "")

    if normalize_key(actual_name) != normalize_key(expected_name):
        raise RuntimeError(
            f"{expected_uid} est nommé {actual_name!r}, "
            f"attendu {expected_name!r}."
        )

    return expected_uid


def backup_files(
    paths: list[Path],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = (
        CATALOG_DIR
        / "backups"
        / f"batch_004_new_heroes_{timestamp}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)

    for path in paths:
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)

    return backup_dir


def main() -> int:
    args = parse_args()

    try:
        hero_fields, heroes = read_csv(HEROES_CSV)
        appearance_fields, appearances = read_csv(APPEARANCES_CSV)
        alias_fields, aliases = read_csv(ALIASES_CSV)
        avatar_fields, avatar_rows = read_csv(AVATAR_MANIFEST_CSV)

        operations: list[dict[str, Any]] = []

        for specification in NEW_HEROES:
            reference_name = str(specification["reference_name"])
            samples = list(specification["samples"])

            source_paths = [
                SOURCE_AVATAR_DIR / sample[3]
                for sample in samples
            ]

            missing_sources = [
                str(path)
                for path in source_paths
                if not path.exists()
            ]

            if missing_sources:
                raise RuntimeError(
                    "Avatars sources absents :\n"
                    + "\n".join(
                        "- " + path
                        for path in missing_sources
                    )
                )

            hero_uid, hero_added = ensure_hero(
                heroes,
                reference_name,
            )

            appearance_id, appearance_added = ensure_appearance(
                appearances,
                hero_uid,
                reference_name,
                [sample[3] for sample in samples],
            )

            alias_results = []

            for alias, language, occurrences in specification["aliases"]:
                alias_results.append(
                    ensure_alias(
                        aliases,
                        hero_uid,
                        str(alias),
                        str(language),
                        int(occurrences),
                        "manual_review_batch_004",
                    )
                )

            avatar_additions = 0

            for screenshot_id, side, slot, avatar_filename in samples:
                avatar_additions += int(
                    ensure_avatar_manifest_entry(
                        avatar_rows,
                        avatar_fields,
                        hero_uid,
                        appearance_id,
                        str(screenshot_id),
                        str(side),
                        str(slot),
                        str(avatar_filename),
                        reference_name,
                    )
                )

            operations.append(
                {
                    "reference_name": reference_name,
                    "hero_uid": hero_uid,
                    "appearance_id": appearance_id,
                    "hero_added": hero_added,
                    "appearance_added": appearance_added,
                    "aliases_added": sum(alias_results),
                    "avatars_added": avatar_additions,
                    "source_paths": source_paths,
                }
            )

        existing_alias_results = []

        for addition in EXISTING_ALIAS_ADDITIONS:
            hero_uid = find_existing_hero_uid(
                heroes,
                str(addition["expected_hero_uid"]),
                str(addition["expected_reference_name"]),
            )

            alias_added = ensure_alias(
                aliases,
                hero_uid,
                str(addition["alias"]),
                str(addition["language"]),
                int(addition["occurrences"]),
                "manual_review_batch_004",
            )

            existing_alias_results.append(
                {
                    "hero_uid": hero_uid,
                    "alias": str(addition["alias"]),
                    "added": alias_added,
                }
            )

    except (RuntimeError, OSError, csv.Error, ValueError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    print("MISE À JOUR DU CATALOGUE — HERO_BATCH_004")
    print("=" * 72)
    print()

    for operation in operations:
        print(
            f"{operation['reference_name']} : "
            f"{operation['hero_uid']}"
        )
        print(f"- Apparence : {operation['appearance_id']}")
        print(
            "- Héros ajouté : "
            f"{'oui' if operation['hero_added'] else 'non, déjà présent'}"
        )
        print(
            "- Apparence ajoutée : "
            f"{'oui' if operation['appearance_added'] else 'non, déjà présente'}"
        )
        print(f"- Alias ajoutés : {operation['aliases_added']}")
        print(f"- Références avatars ajoutées : {operation['avatars_added']}")
        print()

    for result in existing_alias_results:
        print(
            f"Alias {result['alias']} -> {result['hero_uid']} : "
            f"{'ajouté' if result['added'] else 'déjà présent'}"
        )

    print()
    print(f"Héros après simulation : {len(heroes)}")
    print(f"Apparences après simulation : {len(appearances)}")
    print(f"Alias après simulation : {len(aliases)}")
    print(f"Références avatars après simulation : {len(avatar_rows)}")
    print()

    if not args.apply:
        print("MODE SIMULATION : aucun fichier n'a été modifié.")
        print()
        print("Pour appliquer :")
        print(
            "python scripts/update_catalog_batch_004_new_heroes.py --apply"
        )
        return 0

    backup_dir = backup_files(
        [
            HEROES_CSV,
            APPEARANCES_CSV,
            ALIASES_CSV,
            AVATAR_MANIFEST_CSV,
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

    CANONICAL_AVATAR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = 0

    for operation in operations:
        for source_path in operation["source_paths"]:
            destination = CANONICAL_AVATAR_DIR / source_path.name

            if not destination.exists():
                shutil.copy2(source_path, destination)
                copied += 1

    print("Mise à jour appliquée.")
    print(f"Sauvegarde : {backup_dir}")
    print(f"Avatars copiés : {copied}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
