from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


BATCH_ROOT = Path("data/batches/hero_batch_001")

HEROES_CSV = Path("data/catalog/heroes.csv")
APPEARANCES_CSV = Path("data/catalog/hero_appearances.csv")
ALIASES_CSV = Path("data/catalog/hero_name_aliases.csv")
MASTER_AVATAR_MANIFEST_CSV = Path(
    "data/catalog/hero_avatar_manifest.csv"
)

RECONCILIATION_CSV = (
    BATCH_ROOT
    / "reports"
    / "reconciliation_v1"
    / "reconciliation_results.csv"
)

REVIEW_GROUPS_CSV = (
    BATCH_ROOT
    / "reports"
    / "reconciliation_groups_v1"
    / "review_groups.csv"
)

REVIEW_MEMBERS_CSV = (
    BATCH_ROOT
    / "reports"
    / "reconciliation_groups_v1"
    / "review_group_members.csv"
)

CROP_MANIFEST_CSV = (
    BATCH_ROOT
    / "crops_dynamic_v1"
    / "crop_manifest.csv"
)

DEFAULT_DECISIONS_CSV = Path(
    "data/curation/hero_batch_001_review_decisions.csv"
)

VALIDATED_DIR = BATCH_ROOT / "validated"
IDENTITY_MANIFEST_CSV = (
    VALIDATED_DIR
    / "hero_identity_manifest.csv"
)

PREVIEW_DIR = (
    BATCH_ROOT
    / "reports"
    / "catalog_update_preview"
)

APPLIED_REPORT_DIR = (
    BATCH_ROOT
    / "reports"
    / "catalog_update_applied"
)

BACKUP_ROOT = Path("data/catalog/backups")

EXPECTED_GROUP_COUNT = 44
EXPECTED_BATCH_ROWS = 1000
EXPECTED_REVIEW_ROWS = 108


REQUIRED_SCHEMAS = {
    HEROES_CSV: {
        "hero_uid",
        "reference_name",
        "provisional_key",
        "reviewed",
        "notes",
    },
    APPEARANCES_CSV: {
        "appearance_id",
        "hero_uid",
        "technical_cluster_id",
        "avatar_count",
        "sample_files",
        "appearance_type",
        "reviewed",
        "notes",
    },
    ALIASES_CSV: {
        "hero_uid",
        "alias",
        "language",
        "source",
        "occurrences",
        "reviewed",
    },
    MASTER_AVATAR_MANIFEST_CSV: {
        "avatar_file",
        "hero_uid",
        "appearance_id",
        "ocr_text",
        "ocr_confidence",
        "ocr_status",
        "screenshot_id",
        "side",
        "slot",
        "label_source",
        "reviewed",
    },
    RECONCILIATION_CSV: {
        "screenshot_id",
        "side",
        "slot",
        "avatar_file",
        "name_file",
        "visual_hero_uid",
        "visual_hero_name",
        "visual_status",
        "visual_similarity",
        "visual_margin",
        "ocr_text",
        "ocr_confidence",
        "ocr_status",
        "decision",
        "final_hero_uid",
        "final_hero_name",
        "review_required",
    },
    REVIEW_GROUPS_CSV: {
        "group_id",
        "decision",
        "member_count",
        "representative_ocr_text",
    },
    REVIEW_MEMBERS_CSV: {
        "group_id",
        "screenshot_id",
        "side",
        "slot",
        "avatar_file",
        "name_file",
        "ocr_text",
        "ocr_confidence",
        "ocr_status",
    },
    CROP_MANIFEST_CSV: {
        "side",
        "slot",
        "avatar_inner_file",
        "name_file",
    },
}


DECISION_FIELDS = {
    "group_id",
    "target_reference_name",
    "target_mode",
    "existing_hero_uid",
    "appearance_id",
    "corrected_ocr_text",
    "alias_to_add",
    "alias_language",
    "notes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prépare ou applique les décisions humaines du lot "
            "hero_batch_001 sans écraser les catalogues sans sauvegarde."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Applique réellement les modifications. "
            "Sans cette option, seul un aperçu est produit."
        ),
    )

    parser.add_argument(
        "--decisions",
        type=Path,
        default=DEFAULT_DECISIONS_CSV,
        help="Chemin du CSV des 44 décisions validées.",
    )

    return parser.parse_args()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value or "")
    value = value.replace("’", "'")
    value = value.replace("`", "'")
    value = " ".join(value.split())
    return value.strip(" -")


def normalize_key(value: str) -> str:
    value = unicodedata.normalize(
        "NFKD",
        normalize_text(value),
    )

    characters: list[str] = []

    for character in value.casefold():
        if unicodedata.combining(character):
            continue

        if character.isalnum():
            characters.append(character)
        else:
            characters.append(" ")

    return " ".join("".join(characters).split())


def slugify(value: str) -> str:
    value = normalize_key(value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "hero"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise RuntimeError(f"Fichier absent : {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file,
            delimiter=";",
        )

        fieldnames = list(reader.fieldnames or [])
        rows = [
            {
                key: (value or "")
                for key, value in row.items()
            }
            for row in reader
        ]

    return fieldnames, rows


def validate_schema(
    path: Path,
    fieldnames: list[str],
    required: set[str],
) -> None:
    missing = sorted(required - set(fieldnames))

    if missing:
        raise RuntimeError(
            f"Colonnes absentes dans {path} : "
            + ", ".join(missing)
        )


def write_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as temp_file:
        writer = csv.DictWriter(
            temp_file,
            delimiter=";",
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fieldnames
                }
            )

        temp_path = Path(temp_file.name)

    os.replace(temp_path, path)


def next_hero_uid(
    used_uids: set[str],
) -> str:
    numbers = []

    for uid in used_uids:
        match = re.fullmatch(
            r"HW_HERO_(\d+)",
            uid,
        )

        if match:
            numbers.append(int(match.group(1)))

    next_number = (
        max(numbers) + 1
        if numbers
        else 1
    )

    while True:
        candidate = f"HW_HERO_{next_number:04d}"

        if candidate not in used_uids:
            return candidate

        next_number += 1


def make_slot_key(
    screenshot_id: str,
    side: str,
    slot: str,
) -> tuple[str, str, str]:
    return (
        str(screenshot_id),
        str(side),
        str(slot),
    )


def safe_int(
    value: str,
    default: int = 0,
) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def merge_alias(
    aliases: list[dict[str, str]],
    hero_uid: str,
    alias: str,
    language: str,
    occurrences: int,
    source: str,
    reviewed: str,
) -> None:
    alias = normalize_text(alias)
    alias_key = normalize_key(alias)

    if not alias_key:
        return

    for row in aliases:
        if (
            row["hero_uid"] == hero_uid
            and normalize_key(row["alias"]) == alias_key
        ):
            row["occurrences"] = str(
                max(
                    safe_int(row.get("occurrences", "0")),
                    occurrences,
                )
            )

            if source == "human_reviewed_batch_001":
                row["source"] = source
                row["reviewed"] = reviewed

            if language and row.get("language", "") in {
                "",
                "unknown",
            }:
                row["language"] = language

            return

    aliases.append(
        {
            "hero_uid": hero_uid,
            "alias": alias,
            "language": language or "unknown",
            "source": source,
            "occurrences": str(max(occurrences, 1)),
            "reviewed": reviewed,
        }
    )


def build_backup(
    files: list[Path],
) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_dir = (
        BACKUP_ROOT
        / f"hero_batch_001_{timestamp}"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    for source in files:
        if not source.exists():
            continue

        relative = source.as_posix().replace("/", "__")
        destination = backup_dir / relative
        shutil.copy2(source, destination)

    return backup_dir


def main() -> int:
    args = parse_args()

    try:
        loaded: dict[
            Path,
            tuple[list[str], list[dict[str, str]]],
        ] = {}

        for path, required in REQUIRED_SCHEMAS.items():
            fieldnames, rows = read_csv(path)
            validate_schema(
                path,
                fieldnames,
                required,
            )
            loaded[path] = (
                fieldnames,
                rows,
            )

        decision_fields, decision_rows = read_csv(
            args.decisions
        )

        validate_schema(
            args.decisions,
            decision_fields,
            DECISION_FIELDS,
        )

    except RuntimeError as error:
        print(
            f"[ERREUR] {error}",
            file=sys.stderr,
        )
        return 1

    heroes_fields, heroes = loaded[HEROES_CSV]
    appearance_fields, appearances = loaded[
        APPEARANCES_CSV
    ]
    alias_fields, aliases = loaded[ALIASES_CSV]
    master_manifest_fields, master_manifest = loaded[
        MASTER_AVATAR_MANIFEST_CSV
    ]
    reconciliation_fields, reconciliation = loaded[
        RECONCILIATION_CSV
    ]
    _, review_groups = loaded[REVIEW_GROUPS_CSV]
    _, review_members = loaded[REVIEW_MEMBERS_CSV]
    _, crop_manifest = loaded[CROP_MANIFEST_CSV]

    if len(reconciliation) != EXPECTED_BATCH_ROWS:
        print(
            f"[ERREUR] Le lot contient {len(reconciliation)} lignes "
            f"au lieu de {EXPECTED_BATCH_ROWS}.",
            file=sys.stderr,
        )
        return 1

    if len(review_members) != EXPECTED_REVIEW_ROWS:
        print(
            f"[ERREUR] La revue contient {len(review_members)} lignes "
            f"au lieu de {EXPECTED_REVIEW_ROWS}.",
            file=sys.stderr,
        )
        return 1

    decision_by_group = {
        row["group_id"]: row
        for row in decision_rows
    }

    if len(decision_by_group) != len(decision_rows):
        print(
            "[ERREUR] Le CSV de décisions contient "
            "des group_id en double.",
            file=sys.stderr,
        )
        return 1

    group_ids = {
        row["group_id"]
        for row in review_groups
    }

    decision_group_ids = set(decision_by_group)

    if len(group_ids) != EXPECTED_GROUP_COUNT:
        print(
            f"[ERREUR] {len(group_ids)} groupes trouvés "
            f"au lieu de {EXPECTED_GROUP_COUNT}.",
            file=sys.stderr,
        )
        return 1

    if group_ids != decision_group_ids:
        missing = sorted(group_ids - decision_group_ids)
        extra = sorted(decision_group_ids - group_ids)

        print(
            "[ERREUR] Les décisions ne couvrent pas "
            "exactement les groupes de revue.",
            file=sys.stderr,
        )

        if missing:
            print(
                "  Groupes manquants : "
                + ", ".join(missing),
                file=sys.stderr,
            )

        if extra:
            print(
                "  Groupes inconnus : "
                + ", ".join(extra),
                file=sys.stderr,
            )

        return 1

    members_by_group: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    member_group_by_slot: dict[
        tuple[str, str, str],
        str,
    ] = {}

    for row in review_members:
        group_id = row["group_id"]
        members_by_group[group_id].append(row)

        slot_key = make_slot_key(
            row["screenshot_id"],
            row["side"],
            row["slot"],
        )

        if slot_key in member_group_by_slot:
            print(
                f"[ERREUR] Emplacement de revue en double : "
                f"{slot_key}",
                file=sys.stderr,
            )
            return 1

        member_group_by_slot[slot_key] = group_id

    group_counts = {
        row["group_id"]: safe_int(row["member_count"])
        for row in review_groups
    }

    for group_id, expected_count in group_counts.items():
        actual_count = len(
            members_by_group[group_id]
        )

        if actual_count != expected_count:
            print(
                f"[ERREUR] {group_id} contient {actual_count} membres "
                f"au lieu de {expected_count}.",
                file=sys.stderr,
            )
            return 1

    heroes_by_uid = {
        row["hero_uid"]: row
        for row in heroes
    }

    heroes_by_name = {
        normalize_key(row["reference_name"]): row
        for row in heroes
    }

    used_uids = set(heroes_by_uid)
    used_provisional_keys = {
        row["provisional_key"]
        for row in heroes
    }

    target_uid_by_name: dict[str, str] = {}
    created_heroes: list[dict[str, str]] = []

    # Résout d'abord les cibles existantes.
    for decision in decision_rows:
        target_name = normalize_text(
            decision["target_reference_name"]
        )

        target_key = normalize_key(target_name)
        mode = decision["target_mode"].strip().upper()

        if mode == "EXISTING":
            uid = decision["existing_hero_uid"].strip()

            if uid not in heroes_by_uid:
                print(
                    f"[ERREUR] {decision['group_id']} référence "
                    f"un UID absent : {uid}",
                    file=sys.stderr,
                )
                return 1

            existing_name = heroes_by_uid[uid][
                "reference_name"
            ]

            if normalize_key(existing_name) != target_key:
                print(
                    f"[ERREUR] {decision['group_id']} : "
                    f"{uid} correspond à {existing_name}, "
                    f"pas à {target_name}.",
                    file=sys.stderr,
                )
                return 1

            target_uid_by_name[target_key] = uid

        elif mode == "CREATE":
            # Idempotence : si le héros a déjà été créé lors
            # d'une exécution précédente, on réutilise son UID.
            existing = heroes_by_name.get(target_key)

            if existing is not None:
                uid = existing["hero_uid"]
                target_uid_by_name[target_key] = uid
                continue

            if target_key in target_uid_by_name:
                continue

            uid = next_hero_uid(used_uids)
            used_uids.add(uid)

            provisional_key = slugify(target_name)
            base_key = provisional_key
            suffix = 2

            while provisional_key in used_provisional_keys:
                provisional_key = (
                    f"{base_key}_{suffix}"
                )
                suffix += 1

            used_provisional_keys.add(
                provisional_key
            )

            new_row = {
                "hero_uid": uid,
                "reference_name": target_name,
                "provisional_key": provisional_key,
                "reviewed": "1",
                "notes": (
                    "Créé après validation humaine du lot "
                    "hero_batch_001."
                ),
            }

            heroes.append(new_row)
            heroes_by_uid[uid] = new_row
            heroes_by_name[target_key] = new_row
            target_uid_by_name[target_key] = uid
            created_heroes.append(new_row)

        else:
            print(
                f"[ERREUR] Mode inconnu pour "
                f"{decision['group_id']} : {mode}",
                file=sys.stderr,
            )
            return 1

    # Résolution finale de chaque groupe.
    resolved_decisions: dict[
        str,
        dict[str, str],
    ] = {}

    for decision in decision_rows:
        target_name = normalize_text(
            decision["target_reference_name"]
        )

        target_key = normalize_key(target_name)
        uid = target_uid_by_name[target_key]

        resolved = dict(decision)
        resolved["resolved_hero_uid"] = uid
        resolved["target_reference_name"] = target_name
        resolved["corrected_ocr_text"] = (
            normalize_text(
                decision["corrected_ocr_text"]
            )
            or target_name
        )
        resolved["alias_to_add"] = normalize_text(
            decision["alias_to_add"]
        )

        resolved_decisions[
            decision["group_id"]
        ] = resolved

    # Met à jour la réconciliation.
    reconciliation_by_slot = {
        make_slot_key(
            row["screenshot_id"],
            row["side"],
            row["slot"],
        ): row
        for row in reconciliation
    }

    extra_reconciliation_fields = [
        "review_group_id",
        "human_review_action",
        "validated_appearance_id",
        "corrected_ocr_text",
        "human_review_notes",
    ]

    for field in extra_reconciliation_fields:
        if field not in reconciliation_fields:
            reconciliation_fields.append(field)

    for row in reconciliation:
        for field in extra_reconciliation_fields:
            row.setdefault(field, "")

    for group_id, member_rows in members_by_group.items():
        decision = resolved_decisions[group_id]
        uid = decision["resolved_hero_uid"]
        target_name = decision[
            "target_reference_name"
        ]

        for member in member_rows:
            slot_key = make_slot_key(
                member["screenshot_id"],
                member["side"],
                member["slot"],
            )

            row = reconciliation_by_slot.get(
                slot_key
            )

            if row is None:
                print(
                    f"[ERREUR] Ligne de réconciliation absente : "
                    f"{slot_key}",
                    file=sys.stderr,
                )
                return 1

            row["decision"] = "HUMAN_REVIEWED"
            row["final_hero_uid"] = uid
            row["final_hero_name"] = target_name
            row["review_required"] = "0"
            row["review_group_id"] = group_id
            row["human_review_action"] = (
                "ASSIGN_EXISTING_HERO"
                if decision["target_mode"].upper()
                == "EXISTING"
                else "CREATE_OR_ASSIGN_HERO"
            )
            row["validated_appearance_id"] = (
                decision["appearance_id"]
            )
            row["corrected_ocr_text"] = (
                decision["corrected_ocr_text"]
            )
            row["human_review_notes"] = (
                decision["notes"]
            )

    unresolved_rows = [
        row
        for row in reconciliation
        if not row["final_hero_uid"].strip()
    ]

    if unresolved_rows:
        print(
            f"[ERREUR] {len(unresolved_rows)} lignes restent "
            "sans identité finale.",
            file=sys.stderr,
        )
        return 1

    # Alias humains : référence des nouveaux héros +
    # alias explicitement validés.
    new_hero_group_counts: Counter[
        str
    ] = Counter()

    alias_contributions: Counter[
        tuple[str, str, str]
    ] = Counter()

    for group_id, member_rows in members_by_group.items():
        decision = resolved_decisions[group_id]
        uid = decision["resolved_hero_uid"]
        target_name = decision[
            "target_reference_name"
        ]
        target_mode = decision[
            "target_mode"
        ].upper()

        count = len(member_rows)

        corrected = decision[
            "corrected_ocr_text"
        ]

        if (
            target_mode == "CREATE"
            and normalize_key(corrected)
            == normalize_key(target_name)
        ):
            new_hero_group_counts[uid] += count

        alias_to_add = decision[
            "alias_to_add"
        ]

        if alias_to_add:
            alias_contributions[
                (
                    uid,
                    alias_to_add,
                    decision["alias_language"]
                    or "unknown",
                )
            ] += count

    for hero in created_heroes:
        uid = hero["hero_uid"]
        merge_alias(
            aliases=aliases,
            hero_uid=uid,
            alias=hero["reference_name"],
            language="unknown",
            occurrences=max(
                new_hero_group_counts[uid],
                1,
            ),
            source="human_reviewed_batch_001",
            reviewed="1",
        )

    for (
        uid,
        alias,
        language,
    ), occurrences in alias_contributions.items():
        merge_alias(
            aliases=aliases,
            hero_uid=uid,
            alias=alias,
            language=language,
            occurrences=occurrences,
            source="human_reviewed_batch_001",
            reviewed="1",
        )

    # Les alias exacts des héros existants observés dans
    # les groupes humains deviennent eux aussi revus.
    for group_id, member_rows in members_by_group.items():
        decision = resolved_decisions[group_id]
        uid = decision["resolved_hero_uid"]
        target_name = decision[
            "target_reference_name"
        ]

        corrected = decision[
            "corrected_ocr_text"
        ]

        if normalize_key(corrected) == normalize_key(
            target_name
        ):
            merge_alias(
                aliases=aliases,
                hero_uid=uid,
                alias=target_name,
                language="unknown",
                occurrences=len(member_rows),
                source="human_reviewed_batch_001",
                reviewed="1",
            )

    # Ajoute les 108 avatars validés au manifeste maître.
    existing_avatar_files = {
        row["avatar_file"]
        for row in master_manifest
    }

    affected_appearance_ids: set[str] = set()
    added_master_avatar_rows = 0

    for group_id, member_rows in members_by_group.items():
        decision = resolved_decisions[group_id]
        uid = decision["resolved_hero_uid"]
        appearance_id = decision[
            "appearance_id"
        ]

        affected_appearance_ids.add(
            appearance_id
        )

        for member in member_rows:
            avatar_file = member["avatar_file"]

            if avatar_file in existing_avatar_files:
                continue

            master_manifest.append(
                {
                    "avatar_file": avatar_file,
                    "hero_uid": uid,
                    "appearance_id": appearance_id,
                    "ocr_text": decision[
                        "corrected_ocr_text"
                    ],
                    "ocr_confidence": member[
                        "ocr_confidence"
                    ],
                    "ocr_status": member[
                        "ocr_status"
                    ],
                    "screenshot_id": member[
                        "screenshot_id"
                    ],
                    "side": member["side"],
                    "slot": member["slot"],
                    "label_source": (
                        "human_reviewed_batch_001"
                    ),
                    "reviewed": "1",
                }
            )

            existing_avatar_files.add(
                avatar_file
            )
            added_master_avatar_rows += 1

    # Crée ou vérifie les apparences visuelles.
    appearance_by_id = {
        row["appearance_id"]: row
        for row in appearances
    }

    groups_by_appearance: dict[
        str,
        list[str],
    ] = defaultdict(list)

    uid_by_appearance: dict[
        str,
        str,
    ] = {}

    notes_by_appearance: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for group_id, decision in resolved_decisions.items():
        appearance_id = decision[
            "appearance_id"
        ]
        uid = decision[
            "resolved_hero_uid"
        ]

        existing_uid = uid_by_appearance.get(
            appearance_id
        )

        if existing_uid and existing_uid != uid:
            print(
                f"[ERREUR] L'apparence {appearance_id} "
                "est rattachée à deux héros différents.",
                file=sys.stderr,
            )
            return 1

        uid_by_appearance[appearance_id] = uid
        groups_by_appearance[
            appearance_id
        ].append(group_id)
        notes_by_appearance[
            appearance_id
        ].append(decision["notes"])

    for appearance_id, uid in uid_by_appearance.items():
        existing = appearance_by_id.get(
            appearance_id
        )

        if existing is not None:
            if existing["hero_uid"] != uid:
                print(
                    f"[ERREUR] {appearance_id} appartient déjà "
                    f"à {existing['hero_uid']} et non à {uid}.",
                    file=sys.stderr,
                )
                return 1

            # Une apparence existante confirmée humainement.
            existing["reviewed"] = "1"

        else:
            group_list = sorted(
                groups_by_appearance[
                    appearance_id
                ]
            )

            row = {
                "appearance_id": appearance_id,
                "hero_uid": uid,
                "technical_cluster_id": (
                    "batch001:"
                    + ",".join(group_list)
                ),
                "avatar_count": "0",
                "sample_files": "",
                "appearance_type": (
                    "human_reviewed_variant"
                ),
                "reviewed": "1",
                "notes": (
                    "Créée après validation humaine du lot "
                    "hero_batch_001. "
                    + " | ".join(
                        sorted(
                            set(
                                note
                                for note in notes_by_appearance[
                                    appearance_id
                                ]
                                if note
                            )
                        )
                    )
                ).strip(),
            }

            appearances.append(row)
            appearance_by_id[
                appearance_id
            ] = row

    # Recalcule les comptes et exemples uniquement pour
    # les apparences touchées.
    manifest_by_appearance: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in master_manifest:
        manifest_by_appearance[
            row["appearance_id"]
        ].append(row)

    for appearance_id in affected_appearance_ids:
        row = appearance_by_id[
            appearance_id
        ]

        avatar_rows = sorted(
            manifest_by_appearance[
                appearance_id
            ],
            key=lambda item: item[
                "avatar_file"
            ],
        )

        row["avatar_count"] = str(
            len(avatar_rows)
        )

        row["sample_files"] = " | ".join(
            item["avatar_file"]
            for item in avatar_rows[:5]
        )

    # Manifeste d'identité pour les 1000 avatars.
    appearance_ids_by_hero: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for row in appearances:
        appearance_ids_by_hero[
            row["hero_uid"]
        ].append(
            row["appearance_id"]
        )

    identity_fields = [
        "screenshot_id",
        "side",
        "slot",
        "avatar_file",
        "name_file",
        "hero_uid",
        "reference_name",
        "appearance_id",
        "ocr_text",
        "ocr_confidence",
        "ocr_status",
        "source_decision",
        "label_source",
        "reviewed",
        "review_group_id",
    ]

    identity_rows: list[
        dict[str, object]
    ] = []

    for row in reconciliation:
        uid = row["final_hero_uid"]
        appearance_id = row.get(
            "validated_appearance_id",
            "",
        )

        review_group_id = row.get(
            "review_group_id",
            "",
        )

        reviewed = (
            "1"
            if review_group_id
            else "0"
        )

        if not appearance_id:
            hero_appearance_ids = sorted(
                appearance_ids_by_hero.get(
                    uid,
                    [],
                )
            )

            if len(hero_appearance_ids) == 1:
                appearance_id = (
                    hero_appearance_ids[0]
                )

        source_decision = row["decision"]

        if reviewed == "1":
            label_source = (
                "human_reviewed_batch_001"
            )
            ocr_text = row.get(
                "corrected_ocr_text",
                "",
            ) or row["ocr_text"]
        else:
            label_source = {
                "CONFIRMED": "visual_and_ocr",
                "RESCUED_BY_OCR": "ocr_rescue",
                "VISUAL_ONLY": "visual_only",
            }.get(
                source_decision,
                "automatic_reconciliation",
            )
            ocr_text = row["ocr_text"]

        identity_rows.append(
            {
                "screenshot_id": row[
                    "screenshot_id"
                ],
                "side": row["side"],
                "slot": row["slot"],
                "avatar_file": row[
                    "avatar_file"
                ],
                "name_file": row[
                    "name_file"
                ],
                "hero_uid": uid,
                "reference_name": row[
                    "final_hero_name"
                ],
                "appearance_id": (
                    appearance_id
                ),
                "ocr_text": ocr_text,
                "ocr_confidence": row[
                    "ocr_confidence"
                ],
                "ocr_status": row[
                    "ocr_status"
                ],
                "source_decision": (
                    source_decision
                ),
                "label_source": (
                    label_source
                ),
                "reviewed": reviewed,
                "review_group_id": (
                    review_group_id
                ),
            }
        )

    if len(identity_rows) != EXPECTED_BATCH_ROWS:
        print(
            f"[ERREUR] Le manifeste final contient "
            f"{len(identity_rows)} lignes.",
            file=sys.stderr,
        )
        return 1

    if any(
        not row["hero_uid"]
        for row in identity_rows
    ):
        print(
            "[ERREUR] Le manifeste final contient "
            "des identités vides.",
            file=sys.stderr,
        )
        return 1

    # Trie les catalogues de façon déterministe.
    heroes.sort(
        key=lambda row: row["hero_uid"]
    )

    aliases.sort(
        key=lambda row: (
            row["hero_uid"],
            normalize_key(row["alias"]),
        )
    )

    appearances.sort(
        key=lambda row: (
            row["hero_uid"],
            row["appearance_id"],
        )
    )

    master_manifest.sort(
        key=lambda row: (
            safe_int(row["screenshot_id"]),
            row["side"],
            safe_int(row["slot"]),
            row["avatar_file"],
        )
    )

    reconciliation.sort(
        key=lambda row: (
            safe_int(row["screenshot_id"]),
            row["side"],
            safe_int(row["slot"]),
        )
    )

    identity_rows.sort(
        key=lambda row: (
            safe_int(str(row["screenshot_id"])),
            str(row["side"]),
            safe_int(str(row["slot"])),
        )
    )

    summary_lines = [
        "Mise à jour du catalogue — hero_batch_001",
        "",
        f"Mode : {'APPLICATION' if args.apply else 'SIMULATION'}",
        f"Héros avant : {len(heroes) - len(created_heroes)}",
        f"Nouveaux héros : {len(created_heroes)}",
        f"Héros après : {len(heroes)}",
        f"Apparences après : {len(appearances)}",
        f"Alias après : {len(aliases)}",
        (
            "Avatars humains ajoutés au manifeste maître : "
            f"{added_master_avatar_rows}"
        ),
        (
            "Avatars dans le manifeste maître après : "
            f"{len(master_manifest)}"
        ),
        (
            "Avatars du lot avec identité finale : "
            f"{len(identity_rows)}"
        ),
        (
            "Avatars du lot revus humainement : "
            f"{sum(int(row['reviewed']) for row in identity_rows)}"
        ),
        (
            "Avatars du lot sans appearance_id : "
            f"{sum(1 for row in identity_rows if not row['appearance_id'])}"
        ),
        "",
        "Nouveaux héros :",
    ]

    if created_heroes:
        summary_lines.extend(
            f"- {row['hero_uid']} : {row['reference_name']}"
            for row in created_heroes
        )
    else:
        summary_lines.append(
            "- Aucun (déjà présents ou exécution répétée)."
        )

    summary_text = "\n".join(
        summary_lines
    ) + "\n"

    if args.apply:
        files_to_backup = [
            HEROES_CSV,
            APPEARANCES_CSV,
            ALIASES_CSV,
            MASTER_AVATAR_MANIFEST_CSV,
            RECONCILIATION_CSV,
            IDENTITY_MANIFEST_CSV,
        ]

        backup_dir = build_backup(
            files_to_backup
        )

        write_csv_atomic(
            HEROES_CSV,
            heroes_fields,
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
            MASTER_AVATAR_MANIFEST_CSV,
            master_manifest_fields,
            master_manifest,
        )

        write_csv_atomic(
            RECONCILIATION_CSV,
            reconciliation_fields,
            reconciliation,
        )

        write_csv_atomic(
            IDENTITY_MANIFEST_CSV,
            identity_fields,
            identity_rows,
        )

        APPLIED_REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path = (
            APPLIED_REPORT_DIR
            / "catalog_update_summary.txt"
        )

        summary_path.write_text(
            summary_text
            + f"\nSauvegarde : {backup_dir}\n",
            encoding="utf-8",
        )

        print(summary_text)
        print(f"Sauvegarde : {backup_dir}")
        print(f"Résumé : {summary_path}")
        print(
            f"Manifeste du lot : "
            f"{IDENTITY_MANIFEST_CSV}"
        )

    else:
        PREVIEW_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        preview_files = {
            "heroes.proposed.csv": (
                heroes_fields,
                heroes,
            ),
            "hero_appearances.proposed.csv": (
                appearance_fields,
                appearances,
            ),
            "hero_name_aliases.proposed.csv": (
                alias_fields,
                aliases,
            ),
            "hero_avatar_manifest.proposed.csv": (
                master_manifest_fields,
                master_manifest,
            ),
            "reconciliation_results.proposed.csv": (
                reconciliation_fields,
                reconciliation,
            ),
            "hero_identity_manifest.proposed.csv": (
                identity_fields,
                identity_rows,
            ),
        }

        for filename, (
            fields,
            rows,
        ) in preview_files.items():
            write_csv_atomic(
                PREVIEW_DIR / filename,
                fields,
                rows,
            )

        summary_path = (
            PREVIEW_DIR
            / "catalog_update_summary.txt"
        )

        summary_path.write_text(
            summary_text,
            encoding="utf-8",
        )

        print(summary_text)
        print(
            "Aucun catalogue n'a été modifié."
        )
        print(
            f"Aperçu : {PREVIEW_DIR}"
        )
        print(
            f"Résumé : {summary_path}"
        )
        print()
        print(
            "Après contrôle, appliquer avec :"
        )
        print(
            "python scripts/"
            "apply_hero_batch_001_review.py "
            "--apply"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
