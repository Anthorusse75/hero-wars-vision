from __future__ import annotations

import argparse
import csv
import html
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from urllib.parse import quote


BATCH_NAME = os.getenv("HERO_BATCH", "hero_batch_001")
BATCH_ROOT = Path("data/batches") / BATCH_NAME

VISUAL_RESULTS = (
    BATCH_ROOT
    / "reports"
    / "visual_matching_dynamic_v1"
    / "visual_match_results.csv"
)

OCR_RESULTS = (
    BATCH_ROOT
    / "reports"
    / "ocr_dynamic_v1"
    / "hero_names_ocr.csv"
)

HERO_CATALOG = Path("data/catalog/heroes.csv")
HERO_ALIASES = Path("data/catalog/hero_name_aliases.csv")

AVATAR_DIR = (
    BATCH_ROOT
    / "crops_dynamic_v1"
    / "avatars_inner"
)

NAME_DIR = (
    BATCH_ROOT
    / "crops_dynamic_v1"
    / "names"
)

OUTPUT_DIR = (
    BATCH_ROOT
    / "reports"
    / "reconciliation_v2"
)

RESULTS_CSV = OUTPUT_DIR / "reconciliation_results.csv"
ALIAS_CANDIDATES_CSV = OUTPUT_DIR / "alias_candidates.csv"
HTML_REPORT = OUTPUT_DIR / "reconciliation_review.html"


FUZZY_MIN_SCORE = 0.94
FUZZY_MIN_GAP = 0.05
FUZZY_MIN_LENGTH = 5

OCR_HIGH = 0.85
OCR_MEDIUM = 0.60

VISUAL_ACCEPTED = "ACCEPTED"

# Corrections OCR guidées par une identité visuelle déjà ACCEPTED.
# Elles ne créent jamais d'alias dans le catalogue.
VISUAL_ONE_EDIT_MIN_SIMILARITY = 0.90
VISUAL_ONE_EDIT_MIN_MARGIN = 0.02
VISUAL_SHORT_ONE_EDIT_MIN_SIMILARITY = 0.91
VISUAL_SHORT_ONE_EDIT_MIN_MARGIN = 0.03
VISUAL_TWO_EDIT_MIN_SIMILARITY = 0.92
VISUAL_TWO_EDIT_MIN_MARGIN = 0.04
VISUAL_TWO_EDIT_MIN_LENGTH = 5
VISUAL_TRUNCATION_MIN_SIMILARITY = 0.90
VISUAL_TRUNCATION_MIN_MARGIN = 0.02
VISUAL_TRUNCATION_MIN_LENGTH = 3
VISUAL_TRUNCATION_MIN_COVERAGE = 0.60

# Pour un résultat visuel REVIEW, on n'autorise qu'une correction OCR
# très encadrée : même longueur, une seule édition, OCR très fiable,
# similarité et marge visuelles minimales.
VISUAL_REVIEW_ONE_EDIT_MIN_SIMILARITY = 0.88
VISUAL_REVIEW_ONE_EDIT_MIN_MARGIN = 0.01
VISUAL_REVIEW_ONE_EDIT_MIN_OCR_CONFIDENCE = 0.60

# Alias validés humainement pendant la revue du batch 002.
# L'option --apply-reviewed-aliases les ajoute au catalogue avec sauvegarde.
REVIEWED_ALIAS_ADDITIONS = (
    {
        "hero_uid": "HW_HERO_0046",
        "alias": "Sebastian",
        "language": "en",
        "source": "human_reviewed_batch_002",
        "occurrences": "1",
        "reviewed": "1",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Réconcilie les identités visuelles et OCR, en ignorant "
            "les petits symboles parasites lus avant ou après les noms."
        )
    )

    parser.add_argument(
        "--apply-reviewed-aliases",
        action="store_true",
        help=(
            "Ajoute au catalogue les alias explicitement validés "
            "dans REVIEWED_ALIAS_ADDITIONS avant la réconciliation."
        ),
    )

    return parser.parse_args()


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


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
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
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
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

    os.replace(temporary, path)


def normalize_display_text(value: str) -> str:
    value = unicodedata.normalize(
        "NFC",
        value,
    )

    value = value.replace("’", "'")
    value = value.replace("`", "'")
    value = " ".join(value.split())

    return value.strip(" -")


def normalize_alias_key(value: str) -> str:
    value = normalize_display_text(value)
    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    characters: list[str] = []

    for character in value.casefold():
        if unicodedata.combining(character):
            continue

        if character.isalnum():
            characters.append(character)
        else:
            characters.append(" ")

    return " ".join(
        "".join(characters).split()
    )


def edit_distance(left: str, right: str) -> int:
    previous = list(
        range(len(right) + 1)
    )

    for left_index, left_character in enumerate(
        left,
        start=1,
    ):
        current = [left_index]

        for right_index, right_character in enumerate(
            right,
            start=1,
        ):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            replacement = (
                previous[right_index - 1]
                + int(left_character != right_character)
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    replacement,
                )
            )

        previous = current

    return previous[-1]


def remove_edge_noise_tokens(
    alias_key: str,
) -> str:
    """
    Supprime uniquement les petits blocs parasites situés aux extrémités.

    Exemples observés :
    - "g Julius" -> "Julius"
    - "Byrna I" -> "Byrna"
    - "o P Mushy and Shroom" -> "Mushy and Shroom"

    Les mots internes ne sont jamais supprimés.
    """

    tokens = alias_key.split()

    while tokens and len(tokens[0]) <= 2:
        tokens.pop(0)

    while tokens and len(tokens[-1]) <= 2:
        tokens.pop()

    return " ".join(tokens)


def apply_reviewed_alias_additions() -> None:
    alias_fields, alias_rows = read_csv(
        HERO_ALIASES
    )

    required_fields = [
        "hero_uid",
        "alias",
        "language",
        "source",
        "occurrences",
        "reviewed",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in alias_fields
    ]

    if missing_fields:
        raise RuntimeError(
            "Colonnes absentes dans "
            f"{HERO_ALIASES} : "
            + ", ".join(missing_fields)
        )

    _, hero_rows = read_csv(
        HERO_CATALOG
    )

    known_heroes = {
        row["hero_uid"]
        for row in hero_rows
    }

    existing_by_key = {
        (
            row["hero_uid"],
            normalize_alias_key(row["alias"]),
        ): row
        for row in alias_rows
    }

    additions = 0
    updates = 0

    for addition in REVIEWED_ALIAS_ADDITIONS:
        hero_uid = addition["hero_uid"]

        if hero_uid not in known_heroes:
            raise RuntimeError(
                "Héros absent du catalogue pour l'alias "
                f"{addition['alias']!r} : {hero_uid}"
            )

        alias_key = normalize_alias_key(
            addition["alias"]
        )

        other_heroes = {
            row["hero_uid"]
            for row in alias_rows
            if (
                normalize_alias_key(row["alias"])
                == alias_key
                and row["hero_uid"] != hero_uid
            )
        }

        if other_heroes:
            raise RuntimeError(
                f"L'alias {addition['alias']!r} est déjà "
                "attribué à un autre héros : "
                + ", ".join(sorted(other_heroes))
            )

        existing = existing_by_key.get(
            (
                hero_uid,
                alias_key,
            )
        )

        if existing is None:
            alias_rows.append(
                {
                    field: addition.get(field, "")
                    for field in alias_fields
                }
            )
            additions += 1
            continue

        changed = False

        for field in (
            "language",
            "source",
            "reviewed",
        ):
            value = addition.get(field, "")

            if value and existing.get(field, "") != value:
                existing[field] = value
                changed = True

        old_occurrences = int(
            existing.get("occurrences") or 0
        )
        new_occurrences = int(
            addition.get("occurrences") or 0
        )

        if new_occurrences > old_occurrences:
            existing["occurrences"] = str(
                new_occurrences
            )
            changed = True

        if changed:
            updates += 1

    if not additions and not updates:
        print(
            "Alias validés : catalogue déjà à jour."
        )
        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_directory = (
        Path("data/catalog/backups")
        / f"reconciliation_v2_aliases_{timestamp}"
    )

    backup_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        HERO_ALIASES,
        backup_directory / HERO_ALIASES.name,
    )

    write_csv(
        HERO_ALIASES,
        alias_fields,
        alias_rows,
    )

    print(
        f"Alias validés ajoutés : {additions}"
    )
    print(
        f"Alias validés actualisés : {updates}"
    )
    print(
        f"Sauvegarde : {backup_directory}"
    )


def load_catalog() -> tuple[
    dict[str, str],
    dict[str, set[str]],
    list[dict[str, str]],
    dict[str, list[dict[str, str]]],
]:
    _, hero_rows = read_csv(HERO_CATALOG)
    _, alias_rows = read_csv(HERO_ALIASES)

    hero_names = {
        row["hero_uid"]: row["reference_name"]
        for row in hero_rows
    }

    alias_to_heroes: dict[
        str,
        set[str],
    ] = defaultdict(set)

    alias_entries: list[
        dict[str, str]
    ] = []

    def add_alias(
        hero_uid: str,
        alias: str,
        source: str,
    ) -> None:
        alias = normalize_display_text(alias)
        alias_key = normalize_alias_key(alias)

        if not alias_key:
            return

        alias_to_heroes[
            alias_key
        ].add(hero_uid)

        alias_entries.append(
            {
                "hero_uid": hero_uid,
                "alias": alias,
                "alias_key": alias_key,
                "source": source,
            }
        )

    for row in hero_rows:
        add_alias(
            hero_uid=row["hero_uid"],
            alias=row["reference_name"],
            source="reference_name",
        )

    for row in alias_rows:
        add_alias(
            hero_uid=row["hero_uid"],
            alias=row["alias"],
            source=row.get(
                "source",
                "catalog_alias",
            ),
        )

    unique_entries: dict[
        tuple[str, str],
        dict[str, str],
    ] = {}

    for entry in alias_entries:
        key = (
            entry["hero_uid"],
            entry["alias_key"],
        )
        unique_entries[key] = entry

    deduplicated = list(
        unique_entries.values()
    )

    entries_by_hero: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for entry in deduplicated:
        entries_by_hero[
            entry["hero_uid"]
        ].append(entry)

    return (
        hero_names,
        alias_to_heroes,
        deduplicated,
        entries_by_hero,
    )


def empty_match(
    alias_key: str,
    matched_alias: str = "",
    score: float = 0.0,
    method: str = "NO_MATCH",
    ambiguous: bool = False,
) -> dict[str, object]:
    return {
        "matched_hero_uid": "",
        "matched_alias": matched_alias,
        "match_method": method,
        "match_score": score,
        "ambiguous": ambiguous,
        "alias_key": alias_key,
        "cleaned_alias_key": "",
    }


def match_ocr_to_catalog(
    ocr_text: str,
    alias_to_heroes: dict[str, set[str]],
    alias_entries: list[dict[str, str]],
) -> dict[str, object]:
    display_text = normalize_display_text(
        ocr_text
    )

    alias_key = normalize_alias_key(
        display_text
    )

    if not alias_key:
        return empty_match(
            alias_key="",
            method="EMPTY",
        )

    exact_heroes = alias_to_heroes.get(
        alias_key,
        set(),
    )

    if len(exact_heroes) == 1:
        hero_uid = next(
            iter(exact_heroes)
        )

        matched_alias = next(
            (
                entry["alias"]
                for entry in alias_entries
                if (
                    entry["hero_uid"] == hero_uid
                    and entry["alias_key"] == alias_key
                )
            ),
            display_text,
        )

        return {
            "matched_hero_uid": hero_uid,
            "matched_alias": matched_alias,
            "match_method": "EXACT_ALIAS",
            "match_score": 1.0,
            "ambiguous": False,
            "alias_key": alias_key,
            "cleaned_alias_key": alias_key,
        }

    if len(exact_heroes) > 1:
        return empty_match(
            alias_key=alias_key,
            score=1.0,
            method="AMBIGUOUS_EXACT_ALIAS",
            ambiguous=True,
        )

    if len(alias_key) < FUZZY_MIN_LENGTH:
        return empty_match(
            alias_key=alias_key,
        )

    best_by_hero: dict[
        str,
        tuple[float, str],
    ] = {}

    for entry in alias_entries:
        score = SequenceMatcher(
            None,
            alias_key,
            entry["alias_key"],
        ).ratio()

        current = best_by_hero.get(
            entry["hero_uid"]
        )

        if (
            current is None
            or score > current[0]
        ):
            best_by_hero[
                entry["hero_uid"]
            ] = (
                score,
                entry["alias"],
            )

    ranked = sorted(
        (
            (
                hero_uid,
                score_alias[0],
                score_alias[1],
            )
            for hero_uid, score_alias
            in best_by_hero.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    if not ranked:
        return empty_match(
            alias_key=alias_key,
        )

    best_hero, best_score, best_alias = (
        ranked[0]
    )

    second_score = (
        ranked[1][1]
        if len(ranked) > 1
        else 0.0
    )

    gap = best_score - second_score

    if (
        best_score >= FUZZY_MIN_SCORE
        and gap >= FUZZY_MIN_GAP
    ):
        return {
            "matched_hero_uid": best_hero,
            "matched_alias": best_alias,
            "match_method": "FUZZY_ALIAS",
            "match_score": best_score,
            "ambiguous": False,
            "alias_key": alias_key,
            "cleaned_alias_key": alias_key,
        }

    return empty_match(
        alias_key=alias_key,
        matched_alias=best_alias,
        score=best_score,
    )


def visual_identity_match(
    visual_uid: str,
    entry: dict[str, str],
    alias_key: str,
    cleaned_alias_key: str,
    method: str,
    score: float,
) -> dict[str, object]:
    return {
        "matched_hero_uid": visual_uid,
        "matched_alias": entry["alias"],
        "match_method": method,
        "match_score": score,
        "ambiguous": False,
        "alias_key": alias_key,
        "cleaned_alias_key": cleaned_alias_key,
    }


def visual_is_strong_enough(
    visual_similarity: float,
    visual_margin: float,
    minimum_similarity: float,
    minimum_margin: float,
) -> bool:
    return (
        visual_similarity >= minimum_similarity
        and visual_margin >= minimum_margin
    )


def match_ocr_with_visual_identity(
    ocr_text: str,
    visual_uid: str,
    visual_status: str,
    visual_similarity: float,
    visual_margin: float,
    ocr_confidence: float,
    entries_by_hero: dict[
        str,
        list[dict[str, str]],
    ],
) -> dict[str, object] | None:
    """
    Corrige prudemment l'OCR en se limitant au héros déjà reconnu visuellement.

    Les corrections acceptées sont :
    1. petits blocs parasites situés aux extrémités ;
    2. une erreur d'édition sur un nom de même longueur ;
    3. un nom suffisamment long tronqué à gauche ou à droite ;
    4. une troncature accompagnée d'une seule erreur OCR.

    Ces corrections ne deviennent jamais des alias de catalogue.
    """

    alias_key = normalize_alias_key(
        ocr_text
    )

    if not alias_key:
        return None

    visual_entries = entries_by_hero.get(
        visual_uid,
        [],
    )

    if not visual_entries:
        return None

    # Un statut REVIEW n'est pas assez solide pour autoriser les règles
    # générales de troncature ou de suppression de parasites. On accepte
    # uniquement un nom OCR très fiable, de même longueur, à une édition
    # du héros proposé visuellement.
    if visual_status == "REVIEW":
        if (
            ocr_confidence
            < VISUAL_REVIEW_ONE_EDIT_MIN_OCR_CONFIDENCE
            or not visual_is_strong_enough(
                visual_similarity,
                visual_margin,
                VISUAL_REVIEW_ONE_EDIT_MIN_SIMILARITY,
                VISUAL_REVIEW_ONE_EDIT_MIN_MARGIN,
            )
        ):
            return None

        for entry in visual_entries:
            known_key = entry["alias_key"]

            if (
                len(alias_key) != len(known_key)
                or len(known_key) < 4
            ):
                continue

            if edit_distance(alias_key, known_key) == 1:
                return visual_identity_match(
                    visual_uid=visual_uid,
                    entry=entry,
                    alias_key=alias_key,
                    cleaned_alias_key=known_key,
                    method="VISUAL_REVIEW_ALIAS_ONE_EDIT_OCR",
                    score=1.0 - 1.0 / len(known_key),
                )

        return None

    if visual_status != VISUAL_ACCEPTED:
        return None

    candidate_tokens = alias_key.split()

    # 1. Le nom connu est présent intégralement et seuls de petits tokens
    # parasites ont été ajoutés avant ou après.
    for entry in visual_entries:
        known_tokens = entry["alias_key"].split()

        if len(candidate_tokens) <= len(known_tokens):
            continue

        maximum_start = (
            len(candidate_tokens)
            - len(known_tokens)
        )

        for start in range(
            maximum_start + 1
        ):
            end = start + len(
                known_tokens
            )

            if (
                candidate_tokens[start:end]
                != known_tokens
            ):
                continue

            prefix = candidate_tokens[:start]
            suffix = candidate_tokens[end:]

            if not prefix and not suffix:
                continue

            if all(
                len(token) <= 2
                for token in (
                    prefix + suffix
                )
            ):
                return visual_identity_match(
                    visual_uid=visual_uid,
                    entry=entry,
                    alias_key=alias_key,
                    cleaned_alias_key=entry["alias_key"],
                    method="VISUAL_ALIAS_EDGE_NOISE",
                    score=1.0,
                )

    # 2. Une seule insertion, suppression ou substitution sur l'ensemble
    # du nom, espaces compris. Les noms de trois caractères exigent un
    # visuel plus solide afin d'éviter les faux positifs.
    for entry in visual_entries:
        known_key = entry["alias_key"]

        if (
            len(alias_key) != len(known_key)
            or len(alias_key) < 3
        ):
            continue

        if len(known_key) == 3:
            strong_enough = visual_is_strong_enough(
                visual_similarity,
                visual_margin,
                VISUAL_SHORT_ONE_EDIT_MIN_SIMILARITY,
                VISUAL_SHORT_ONE_EDIT_MIN_MARGIN,
            )
        else:
            strong_enough = visual_is_strong_enough(
                visual_similarity,
                visual_margin,
                VISUAL_ONE_EDIT_MIN_SIMILARITY,
                VISUAL_ONE_EDIT_MIN_MARGIN,
            )

        if not strong_enough:
            continue

        if edit_distance(
            alias_key,
            known_key,
        ) == 1:
            return visual_identity_match(
                visual_uid=visual_uid,
                entry=entry,
                alias_key=alias_key,
                cleaned_alias_key=known_key,
                method="VISUAL_ALIAS_ONE_EDIT_OCR",
                score=(
                    1.0
                    - 1.0 / len(known_key)
                ),
            )

    # 3. Deux erreurs OCR sur un nom d'au moins cinq caractères,
    # uniquement lorsque l'identité visuelle ACCEPTED est très solide.
    # Cette règle couvre notamment des permutations ou substitutions comme
    # "Eolfo" pour "Folio" et "Dolaric" pour "Polaris".
    if visual_is_strong_enough(
        visual_similarity,
        visual_margin,
        VISUAL_TWO_EDIT_MIN_SIMILARITY,
        VISUAL_TWO_EDIT_MIN_MARGIN,
    ):
        for entry in visual_entries:
            known_key = entry["alias_key"]

            if (
                len(alias_key) != len(known_key)
                or len(known_key) < VISUAL_TWO_EDIT_MIN_LENGTH
            ):
                continue

            if edit_distance(alias_key, known_key) == 2:
                return visual_identity_match(
                    visual_uid=visual_uid,
                    entry=entry,
                    alias_key=alias_key,
                    cleaned_alias_key=known_key,
                    method="VISUAL_ALIAS_TWO_EDIT_OCR",
                    score=1.0 - 2.0 / len(known_key),
                )

    # 4 et 5. Le recadrage du nom peut supprimer le début ou la fin
    # d'un nom. On exige une portion observée d'au moins 60 % du nom
    # connu et une identité visuelle solide.
    # au moins 60 % du nom connu, et une identité visuelle solide.
    if not visual_is_strong_enough(
        visual_similarity,
        visual_margin,
        VISUAL_TRUNCATION_MIN_SIMILARITY,
        VISUAL_TRUNCATION_MIN_MARGIN,
    ):
        return None

    for entry in visual_entries:
        known_key = entry["alias_key"]

        if (
            len(alias_key) >= len(known_key)
            or len(alias_key)
            < VISUAL_TRUNCATION_MIN_LENGTH
        ):
            continue

        coverage = (
            len(alias_key)
            / len(known_key)
        )

        if (
            coverage
            < VISUAL_TRUNCATION_MIN_COVERAGE
        ):
            continue

        known_prefix = known_key[
            : len(alias_key)
        ]
        known_suffix = known_key[
            -len(alias_key) :
        ]

        if alias_key == known_prefix:
            return visual_identity_match(
                visual_uid=visual_uid,
                entry=entry,
                alias_key=alias_key,
                cleaned_alias_key=known_key,
                method="VISUAL_ALIAS_TRUNCATED_RIGHT",
                score=coverage,
            )

        if alias_key == known_suffix:
            return visual_identity_match(
                visual_uid=visual_uid,
                entry=entry,
                alias_key=alias_key,
                cleaned_alias_key=known_key,
                method="VISUAL_ALIAS_TRUNCATED_LEFT",
                score=coverage,
            )

        prefix_distance = edit_distance(
            alias_key,
            known_prefix,
        )
        suffix_distance = edit_distance(
            alias_key,
            known_suffix,
        )

        if min(
            prefix_distance,
            suffix_distance,
        ) == 1:
            method = (
                "VISUAL_ALIAS_TRUNCATED_RIGHT_ONE_EDIT"
                if prefix_distance
                <= suffix_distance
                else "VISUAL_ALIAS_TRUNCATED_LEFT_ONE_EDIT"
            )

            return visual_identity_match(
                visual_uid=visual_uid,
                entry=entry,
                alias_key=alias_key,
                cleaned_alias_key=known_key,
                method=method,
                score=(
                    coverage
                    * (
                        1.0
                        - 1.0
                        / len(alias_key)
                    )
                ),
            )

    return None


def make_join_key(
    row: dict[str, str],
) -> tuple[str, str, str]:
    return (
        str(row["screenshot_id"]),
        str(row["side"]),
        str(row["slot"]),
    )


def reconcile_row(
    visual_row: dict[str, str],
    ocr_row: dict[str, str],
    hero_names: dict[str, str],
    alias_to_heroes: dict[str, set[str]],
    alias_entries: list[dict[str, str]],
    entries_by_hero: dict[
        str,
        list[dict[str, str]],
    ],
) -> dict[str, object]:
    visual_uid = visual_row[
        "predicted_hero_uid"
    ]

    visual_name = visual_row[
        "predicted_name"
    ]

    visual_status = visual_row[
        "status"
    ]

    visual_similarity = float(
        visual_row["similarity"]
    )

    visual_margin = float(
        visual_row["margin"]
    )

    ocr_text = normalize_display_text(
        ocr_row["ocr_text"]
    )

    ocr_confidence = float(
        ocr_row["confidence"]
    )

    ocr_status = ocr_row["status"]

    catalog_match = match_ocr_to_catalog(
        ocr_text=ocr_text,
        alias_to_heroes=alias_to_heroes,
        alias_entries=alias_entries,
    )

    if (
        not catalog_match[
            "matched_hero_uid"
        ]
        and not catalog_match[
            "ambiguous"
        ]
        and visual_status
        in {VISUAL_ACCEPTED, "REVIEW"}
        and ocr_text
        and ocr_confidence >= OCR_HIGH
    ):
        identity_match = (
            match_ocr_with_visual_identity(
                ocr_text=ocr_text,
                visual_uid=visual_uid,
                visual_status=visual_status,
                visual_similarity=(
                    visual_similarity
                ),
                visual_margin=visual_margin,
                ocr_confidence=ocr_confidence,
                entries_by_hero=(
                    entries_by_hero
                ),
            )
        )

        if identity_match is not None:
            catalog_match = identity_match

    ocr_uid = str(
        catalog_match[
            "matched_hero_uid"
        ]
    )

    ocr_match_method = str(
        catalog_match["match_method"]
    )

    ocr_match_score = float(
        catalog_match["match_score"]
    )

    alias_ambiguous = bool(
        catalog_match["ambiguous"]
    )

    final_hero_uid = ""
    final_hero_name = ""
    decision = ""
    review_required = 1

    if alias_ambiguous:
        decision = "AMBIGUOUS_ALIAS"

    elif ocr_uid:
        if ocr_uid == visual_uid:
            final_hero_uid = visual_uid
            final_hero_name = (
                hero_names.get(
                    visual_uid,
                    visual_name,
                )
            )

            if (
                visual_status
                == VISUAL_ACCEPTED
            ):
                decision = "CONFIRMED"
                review_required = 0

            elif (
                ocr_confidence >= OCR_MEDIUM
            ):
                decision = "RESCUED_BY_OCR"
                review_required = 0

            else:
                decision = (
                    "AGREEMENT_REVIEW"
                )

        else:
            # Lorsque le visuel est seulement REVIEW, avec une marge très
            # faible, un nom OCR exact et très fiable est plus probant que
            # le voisin visuel proposé. Cette règle ne s'applique jamais à
            # un visuel ACCEPTED.
            if (
                visual_status == "REVIEW"
                and ocr_confidence >= OCR_HIGH
                and ocr_match_method
                in {"EXACT_ALIAS", "FUZZY_ALIAS"}
                and visual_margin <= 0.015
            ):
                final_hero_uid = ocr_uid
                final_hero_name = hero_names.get(
                    ocr_uid,
                    str(catalog_match["matched_alias"]),
                )
                decision = "RESCUED_BY_OCR"
                review_required = 0
            else:
                decision = "CONFLICT"

    elif visual_status == VISUAL_ACCEPTED:
        final_hero_uid = visual_uid
        final_hero_name = (
            hero_names.get(
                visual_uid,
                visual_name,
            )
        )

        if (
            ocr_text
            and ocr_confidence >= OCR_HIGH
        ):
            decision = (
                "NEW_ALIAS_CANDIDATE"
            )
        else:
            decision = "VISUAL_ONLY"

        review_required = int(
            decision
            == "NEW_ALIAS_CANDIDATE"
        )

    else:
        if (
            ocr_text
            and ocr_confidence >= OCR_HIGH
        ):
            decision = (
                "UNRESOLVED_OCR_TEXT"
            )
        else:
            decision = "MANUAL_REVIEW"

    return {
        "screenshot_id": visual_row[
            "screenshot_id"
        ],
        "side": visual_row["side"],
        "slot": visual_row["slot"],
        "avatar_file": visual_row[
            "avatar_file"
        ],
        "name_file": ocr_row["filename"],
        "visual_hero_uid": visual_uid,
        "visual_hero_name": visual_name,
        "visual_status": visual_status,
        "visual_similarity": (
            visual_similarity
        ),
        "visual_margin": visual_margin,
        "ocr_text": ocr_text,
        "ocr_confidence": (
            ocr_confidence
        ),
        "ocr_status": ocr_status,
        "ocr_matched_hero_uid": (
            ocr_uid
        ),
        "ocr_matched_hero_name": (
            hero_names.get(
                ocr_uid,
                "",
            )
            if ocr_uid
            else ""
        ),
        "ocr_matched_alias": str(
            catalog_match[
                "matched_alias"
            ]
        ),
        "ocr_match_method": (
            ocr_match_method
        ),
        "ocr_match_score": (
            ocr_match_score
        ),
        "ocr_alias_key": str(
            catalog_match[
                "alias_key"
            ]
        ),
        "ocr_cleaned_alias_key": str(
            catalog_match.get(
                "cleaned_alias_key",
                "",
            )
        ),
        "decision": decision,
        "final_hero_uid": (
            final_hero_uid
        ),
        "final_hero_name": (
            final_hero_name
        ),
        "review_required": (
            review_required
        ),
    }


def build_alias_candidates(
    result_rows: list[
        dict[str, object]
    ],
    hero_names: dict[str, str],
) -> list[dict[str, object]]:
    observations: dict[
        tuple[str, str],
        list[dict[str, object]],
    ] = defaultdict(list)

    heroes_by_alias_key: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for row in result_rows:
        if (
            row["decision"]
            != "NEW_ALIAS_CANDIDATE"
        ):
            continue

        hero_uid = str(
            row["visual_hero_uid"]
        )

        alias_key = str(
            row["ocr_alias_key"]
        )

        if not alias_key:
            continue

        observations[
            (
                hero_uid,
                alias_key,
            )
        ].append(row)

        heroes_by_alias_key[
            alias_key
        ].add(hero_uid)

    candidate_rows: list[
        dict[str, object]
    ] = []

    for (
        hero_uid,
        alias_key,
    ), rows in observations.items():
        display_counter = Counter(
            str(row["ocr_text"])
            for row in rows
        )

        display_alias = (
            display_counter
            .most_common(1)[0][0]
        )

        ocr_confidences = [
            float(
                row["ocr_confidence"]
            )
            for row in rows
        ]

        visual_similarities = [
            float(
                row["visual_similarity"]
            )
            for row in rows
        ]

        visual_margins = [
            float(
                row["visual_margin"]
            )
            for row in rows
        ]

        cross_hero_conflict = (
            len(
                heroes_by_alias_key[
                    alias_key
                ]
            )
            > 1
        )

        if cross_hero_conflict:
            candidate_status = (
                "CONFLICT_BETWEEN_HEROES"
            )
        elif len(rows) >= 2:
            candidate_status = (
                "READY_FOR_REVIEW"
            )
        else:
            candidate_status = (
                "SINGLE_OBSERVATION"
            )

        examples = " | ".join(
            (
                f"{row['screenshot_id']}"
                f"-{row['side']}"
                f"{row['slot']}"
            )
            for row in rows[:10]
        )

        candidate_rows.append(
            {
                "hero_uid": hero_uid,
                "hero_name": (
                    hero_names.get(
                        hero_uid,
                        hero_uid,
                    )
                ),
                "alias": display_alias,
                "alias_key": alias_key,
                "occurrences": len(rows),
                "mean_ocr_confidence": (
                    mean(
                        ocr_confidences
                    )
                ),
                "minimum_ocr_confidence": (
                    min(
                        ocr_confidences
                    )
                ),
                "mean_visual_similarity": (
                    mean(
                        visual_similarities
                    )
                ),
                "minimum_visual_similarity": (
                    min(
                        visual_similarities
                    )
                ),
                "mean_visual_margin": (
                    mean(
                        visual_margins
                    )
                ),
                "cross_hero_conflict": int(
                    cross_hero_conflict
                ),
                "candidate_status": (
                    candidate_status
                ),
                "examples": examples,
            }
        )

    return sorted(
        candidate_rows,
        key=lambda row: (
            row["candidate_status"]
            == "CONFLICT_BETWEEN_HEROES",
            -int(row["occurrences"]),
            str(row["hero_name"]),
            str(row["alias"]),
        ),
    )


def relative_image_url(
    image_path: Path,
) -> str:
    relative_path = os.path.relpath(
        image_path,
        OUTPUT_DIR,
    ).replace("\\", "/")

    return quote(relative_path)


def status_css_class(
    decision: str,
) -> str:
    return {
        "CONFIRMED": "confirmed",
        "RESCUED_BY_OCR": "rescued",
        "NEW_ALIAS_CANDIDATE": "alias",
        "VISUAL_ONLY": "visual",
        "AGREEMENT_REVIEW": "review",
        "UNRESOLVED_OCR_TEXT": "review",
        "MANUAL_REVIEW": "review",
        "AMBIGUOUS_ALIAS": "ambiguous",
        "CONFLICT": "conflict",
    }.get(
        decision,
        "review",
    )


def create_html_report(
    result_rows: list[
        dict[str, object]
    ],
    alias_candidates: list[
        dict[str, object]
    ],
) -> None:
    decision_counts = Counter(
        str(row["decision"])
        for row in result_rows
    )

    priority = {
        "CONFLICT": 0,
        "AMBIGUOUS_ALIAS": 1,
        "UNRESOLVED_OCR_TEXT": 2,
        "MANUAL_REVIEW": 3,
        "AGREEMENT_REVIEW": 4,
        "NEW_ALIAS_CANDIDATE": 5,
        "VISUAL_ONLY": 6,
        "RESCUED_BY_OCR": 7,
        "CONFIRMED": 8,
    }

    review_rows = [
        row
        for row in result_rows
        if row["decision"]
        != "CONFIRMED"
    ]

    weakest_confirmed = sorted(
        (
            row
            for row in result_rows
            if row["decision"]
            == "CONFIRMED"
        ),
        key=lambda row: (
            float(
                row[
                    "visual_margin"
                ]
            ),
            float(
                row[
                    "visual_similarity"
                ]
            ),
            float(
                row[
                    "ocr_confidence"
                ]
            ),
        ),
    )[:50]

    report_rows = (
        review_rows
        + weakest_confirmed
    )

    report_rows.sort(
        key=lambda row: (
            priority.get(
                str(
                    row["decision"]
                ),
                99,
            ),
            float(
                row[
                    "ocr_confidence"
                ]
            ),
            float(
                row[
                    "visual_similarity"
                ]
            ),
        )
    )

    alias_table_rows: list[str] = []

    for row in alias_candidates:
        alias_table_rows.append(
            f"""
            <tr>
                <td>{html.escape(str(row["hero_name"]))}</td>
                <td class="ocr">{html.escape(str(row["alias"]))}</td>
                <td>{int(row["occurrences"])}</td>
                <td>{float(row["mean_ocr_confidence"]):.4f}</td>
                <td>{float(row["mean_visual_similarity"]):.4f}</td>
                <td>{float(row["mean_visual_margin"]):.4f}</td>
                <td>{html.escape(str(row["candidate_status"]))}</td>
                <td>{html.escape(str(row["examples"]))}</td>
            </tr>
            """
        )

    cards: list[str] = []

    for row in report_rows:
        avatar_path = (
            AVATAR_DIR
            / str(
                row[
                    "avatar_file"
                ]
            )
        )

        name_path = (
            NAME_DIR
            / str(
                row[
                    "name_file"
                ]
            )
        )

        decision = str(
            row["decision"]
        )

        cards.append(
            f"""
            <article class="card {status_css_class(decision)}">
                <div class="images">
                    <img
                        class="avatar"
                        src="{relative_image_url(avatar_path)}"
                        alt="{html.escape(str(row["avatar_file"]))}"
                    >
                    <img
                        class="name"
                        src="{relative_image_url(name_path)}"
                        alt="{html.escape(str(row["name_file"]))}"
                    >
                </div>
                <h2>{html.escape(decision)}</h2>
                <p>
                    <strong>Visuel :</strong>
                    {html.escape(str(row["visual_hero_name"]))}
                    — {html.escape(str(row["visual_status"]))}<br>
                    Similarité : {float(row["visual_similarity"]):.4f}<br>
                    Marge : {float(row["visual_margin"]):.4f}
                </p>
                <p>
                    <strong>OCR :</strong>
                    <span class="ocr">{html.escape(str(row["ocr_text"]))}</span><br>
                    Confiance : {float(row["ocr_confidence"]):.4f}<br>
                    Correspondance : {html.escape(str(row["ocr_matched_hero_name"]) or "aucune")}<br>
                    Méthode : {html.escape(str(row["ocr_match_method"]))}
                </p>
                <p>
                    <strong>Résultat final :</strong>
                    {html.escape(str(row["final_hero_name"]) or "à vérifier")}
                </p>
                <p class="filename">
                    Capture {html.escape(str(row["screenshot_id"]))}
                    — {html.escape(str(row["side"]))}
                    {html.escape(str(row["slot"]))}
                </p>
            </article>
            """
        )

    summary_items = "".join(
        (
            "<li>"
            f"{html.escape(decision)} : {count}"
            "</li>"
        )
        for decision, count in sorted(
            decision_counts.items(),
            key=lambda item: (
                priority.get(
                    item[0],
                    99,
                ),
                item[0],
            ),
        )
    )

    document = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <title>Réconciliation visuelle et OCR V3</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #eeeeee;
        }}
        .summary, .aliases {{
            background: white;
            border: 1px solid #cccccc;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            border: 1px solid #cccccc;
            padding: 7px;
            text-align: left;
        }}
        th {{
            background: #222222;
            color: white;
        }}
        .grid {{
            display: grid;
            grid-template-columns:
                repeat(auto-fill, minmax(330px, 1fr));
            gap: 14px;
        }}
        .card {{
            background: white;
            border: 4px solid #bdbdbd;
            border-radius: 8px;
            padding: 12px;
        }}
        .card.confirmed {{ border-color: #43a047; }}
        .card.rescued {{ border-color: #00897b; }}
        .card.alias {{ border-color: #1e88e5; }}
        .card.visual {{ border-color: #7e57c2; }}
        .card.review {{ border-color: #f9a825; }}
        .card.ambiguous {{ border-color: #ef6c00; }}
        .card.conflict {{
            border-color: #c62828;
            background: #ffebee;
        }}
        .images {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }}
        .avatar {{
            width: 135px;
            height: 135px;
            object-fit: contain;
            background: #222222;
        }}
        .name {{
            width: 300px;
            max-width: 100%;
            height: 62px;
            object-fit: contain;
            background: #222222;
        }}
        .ocr {{
            font-size: 20px;
            font-weight: bold;
        }}
        .filename {{
            color: #555555;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <section class="summary">
        <h1>Réconciliation visuelle et OCR V3</h1>
        <p>
            Les parasites, erreurs d'un caractère et noms tronqués sont corrigés
            sans être ajoutés comme alias.
        </p>
        <ul>{summary_items}</ul>
    </section>
    <section class="aliases">
        <h2>Vrais alias encore candidats</h2>
        <table>
            <thead>
                <tr>
                    <th>Héros</th>
                    <th>Alias OCR proposé</th>
                    <th>Occurrences</th>
                    <th>Confiance OCR moyenne</th>
                    <th>Similarité visuelle moyenne</th>
                    <th>Marge visuelle moyenne</th>
                    <th>Statut</th>
                    <th>Exemples</th>
                </tr>
            </thead>
            <tbody>{''.join(alias_table_rows)}</tbody>
        </table>
    </section>
    <main class="grid">{''.join(cards)}</main>
</body>
</html>
"""

    HTML_REPORT.write_text(
        document,
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    try:
        if args.apply_reviewed_aliases:
            apply_reviewed_alias_additions()

        (
            hero_names,
            alias_to_heroes,
            alias_entries,
            entries_by_hero,
        ) = load_catalog()

        _, visual_rows = read_csv(
            VISUAL_RESULTS
        )

        _, ocr_rows = read_csv(
            OCR_RESULTS
        )

    except RuntimeError as error:
        print(
            error,
            file=sys.stderr,
        )
        return 1

    ocr_by_key = {
        make_join_key(row): row
        for row in ocr_rows
    }

    print(
        f"Batch : {BATCH_NAME}"
    )
    print(
        f"Résultats visuels : "
        f"{len(visual_rows)}"
    )
    print(
        f"Résultats OCR : "
        f"{len(ocr_rows)}"
    )
    print()

    result_rows: list[
        dict[str, object]
    ] = []

    missing_ocr = 0

    for visual_row in visual_rows:
        key = make_join_key(
            visual_row
        )

        ocr_row = ocr_by_key.get(
            key
        )

        if ocr_row is None:
            print(
                "OCR absent pour "
                f"{key[0]} "
                f"{key[1]}{key[2]}",
                file=sys.stderr,
            )
            missing_ocr += 1
            continue

        result_rows.append(
            reconcile_row(
                visual_row=visual_row,
                ocr_row=ocr_row,
                hero_names=hero_names,
                alias_to_heroes=(
                    alias_to_heroes
                ),
                alias_entries=(
                    alias_entries
                ),
                entries_by_hero=(
                    entries_by_hero
                ),
            )
        )

    if missing_ocr:
        print(
            "Correspondances OCR "
            f"manquantes : {missing_ocr}",
            file=sys.stderr,
        )
        return 1

    alias_candidates = (
        build_alias_candidates(
            result_rows=result_rows,
            hero_names=hero_names,
        )
    )

    result_fieldnames = [
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
        "ocr_matched_hero_uid",
        "ocr_matched_hero_name",
        "ocr_matched_alias",
        "ocr_match_method",
        "ocr_match_score",
        "ocr_alias_key",
        "ocr_cleaned_alias_key",
        "decision",
        "final_hero_uid",
        "final_hero_name",
        "review_required",
    ]

    write_csv(
        RESULTS_CSV,
        result_fieldnames,
        result_rows,
    )

    alias_fieldnames = [
        "hero_uid",
        "hero_name",
        "alias",
        "alias_key",
        "occurrences",
        "mean_ocr_confidence",
        "minimum_ocr_confidence",
        "mean_visual_similarity",
        "minimum_visual_similarity",
        "mean_visual_margin",
        "cross_hero_conflict",
        "candidate_status",
        "examples",
    ]

    write_csv(
        ALIAS_CANDIDATES_CSV,
        alias_fieldnames,
        alias_candidates,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    create_html_report(
        result_rows=result_rows,
        alias_candidates=(
            alias_candidates
        ),
    )

    decision_counts = Counter(
        str(row["decision"])
        for row in result_rows
    )

    final_assigned = sum(
        1
        for row in result_rows
        if row["final_hero_uid"]
    )

    review_required = sum(
        int(row["review_required"])
        for row in result_rows
    )

    corrected_edge_noise = sum(
        1
        for row in result_rows
        if row["ocr_match_method"]
        == "VISUAL_ALIAS_EDGE_NOISE"
    )

    corrected_one_edit = sum(
        1
        for row in result_rows
        if row["ocr_match_method"]
        in {
            "VISUAL_ALIAS_ONE_EDIT_OCR",
            "VISUAL_REVIEW_ALIAS_ONE_EDIT_OCR",
        }
    )

    corrected_two_edit = sum(
        1
        for row in result_rows
        if row["ocr_match_method"]
        == "VISUAL_ALIAS_TWO_EDIT_OCR"
    )

    corrected_truncation = sum(
        1
        for row in result_rows
        if str(row["ocr_match_method"]).startswith(
            "VISUAL_ALIAS_TRUNCATED_"
        )
    )

    corrected_truncation_one_edit = sum(
        1
        for row in result_rows
        if str(row["ocr_match_method"]).endswith(
            "_ONE_EDIT"
        )
    )

    print("Résumé :")

    ordered_decisions = (
        "CONFIRMED",
        "RESCUED_BY_OCR",
        "NEW_ALIAS_CANDIDATE",
        "VISUAL_ONLY",
        "AGREEMENT_REVIEW",
        "UNRESOLVED_OCR_TEXT",
        "MANUAL_REVIEW",
        "AMBIGUOUS_ALIAS",
        "CONFLICT",
    )

    for decision in ordered_decisions:
        print(
            f"- {decision:<23} : "
            f"{decision_counts.get(decision, 0)}"
        )

    print()
    print(
        "- Parasites de bord corrigés : "
        f"{corrected_edge_noise}"
    )
    print(
        "- Erreurs OCR à 1 caractère : "
        f"{corrected_one_edit}"
    )

    print(
        "- Erreurs OCR à 2 caractères : "
        f"{corrected_two_edit}"
    )
    print(
        "- Noms tronqués corrigés : "
        f"{corrected_truncation}"
    )
    print(
        "- Dont troncatures avec 1 erreur OCR : "
        f"{corrected_truncation_one_edit}"
    )
    print(
        "- Identités finales attribuées : "
        f"{final_assigned}"
    )
    print(
        "- Cas nécessitant une revue : "
        f"{review_required}"
    )
    print(
        "- Alias candidats : "
        f"{len(alias_candidates)}"
    )
    print()
    print(
        f"Résultats : {RESULTS_CSV}"
    )
    print(
        "Alias candidats : "
        f"{ALIAS_CANDIDATES_CSV}"
    )
    print(
        f"Contrôle visuel : {HTML_REPORT}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
