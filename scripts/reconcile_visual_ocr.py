from __future__ import annotations

import csv
import html
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from urllib.parse import quote


BATCH_ROOT = Path("data/batches/hero_batch_001")

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
    / "reconciliation_v1"
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Fichier absent : {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        return list(
            csv.DictReader(
                csv_file,
                delimiter=";",
            )
        )


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            delimiter=";",
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


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
    """
    Clé insensible :
    - à la casse ;
    - aux accents ;
    - aux apostrophes et séparateurs ;
    - aux espaces multiples.

    Les alphabets non latins sont conservés.
    """

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


def load_catalog() -> tuple[
    dict[str, str],
    dict[str, set[str]],
    list[dict[str, str]],
]:
    hero_rows = read_csv(HERO_CATALOG)
    alias_rows = read_csv(HERO_ALIASES)

    hero_names = {
        row["hero_uid"]: row["reference_name"]
        for row in hero_rows
    }

    alias_to_heroes: dict[str, set[str]] = defaultdict(set)
    alias_entries: list[dict[str, str]] = []

    def add_alias(
        hero_uid: str,
        alias: str,
        source: str,
    ) -> None:
        alias = normalize_display_text(alias)
        alias_key = normalize_alias_key(alias)

        if not alias_key:
            return

        alias_to_heroes[alias_key].add(hero_uid)

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
            source=row.get("source", "catalog_alias"),
        )

    # Déduplique les mêmes clés pour éviter de refaire
    # inutilement les mêmes comparaisons floues.
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

    return (
        hero_names,
        alias_to_heroes,
        list(unique_entries.values()),
    )


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
        return {
            "matched_hero_uid": "",
            "matched_alias": "",
            "match_method": "EMPTY",
            "match_score": 0.0,
            "ambiguous": False,
            "alias_key": "",
        }

    exact_heroes = alias_to_heroes.get(
        alias_key,
        set(),
    )

    if len(exact_heroes) == 1:
        hero_uid = next(iter(exact_heroes))

        matched_alias = next(
            (
                entry["alias"]
                for entry in alias_entries
                if entry["hero_uid"] == hero_uid
                and entry["alias_key"] == alias_key
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
        }

    if len(exact_heroes) > 1:
        return {
            "matched_hero_uid": "",
            "matched_alias": "",
            "match_method": "AMBIGUOUS_EXACT_ALIAS",
            "match_score": 1.0,
            "ambiguous": True,
            "alias_key": alias_key,
        }

    if len(alias_key) < FUZZY_MIN_LENGTH:
        return {
            "matched_hero_uid": "",
            "matched_alias": "",
            "match_method": "NO_MATCH",
            "match_score": 0.0,
            "ambiguous": False,
            "alias_key": alias_key,
        }

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

        if current is None or score > current[0]:
            best_by_hero[entry["hero_uid"]] = (
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
        return {
            "matched_hero_uid": "",
            "matched_alias": "",
            "match_method": "NO_MATCH",
            "match_score": 0.0,
            "ambiguous": False,
            "alias_key": alias_key,
        }

    best_hero, best_score, best_alias = ranked[0]

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
        }

    return {
        "matched_hero_uid": "",
        "matched_alias": best_alias,
        "match_method": "NO_MATCH",
        "match_score": best_score,
        "ambiguous": False,
        "alias_key": alias_key,
    }


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
) -> dict[str, object]:
    visual_uid = visual_row[
        "predicted_hero_uid"
    ]

    visual_name = visual_row[
        "predicted_name"
    ]

    visual_status = visual_row["status"]
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

    ocr_uid = str(
        catalog_match["matched_hero_uid"]
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
            final_hero_name = hero_names.get(
                visual_uid,
                visual_name,
            )

            if visual_status == VISUAL_ACCEPTED:
                decision = "CONFIRMED"
                review_required = 0

            elif ocr_confidence >= OCR_HIGH:
                decision = "RESCUED_BY_OCR"
                review_required = 0

            else:
                decision = "AGREEMENT_REVIEW"

        else:
            decision = "CONFLICT"

    elif visual_status == VISUAL_ACCEPTED:
        final_hero_uid = visual_uid
        final_hero_name = hero_names.get(
            visual_uid,
            visual_name,
        )

        if (
            ocr_text
            and ocr_confidence >= OCR_HIGH
        ):
            decision = "NEW_ALIAS_CANDIDATE"
        else:
            decision = "VISUAL_ONLY"

        review_required = (
            1
            if decision == "NEW_ALIAS_CANDIDATE"
            else 0
        )

    else:
        if (
            ocr_text
            and ocr_confidence >= OCR_HIGH
        ):
            decision = "UNRESOLVED_OCR_TEXT"
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
        "visual_similarity": visual_similarity,
        "visual_margin": visual_margin,
        "ocr_text": ocr_text,
        "ocr_confidence": ocr_confidence,
        "ocr_status": ocr_status,
        "ocr_matched_hero_uid": ocr_uid,
        "ocr_matched_hero_name": (
            hero_names.get(ocr_uid, "")
            if ocr_uid
            else ""
        ),
        "ocr_matched_alias": str(
            catalog_match["matched_alias"]
        ),
        "ocr_match_method": ocr_match_method,
        "ocr_match_score": ocr_match_score,
        "ocr_alias_key": str(
            catalog_match["alias_key"]
        ),
        "decision": decision,
        "final_hero_uid": final_hero_uid,
        "final_hero_name": final_hero_name,
        "review_required": review_required,
    }


def build_alias_candidates(
    result_rows: list[dict[str, object]],
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
        if row["decision"] != "NEW_ALIAS_CANDIDATE":
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

    candidate_rows: list[dict[str, object]] = []

    for (
        hero_uid,
        alias_key,
    ), rows in observations.items():
        display_counter = Counter(
            str(row["ocr_text"])
            for row in rows
        )

        display_alias = (
            display_counter.most_common(1)[0][0]
        )

        ocr_confidences = [
            float(row["ocr_confidence"])
            for row in rows
        ]

        visual_similarities = [
            float(row["visual_similarity"])
            for row in rows
        ]

        visual_margins = [
            float(row["visual_margin"])
            for row in rows
        ]

        cross_hero_conflict = (
            len(heroes_by_alias_key[alias_key]) > 1
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
                f"-{row['side']}{row['slot']}"
            )
            for row in rows[:10]
        )

        candidate_rows.append(
            {
                "hero_uid": hero_uid,
                "hero_name": hero_names.get(
                    hero_uid,
                    hero_uid,
                ),
                "alias": display_alias,
                "alias_key": alias_key,
                "occurrences": len(rows),
                "mean_ocr_confidence": (
                    mean(ocr_confidences)
                ),
                "minimum_ocr_confidence": (
                    min(ocr_confidences)
                ),
                "mean_visual_similarity": (
                    mean(visual_similarities)
                ),
                "minimum_visual_similarity": (
                    min(visual_similarities)
                ),
                "mean_visual_margin": (
                    mean(visual_margins)
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


def status_css_class(decision: str) -> str:
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
    }.get(decision, "review")


def create_html_report(
    result_rows: list[dict[str, object]],
    alias_candidates: list[dict[str, object]],
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
        if row["decision"] != "CONFIRMED"
    ]

    weakest_confirmed = sorted(
        (
            row
            for row in result_rows
            if row["decision"] == "CONFIRMED"
        ),
        key=lambda row: (
            float(row["visual_margin"]),
            float(row["visual_similarity"]),
            float(row["ocr_confidence"]),
        ),
    )[:50]

    report_rows = review_rows + weakest_confirmed

    report_rows.sort(
        key=lambda row: (
            priority.get(
                str(row["decision"]),
                99,
            ),
            float(row["ocr_confidence"]),
            float(row["visual_similarity"]),
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
            / str(row["avatar_file"])
        )

        name_path = (
            NAME_DIR
            / str(row["name_file"])
        )

        decision = str(row["decision"])

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

                    Similarité :
                    {float(row["visual_similarity"]):.4f}<br>

                    Marge :
                    {float(row["visual_margin"]):.4f}
                </p>

                <p>
                    <strong>OCR :</strong>
                    <span class="ocr">
                        {html.escape(str(row["ocr_text"]))}
                    </span><br>

                    Confiance :
                    {float(row["ocr_confidence"]):.4f}<br>

                    Correspondance catalogue :
                    {html.escape(str(row["ocr_matched_hero_name"]) or "aucune")}<br>

                    Méthode :
                    {html.escape(str(row["ocr_match_method"]))}
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
                priority.get(item[0], 99),
                item[0],
            ),
        )
    )

    document = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <title>Réconciliation visuelle et OCR</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #eeeeee;
        }}

        .summary,
        .aliases {{
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

        th,
        td {{
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

        .card.confirmed {{
            border-color: #43a047;
        }}

        .card.rescued {{
            border-color: #00897b;
        }}

        .card.alias {{
            border-color: #1e88e5;
        }}

        .card.visual {{
            border-color: #7e57c2;
        }}

        .card.review {{
            border-color: #f9a825;
        }}

        .card.ambiguous {{
            border-color: #ef6c00;
        }}

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
        <h1>Réconciliation visuelle et OCR</h1>

        <p>
            Les cartes contiennent tous les cas non confirmés,
            puis les 50 confirmations les plus fragiles.
        </p>

        <ul>
            {summary_items}
        </ul>

        <p>
            L'outil ne modifie aucun catalogue.
            Les nouveaux alias restent des propositions à valider.
        </p>
    </section>

    <section class="aliases">
        <h2>Alias candidats</h2>

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

            <tbody>
                {''.join(alias_table_rows)}
            </tbody>
        </table>
    </section>

    <main class="grid">
        {''.join(cards)}
    </main>
</body>
</html>
"""

    HTML_REPORT.write_text(
        document,
        encoding="utf-8",
    )


def main() -> int:
    try:
        (
            hero_names,
            alias_to_heroes,
            alias_entries,
        ) = load_catalog()

        visual_rows = read_csv(
            VISUAL_RESULTS
        )

        ocr_rows = read_csv(
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
        f"Résultats visuels : {len(visual_rows)}"
    )
    print(
        f"Résultats OCR : {len(ocr_rows)}"
    )
    print()

    result_rows: list[dict[str, object]] = []
    missing_ocr = 0

    for visual_row in visual_rows:
        key = make_join_key(
            visual_row
        )

        ocr_row = ocr_by_key.get(key)

        if ocr_row is None:
            print(
                "OCR absent pour "
                f"{key[0]} {key[1]}{key[2]}",
                file=sys.stderr,
            )
            missing_ocr += 1
            continue

        result_rows.append(
            reconcile_row(
                visual_row=visual_row,
                ocr_row=ocr_row,
                hero_names=hero_names,
                alias_to_heroes=alias_to_heroes,
                alias_entries=alias_entries,
            )
        )

    if missing_ocr:
        print(
            f"Correspondances OCR manquantes : "
            f"{missing_ocr}",
            file=sys.stderr,
        )
        return 1

    alias_candidates = build_alias_candidates(
        result_rows=result_rows,
        hero_names=hero_names,
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
        alias_candidates=alias_candidates,
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
        f"- Identités finales attribuées : "
        f"{final_assigned}"
    )
    print(
        f"- Cas nécessitant une revue : "
        f"{review_required}"
    )
    print(
        f"- Alias candidats : "
        f"{len(alias_candidates)}"
    )
    print()
    print(f"Résultats : {RESULTS_CSV}")
    print(
        f"Alias candidats : "
        f"{ALIAS_CANDIDATES_CSV}"
    )
    print(f"Contrôle visuel : {HTML_REPORT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
