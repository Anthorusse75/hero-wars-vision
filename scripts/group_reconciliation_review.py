from __future__ import annotations

import csv
import html
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote


BATCH_ROOT = Path("data/batches/hero_batch_001")

RECONCILIATION_CSV = (
    BATCH_ROOT
    / "reports"
    / "reconciliation_v1"
    / "reconciliation_results.csv"
)

ALIAS_CANDIDATES_CSV = (
    BATCH_ROOT
    / "reports"
    / "reconciliation_v1"
    / "alias_candidates.csv"
)

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
    / "reconciliation_groups_v1"
)

GROUPS_CSV = OUTPUT_DIR / "review_groups.csv"
MEMBERS_CSV = OUTPUT_DIR / "review_group_members.csv"
HTML_REPORT = OUTPUT_DIR / "review_groups.html"


REVIEW_DECISIONS = {
    "CONFLICT",
    "NEW_ALIAS_CANDIDATE",
    "AGREEMENT_REVIEW",
    "UNRESOLVED_OCR_TEXT",
    "MANUAL_REVIEW",
    "AMBIGUOUS_ALIAS",
}

DECISION_PRIORITY = {
    "CONFLICT": 0,
    "AMBIGUOUS_ALIAS": 1,
    "NEW_ALIAS_CANDIDATE": 2,
    "AGREEMENT_REVIEW": 3,
    "UNRESOLVED_OCR_TEXT": 4,
    "MANUAL_REVIEW": 5,
}

MAX_SAMPLES_PER_GROUP = 20


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
    value = unicodedata.normalize("NFC", value)
    value = value.replace("’", "'")
    value = value.replace("`", "'")
    value = " ".join(value.split())

    return value.strip(" -")


def normalize_key(value: str) -> str:
    value = normalize_display_text(value)
    value = unicodedata.normalize("NFKD", value)

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


def sanitize_key(value: str) -> str:
    value = normalize_key(value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")

    return value or "none"


def relative_image_url(path: Path) -> str:
    relative_path = os.path.relpath(
        path,
        OUTPUT_DIR,
    ).replace("\\", "/")

    return quote(relative_path)


def make_group_key(
    row: dict[str, str],
) -> tuple[str, str]:
    decision = row["decision"]

    visual_uid = row["visual_hero_uid"].strip()
    ocr_uid = row["ocr_matched_hero_uid"].strip()
    ocr_key = normalize_key(row["ocr_text"])

    if decision == "CONFLICT":
        return (
            decision,
            f"{visual_uid}__{ocr_uid}",
        )

    if decision == "AMBIGUOUS_ALIAS":
        return (
            decision,
            ocr_key or "empty",
        )

    if decision == "NEW_ALIAS_CANDIDATE":
        return (
            decision,
            f"{visual_uid}__{ocr_key or 'empty'}",
        )

    if decision == "AGREEMENT_REVIEW":
        return (
            decision,
            visual_uid or "unknown_visual",
        )

    if decision == "UNRESOLVED_OCR_TEXT":
        return (
            decision,
            ocr_key or "empty",
        )

    if decision == "MANUAL_REVIEW":
        return (
            decision,
            (
                f"{visual_uid or 'unknown_visual'}"
                f"__{ocr_key or 'empty'}"
            ),
        )

    return (
        decision,
        (
            f"{visual_uid or 'unknown_visual'}"
            f"__{ocr_key or 'empty'}"
        ),
    )


def representative_text(
    rows: list[dict[str, str]],
    field: str,
) -> str:
    values = [
        normalize_display_text(row[field])
        for row in rows
        if normalize_display_text(row[field])
    ]

    if not values:
        return ""

    return Counter(values).most_common(1)[0][0]


def counter_text(
    values: list[str],
) -> str:
    counts = Counter(
        value
        for value in values
        if value
    )

    return " | ".join(
        f"{value}: {count}"
        for value, count in counts.most_common()
    )


def build_groups(
    rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, list[dict[str, str]]],
]:
    grouped: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        if row["decision"] not in REVIEW_DECISIONS:
            continue

        grouped[make_group_key(row)].append(row)

    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: (
            DECISION_PRIORITY.get(item[0][0], 99),
            -len(item[1]),
            item[0][1],
        ),
    )

    group_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    members_by_group_id: dict[
        str,
        list[dict[str, str]],
    ] = {}

    for index, (
        (decision, technical_key),
        members,
    ) in enumerate(
        sorted_groups,
        start=1,
    ):
        group_id = f"REVIEW_GROUP_{index:03d}"

        visual_names = [
            row["visual_hero_name"]
            for row in members
        ]

        visual_statuses = [
            row["visual_status"]
            for row in members
        ]

        ocr_texts = [
            normalize_display_text(
                row["ocr_text"]
            )
            for row in members
        ]

        ocr_matches = [
            row["ocr_matched_hero_name"]
            for row in members
        ]

        representative_ocr = representative_text(
            members,
            "ocr_text",
        )

        group_rows.append(
            {
                "group_id": group_id,
                "decision": decision,
                "technical_key": technical_key,
                "member_count": len(members),
                "representative_ocr_text": representative_ocr,
                "visual_predictions": counter_text(
                    visual_names
                ),
                "visual_statuses": counter_text(
                    visual_statuses
                ),
                "ocr_catalog_matches": counter_text(
                    ocr_matches
                ),
                "ocr_variants": counter_text(
                    ocr_texts
                ),
                "minimum_visual_similarity": min(
                    float(row["visual_similarity"])
                    for row in members
                ),
                "maximum_visual_similarity": max(
                    float(row["visual_similarity"])
                    for row in members
                ),
                "minimum_visual_margin": min(
                    float(row["visual_margin"])
                    for row in members
                ),
                "maximum_visual_margin": max(
                    float(row["visual_margin"])
                    for row in members
                ),
                "minimum_ocr_confidence": min(
                    float(row["ocr_confidence"])
                    for row in members
                ),
                "maximum_ocr_confidence": max(
                    float(row["ocr_confidence"])
                    for row in members
                ),
                "review_action": "",
                "validated_hero_uid": "",
                "validated_hero_name": "",
                "validated_alias": "",
                "notes": "",
            }
        )

        members_by_group_id[group_id] = members

        for row in members:
            member_rows.append(
                {
                    "group_id": group_id,
                    "decision": decision,
                    "screenshot_id": row[
                        "screenshot_id"
                    ],
                    "side": row["side"],
                    "slot": row["slot"],
                    "avatar_file": row[
                        "avatar_file"
                    ],
                    "name_file": row["name_file"],
                    "visual_hero_uid": row[
                        "visual_hero_uid"
                    ],
                    "visual_hero_name": row[
                        "visual_hero_name"
                    ],
                    "visual_status": row[
                        "visual_status"
                    ],
                    "visual_similarity": row[
                        "visual_similarity"
                    ],
                    "visual_margin": row[
                        "visual_margin"
                    ],
                    "ocr_text": row["ocr_text"],
                    "ocr_confidence": row[
                        "ocr_confidence"
                    ],
                    "ocr_status": row["ocr_status"],
                    "ocr_matched_hero_uid": row[
                        "ocr_matched_hero_uid"
                    ],
                    "ocr_matched_hero_name": row[
                        "ocr_matched_hero_name"
                    ],
                    "ocr_match_method": row[
                        "ocr_match_method"
                    ],
                }
            )

    return (
        group_rows,
        member_rows,
        members_by_group_id,
    )


def decision_css_class(decision: str) -> str:
    return {
        "CONFLICT": "conflict",
        "AMBIGUOUS_ALIAS": "ambiguous",
        "NEW_ALIAS_CANDIDATE": "alias",
        "AGREEMENT_REVIEW": "agreement",
        "UNRESOLVED_OCR_TEXT": "unresolved",
        "MANUAL_REVIEW": "manual",
    }.get(decision, "manual")


def create_html_report(
    group_rows: list[dict[str, object]],
    members_by_group_id: dict[
        str,
        list[dict[str, str]],
    ],
    alias_candidates: list[dict[str, str]],
) -> None:
    decision_counts = Counter(
        str(row["decision"])
        for row in group_rows
    )

    sections: list[str] = []

    for group in group_rows:
        group_id = str(group["group_id"])
        decision = str(group["decision"])

        members = members_by_group_id[
            group_id
        ]

        cards: list[str] = []

        for member in members[
            :MAX_SAMPLES_PER_GROUP
        ]:
            avatar_path = (
                AVATAR_DIR
                / member["avatar_file"]
            )

            name_path = (
                NAME_DIR
                / member["name_file"]
            )

            cards.append(
                f"""
                <article class="card">
                    <img
                        class="avatar"
                        src="{relative_image_url(avatar_path)}"
                        alt="{html.escape(member["avatar_file"])}"
                    >

                    <img
                        class="name"
                        src="{relative_image_url(name_path)}"
                        alt="{html.escape(member["name_file"])}"
                    >

                    <p>
                        <strong>Visuel :</strong>
                        {html.escape(member["visual_hero_name"])}
                        — {html.escape(member["visual_status"])}<br>

                        Similarité :
                        {float(member["visual_similarity"]):.4f}<br>

                        Marge :
                        {float(member["visual_margin"]):.4f}
                    </p>

                    <p>
                        <strong>OCR :</strong>
                        <span class="ocr">
                            {html.escape(member["ocr_text"])}
                        </span><br>

                        Confiance :
                        {float(member["ocr_confidence"]):.4f}<br>

                        Catalogue OCR :
                        {html.escape(
                            member["ocr_matched_hero_name"]
                            or "aucun"
                        )}
                    </p>

                    <p class="filename">
                        {html.escape(member["screenshot_id"])}
                        — {html.escape(member["side"])}
                        {html.escape(member["slot"])}
                    </p>
                </article>
                """
            )

        hidden_count = max(
            0,
            len(members) - MAX_SAMPLES_PER_GROUP,
        )

        hidden_text = (
            f"<p>{hidden_count} autre(s) membre(s) "
            "sont présents dans le CSV.</p>"
            if hidden_count
            else ""
        )

        sections.append(
            f"""
            <section class="group {
                decision_css_class(decision)
            }">
                <h2>
                    {html.escape(group_id)}
                    — {html.escape(decision)}
                    — {int(group["member_count"])} cas
                </h2>

                <div class="summary">
                    <strong>OCR représentatif :</strong>
                    {html.escape(
                        str(group["representative_ocr_text"])
                        or "(vide)"
                    )}<br>

                    <strong>Prédictions visuelles :</strong>
                    {html.escape(
                        str(group["visual_predictions"])
                        or "aucune"
                    )}<br>

                    <strong>Statuts visuels :</strong>
                    {html.escape(
                        str(group["visual_statuses"])
                    )}<br>

                    <strong>Correspondances OCR catalogue :</strong>
                    {html.escape(
                        str(group["ocr_catalog_matches"])
                        or "aucune"
                    )}<br>

                    <strong>Variantes OCR :</strong>
                    {html.escape(
                        str(group["ocr_variants"])
                        or "aucune"
                    )}
                </div>

                {hidden_text}

                <div class="cards">
                    {''.join(cards)}
                </div>
            </section>
            """
        )

    alias_rows: list[str] = []

    for row in alias_candidates:
        alias_rows.append(
            f"""
            <tr>
                <td>{html.escape(row["hero_name"])}</td>
                <td class="ocr">{html.escape(row["alias"])}</td>
                <td>{html.escape(row["occurrences"])}</td>
                <td>{html.escape(row["candidate_status"])}</td>
                <td>{html.escape(row["examples"])}</td>
            </tr>
            """
        )

    decision_summary = "".join(
        f"<li>{html.escape(decision)} : {count}</li>"
        for decision, count in sorted(
            decision_counts.items(),
            key=lambda item: (
                DECISION_PRIORITY.get(
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
    <title>Groupes de revue de la réconciliation</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #eeeeee;
        }}

        .global,
        .aliases,
        .group {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            border: 4px solid #bdbdbd;
        }}

        .group.conflict {{
            border-color: #c62828;
            background: #ffebee;
        }}

        .group.ambiguous {{
            border-color: #ef6c00;
        }}

        .group.alias {{
            border-color: #1e88e5;
        }}

        .group.agreement {{
            border-color: #00897b;
        }}

        .group.unresolved {{
            border-color: #f9a825;
            background: #fffde7;
        }}

        .group.manual {{
            border-color: #7e57c2;
        }}

        .summary {{
            line-height: 1.6;
            margin-bottom: 12px;
        }}

        .cards {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .card {{
            width: 210px;
            border: 1px solid #cccccc;
            padding: 8px;
            background: #fafafa;
            text-align: center;
        }}

        .avatar {{
            width: 135px;
            height: 135px;
            object-fit: contain;
            background: #222222;
        }}

        .name {{
            display: block;
            width: 200px;
            height: 55px;
            object-fit: contain;
            background: #222222;
            margin: 7px auto;
        }}

        .ocr {{
            font-size: 18px;
            font-weight: bold;
        }}

        .filename {{
            font-size: 11px;
            color: #555555;
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
    </style>
</head>

<body>
    <section class="global">
        <h1>Groupes de revue de la réconciliation</h1>

        <p>
            Les 108 cas à contrôler ont été regroupés par
            type de problème et par texte OCR ou identité.
        </p>

        <ul>
            {decision_summary}
        </ul>

        <p>
            Groupes créés : {len(group_rows)}
        </p>
    </section>

    <section class="aliases">
        <h2>Alias candidats déjà regroupés</h2>

        <table>
            <thead>
                <tr>
                    <th>Héros</th>
                    <th>Alias</th>
                    <th>Occurrences</th>
                    <th>Statut</th>
                    <th>Exemples</th>
                </tr>
            </thead>

            <tbody>
                {''.join(alias_rows)}
            </tbody>
        </table>
    </section>

    {''.join(sections)}
</body>
</html>
"""

    HTML_REPORT.write_text(
        document,
        encoding="utf-8",
    )


def main() -> int:
    try:
        reconciliation_rows = read_csv(
            RECONCILIATION_CSV
        )

        alias_candidates = read_csv(
            ALIAS_CANDIDATES_CSV
        )

    except RuntimeError as error:
        print(
            error,
            file=sys.stderr,
        )
        return 1

    (
        group_rows,
        member_rows,
        members_by_group_id,
    ) = build_groups(
        reconciliation_rows
    )

    if not group_rows:
        print(
            "Aucun cas à regrouper.",
            file=sys.stderr,
        )
        return 1

    group_fieldnames = [
        "group_id",
        "decision",
        "technical_key",
        "member_count",
        "representative_ocr_text",
        "visual_predictions",
        "visual_statuses",
        "ocr_catalog_matches",
        "ocr_variants",
        "minimum_visual_similarity",
        "maximum_visual_similarity",
        "minimum_visual_margin",
        "maximum_visual_margin",
        "minimum_ocr_confidence",
        "maximum_ocr_confidence",
        "review_action",
        "validated_hero_uid",
        "validated_hero_name",
        "validated_alias",
        "notes",
    ]

    member_fieldnames = [
        "group_id",
        "decision",
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
        "ocr_match_method",
    ]

    write_csv(
        GROUPS_CSV,
        group_fieldnames,
        group_rows,
    )

    write_csv(
        MEMBERS_CSV,
        member_fieldnames,
        member_rows,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    create_html_report(
        group_rows=group_rows,
        members_by_group_id=(
            members_by_group_id
        ),
        alias_candidates=alias_candidates,
    )

    decision_counts = Counter(
        str(row["decision"])
        for row in group_rows
    )

    print("Résumé :")
    print(
        f"- Cas à revoir : "
        f"{len(member_rows)}"
    )
    print(
        f"- Groupes créés : "
        f"{len(group_rows)}"
    )
    print()

    for decision in (
        "CONFLICT",
        "AMBIGUOUS_ALIAS",
        "NEW_ALIAS_CANDIDATE",
        "AGREEMENT_REVIEW",
        "UNRESOLVED_OCR_TEXT",
        "MANUAL_REVIEW",
    ):
        print(
            f"- Groupes {decision:<22} : "
            f"{decision_counts.get(decision, 0)}"
        )

    print()
    print(f"Groupes : {GROUPS_CSV}")
    print(f"Membres : {MEMBERS_CSV}")
    print(f"Rapport : {HTML_REPORT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
