from __future__ import annotations

import csv
import html
import os
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import torch

from cluster_hero_avatars import (
    SUPPORTED_EXTENSIONS,
    create_clusters,
    extract_embeddings,
    group_cluster_indices,
)


BATCH_ROOT = Path("data/batches/hero_batch_001")

AVATAR_DIR = BATCH_ROOT / "crops/avatars_inner"
NAME_DIR = BATCH_ROOT / "crops/names"

VISUAL_MATCH_CSV = (
    BATCH_ROOT
    / "reports/visual_matching/visual_match_results.csv"
)

HERO_CATALOG_CSV = Path("data/catalog/heroes.csv")

OUTPUT_DIR = (
    BATCH_ROOT
    / "reports/anchored_clusters"
)

ASSIGNMENTS_CSV = OUTPUT_DIR / "cluster_assignments.csv"
CLUSTERS_CSV = OUTPUT_DIR / "cluster_summary.csv"
HTML_REPORT = OUTPUT_DIR / "cluster_review.html"


# Seuil déjà validé sur les 200 avatars initiaux.
CLUSTER_DISTANCE_THRESHOLD = 0.10

# Un résultat ACCEPTED à 0,90 reste utilisable dans le rapport,
# mais il est trop faible pour devenir un point d’ancrage.
#
# Un ancrage doit être particulièrement fiable afin de ne pas
# propager une mauvaise identité à tout un groupe.
STRONG_ANCHOR_SIMILARITY = 0.94
STRONG_ANCHOR_MARGIN = 0.04


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


def load_hero_names() -> dict[str, str]:
    rows = read_csv(HERO_CATALOG_CSV)

    return {
        row["hero_uid"]: row["reference_name"]
        for row in rows
    }


def load_visual_matches() -> dict[str, dict[str, object]]:
    rows = read_csv(VISUAL_MATCH_CSV)
    matches: dict[str, dict[str, object]] = {}

    for row in rows:
        filename = row["avatar_file"]

        try:
            similarity = float(row["similarity"])
            margin = float(row["margin"])

        except ValueError as error:
            raise RuntimeError(
                f"Valeur numérique invalide pour {filename}: {error}"
            ) from error

        is_strong_anchor = (
            row["status"] == "ACCEPTED"
            and similarity >= STRONG_ANCHOR_SIMILARITY
            and margin >= STRONG_ANCHOR_MARGIN
        )

        matches[filename] = {
            **row,
            "similarity_value": similarity,
            "margin_value": margin,
            "is_strong_anchor": is_strong_anchor,
        }

    return matches


def find_avatar_images() -> list[Path]:
    if not AVATAR_DIR.exists():
        raise RuntimeError(
            f"Dossier absent : {AVATAR_DIR}"
        )

    image_paths = sorted(
        path
        for path in AVATAR_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"Aucun avatar dans {AVATAR_DIR}"
        )

    return image_paths


def relative_image_url(
    image_path: Path,
) -> str:
    relative_path = os.path.relpath(
        image_path,
        OUTPUT_DIR,
    ).replace("\\", "/")

    return quote(relative_path)


def cluster_status_class(status: str) -> str:
    return {
        "ANCHORED": "anchored",
        "UNANCHORED": "unanchored",
        "CONFLICT": "conflict",
    }.get(status, "unanchored")


def original_status_class(status: str) -> str:
    return {
        "ACCEPTED": "accepted",
        "AMBIGUOUS": "ambiguous",
        "REVIEW": "review",
        "UNKNOWN": "unknown",
    }.get(status, "unknown")


def create_html_report(
    cluster_records: list[dict[str, object]],
    assignment_rows: list[dict[str, object]],
) -> None:
    assignments_by_cluster: dict[
        str,
        list[dict[str, object]],
    ] = {}

    for row in assignment_rows:
        cluster_id = str(row["cluster_id"])

        assignments_by_cluster.setdefault(
            cluster_id,
            [],
        ).append(row)

    status_order = {
        "CONFLICT": 0,
        "UNANCHORED": 1,
        "ANCHORED": 2,
    }

    sorted_clusters = sorted(
        cluster_records,
        key=lambda row: (
            status_order.get(
                str(row["cluster_status"]),
                99,
            ),
            -int(row["avatar_count"]),
            str(row["cluster_id"]),
        ),
    )

    cluster_sections: list[str] = []

    for cluster in sorted_clusters:
        cluster_id = str(cluster["cluster_id"])
        cluster_status = str(cluster["cluster_status"])

        assignments = assignments_by_cluster.get(
            cluster_id,
            [],
        )

        # Les points d’ancrage sont affichés en premier.
        assignments = sorted(
            assignments,
            key=lambda row: (
                not bool(row["strong_anchor"]),
                str(row["original_status"]),
                -float(row["similarity"]),
            ),
        )

        cards: list[str] = []

        for row in assignments:
            avatar_filename = str(row["avatar_file"])
            name_filename = str(row["name_file"])

            avatar_path = AVATAR_DIR / avatar_filename
            name_path = NAME_DIR / name_filename

            original_status = str(
                row["original_status"]
            )

            anchor_badge = (
                '<div class="anchor-badge">'
                "ANCRAGE FORT"
                "</div>"
                if bool(row["strong_anchor"])
                else ""
            )

            cards.append(
                f"""
                <article class="avatar-card {
                    original_status_class(original_status)
                }">
                    {anchor_badge}

                    <img
                        class="avatar"
                        src="{relative_image_url(avatar_path)}"
                        alt="{html.escape(avatar_filename)}"
                    >

                    <img
                        class="name-image"
                        src="{relative_image_url(name_path)}"
                        alt="Nom affiché"
                    >

                    <div class="prediction">
                        {html.escape(
                            str(row["original_predicted_name"])
                        )}
                    </div>

                    <div class="metrics">
                        Statut initial :
                        {html.escape(original_status)}<br>

                        Similarité :
                        {float(row["similarity"]):.4f}<br>

                        Marge :
                        {float(row["margin"]):.4f}
                    </div>

                    <div class="filename">
                        {html.escape(avatar_filename)}
                    </div>
                </article>
                """
            )

        cluster_hero_name = str(
            cluster["cluster_hero_name"]
        )

        if not cluster_hero_name:
            cluster_hero_name = "Nouvelle apparence à identifier"

        anchor_details = str(
            cluster["anchor_details"]
        ) or "aucun"

        cluster_sections.append(
            f"""
            <section class="cluster {
                cluster_status_class(cluster_status)
            }">
                <h2>
                    {html.escape(cluster_id)}
                    — {int(cluster["avatar_count"])} avatar(s)
                </h2>

                <div class="cluster-summary">
                    <strong>Statut :</strong>
                    {html.escape(cluster_status)}<br>

                    <strong>Identité proposée :</strong>
                    {html.escape(cluster_hero_name)}<br>

                    <strong>Ancrages forts :</strong>
                    {int(cluster["strong_anchor_count"])}<br>

                    <strong>Détail des ancrages :</strong>
                    {html.escape(anchor_details)}<br>

                    <strong>Cas récupérés :</strong>
                    {int(cluster["rescued_count"])}
                </div>

                <div class="cards">
                    {''.join(cards)}
                </div>
            </section>
            """
        )

    cluster_status_counts = Counter(
        str(row["cluster_status"])
        for row in cluster_records
    )

    avatar_status_counts = Counter(
        str(row["final_status"])
        for row in assignment_rows
    )

    rescued_total = sum(
        int(row["rescued_count"])
        for row in cluster_records
    )

    document = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8">

    <title>
        Regroupement du lot avec ancrages catalogue
    </title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #eeeeee;
        }}

        .global-summary,
        .cluster {{
            background: white;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
            border: 3px solid #cccccc;
        }}

        .cluster.anchored {{
            border-color: #43a047;
        }}

        .cluster.unanchored {{
            border-color: #f9a825;
            background: #fffde7;
        }}

        .cluster.conflict {{
            border-color: #c62828;
            background: #ffebee;
        }}

        .cluster-summary {{
            line-height: 1.6;
            margin-bottom: 14px;
        }}

        .cards {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .avatar-card {{
            position: relative;
            width: 180px;
            padding: 8px;
            border: 2px solid #bbbbbb;
            background: #fafafa;
            text-align: center;
        }}

        .avatar-card.accepted {{
            border-color: #43a047;
        }}

        .avatar-card.review {{
            border-color: #f9a825;
        }}

        .avatar-card.ambiguous {{
            border-color: #ef6c00;
        }}

        .avatar-card.unknown {{
            border-color: #c62828;
        }}

        .avatar {{
            width: 125px;
            height: 125px;
            object-fit: contain;
            background: #222222;
        }}

        .name-image {{
            display: block;
            width: 170px;
            height: 42px;
            object-fit: contain;
            margin: 7px auto;
            background: #222222;
        }}

        .prediction {{
            font-weight: bold;
            font-size: 17px;
            margin: 6px 0;
        }}

        .metrics {{
            font-size: 12px;
            line-height: 1.5;
        }}

        .filename {{
            margin-top: 6px;
            font-size: 9px;
            color: #555555;
            overflow-wrap: anywhere;
        }}

        .anchor-badge {{
            position: absolute;
            top: 5px;
            left: 5px;
            padding: 3px 5px;
            background: #1b5e20;
            color: white;
            font-size: 9px;
            font-weight: bold;
        }}
    </style>
</head>

<body>
    <section class="global-summary">
        <h1>
            Regroupement visuel avec ancrages catalogue
        </h1>

        <p>
            Seuil de regroupement :
            <strong>{CLUSTER_DISTANCE_THRESHOLD:.2f}</strong>
        </p>

        <p>
            Un ancrage fort exige :
            similarité ≥ {STRONG_ANCHOR_SIMILARITY:.2f}
            et marge ≥ {STRONG_ANCHOR_MARGIN:.2f}.
        </p>

        <h3>Groupes</h3>

        <ul>
            <li>
                ANCHORED :
                {cluster_status_counts.get("ANCHORED", 0)}
            </li>

            <li>
                UNANCHORED :
                {cluster_status_counts.get("UNANCHORED", 0)}
            </li>

            <li>
                CONFLICT :
                {cluster_status_counts.get("CONFLICT", 0)}
            </li>
        </ul>

        <h3>Avatars après consensus de groupe</h3>

        <ul>
            <li>
                ASSIGNED :
                {avatar_status_counts.get("ASSIGNED", 0)}
            </li>

            <li>
                UNRESOLVED :
                {avatar_status_counts.get("UNRESOLVED", 0)}
            </li>

            <li>
                CONFLICT :
                {avatar_status_counts.get("CONFLICT", 0)}
            </li>
        </ul>

        <p>
            Avatars REVIEW, UNKNOWN ou AMBIGUOUS
            récupérés grâce au groupe :
            <strong>{rescued_total}</strong>
        </p>

        <p>
            Les images du nom sont affichées uniquement pour
            faciliter le contrôle humain. Elles ne participent
            jamais à la création des groupes.
        </p>
    </section>

    {''.join(cluster_sections)}
</body>
</html>
"""

    HTML_REPORT.write_text(
        document,
        encoding="utf-8",
    )


def main() -> int:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        hero_names = load_hero_names()
        visual_matches = load_visual_matches()
        avatar_paths = find_avatar_images()

    except RuntimeError as error:
        print(
            error,
            file=sys.stderr,
        )
        return 1

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Appareil utilisé : {device}")

    if device.type == "cuda":
        print(
            f"GPU : {torch.cuda.get_device_name(0)}"
        )

    print(f"Avatars à regrouper : {len(avatar_paths)}")
    print(
        f"Seuil de distance : "
        f"{CLUSTER_DISTANCE_THRESHOLD:.2f}"
    )
    print()

    started_at = time.perf_counter()

    try:
        embeddings, filenames = extract_embeddings(
            image_paths=avatar_paths,
            device=device,
        )

    except (
        RuntimeError,
        OSError,
        ValueError,
    ) as error:
        print(
            f"Échec de l'extraction : {error}",
            file=sys.stderr,
        )
        return 1

    cluster_labels = create_clusters(
        embeddings=embeddings,
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
    )

    grouped_clusters = group_cluster_indices(
        cluster_labels
    )

    cluster_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []

    cluster_status_counts: Counter[str] = Counter()
    final_status_counts: Counter[str] = Counter()

    total_rescued = 0

    for display_number, (
        technical_cluster_id,
        indices,
    ) in enumerate(
        grouped_clusters,
        start=1,
    ):
        cluster_id = (
            f"B001_CLUSTER_{display_number:04d}"
        )

        strong_anchor_rows: list[
            dict[str, object]
        ] = []

        for index in indices:
            filename = filenames[index]
            match = visual_matches.get(filename)

            if (
                match is not None
                and bool(match["is_strong_anchor"])
            ):
                strong_anchor_rows.append(match)

        anchor_counts = Counter(
            str(row["predicted_hero_uid"])
            for row in strong_anchor_rows
        )

        if len(anchor_counts) == 0:
            cluster_status = "UNANCHORED"
            cluster_hero_uid = ""
            cluster_hero_name = ""

        elif len(anchor_counts) == 1:
            cluster_status = "ANCHORED"
            cluster_hero_uid = next(
                iter(anchor_counts)
            )

            cluster_hero_name = hero_names.get(
                cluster_hero_uid,
                cluster_hero_uid,
            )

        else:
            cluster_status = "CONFLICT"
            cluster_hero_uid = ""
            cluster_hero_name = ""

        anchor_details = " | ".join(
            (
                f"{hero_names.get(hero_uid, hero_uid)}"
                f": {count}"
            )
            for hero_uid, count in anchor_counts.most_common()
        )

        rescued_count = 0

        for index in indices:
            filename = filenames[index]
            match = visual_matches.get(filename)

            if match is None:
                print(
                    f"Correspondance absente pour {filename}",
                    file=sys.stderr,
                )
                return 1

            original_status = str(match["status"])

            if cluster_status == "ANCHORED":
                final_status = "ASSIGNED"

                if original_status != "ACCEPTED":
                    rescued_count += 1

            elif cluster_status == "CONFLICT":
                final_status = "CONFLICT"

            else:
                final_status = "UNRESOLVED"

            name_path = NAME_DIR / filename

            if not name_path.exists():
                print(
                    f"Zone de nom absente : {name_path}",
                    file=sys.stderr,
                )
                return 1

            assignment_rows.append(
                {
                    "cluster_id": cluster_id,
                    "technical_cluster_id": (
                        technical_cluster_id
                    ),
                    "cluster_status": cluster_status,
                    "cluster_hero_uid": cluster_hero_uid,
                    "cluster_hero_name": cluster_hero_name,
                    "avatar_file": filename,
                    "name_file": filename,
                    "original_predicted_hero_uid": match[
                        "predicted_hero_uid"
                    ],
                    "original_predicted_name": match[
                        "predicted_name"
                    ],
                    "similarity": float(
                        match["similarity_value"]
                    ),
                    "margin": float(
                        match["margin_value"]
                    ),
                    "original_status": original_status,
                    "strong_anchor": bool(
                        match["is_strong_anchor"]
                    ),
                    "final_status": final_status,
                }
            )

            final_status_counts[final_status] += 1

        total_rescued += rescued_count
        cluster_status_counts[cluster_status] += 1

        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "technical_cluster_id": (
                    technical_cluster_id
                ),
                "cluster_status": cluster_status,
                "cluster_hero_uid": cluster_hero_uid,
                "cluster_hero_name": cluster_hero_name,
                "avatar_count": len(indices),
                "strong_anchor_count": len(
                    strong_anchor_rows
                ),
                "anchor_details": anchor_details,
                "rescued_count": rescued_count,
            }
        )

    write_csv(
        CLUSTERS_CSV,
        [
            "cluster_id",
            "technical_cluster_id",
            "cluster_status",
            "cluster_hero_uid",
            "cluster_hero_name",
            "avatar_count",
            "strong_anchor_count",
            "anchor_details",
            "rescued_count",
        ],
        cluster_rows,
    )

    write_csv(
        ASSIGNMENTS_CSV,
        [
            "cluster_id",
            "technical_cluster_id",
            "cluster_status",
            "cluster_hero_uid",
            "cluster_hero_name",
            "avatar_file",
            "name_file",
            "original_predicted_hero_uid",
            "original_predicted_name",
            "similarity",
            "margin",
            "original_status",
            "strong_anchor",
            "final_status",
        ],
        assignment_rows,
    )

    create_html_report(
        cluster_records=cluster_rows,
        assignment_rows=assignment_rows,
    )

    elapsed = time.perf_counter() - started_at

    print("Résumé :")
    print(f"- Groupes créés : {len(cluster_rows)}")
    print(
        f"- Groupes ANCHORED : "
        f"{cluster_status_counts.get('ANCHORED', 0)}"
    )
    print(
        f"- Groupes UNANCHORED : "
        f"{cluster_status_counts.get('UNANCHORED', 0)}"
    )
    print(
        f"- Groupes CONFLICT : "
        f"{cluster_status_counts.get('CONFLICT', 0)}"
    )

    print()
    print("Avatars après consensus :")
    print(
        f"- ASSIGNED : "
        f"{final_status_counts.get('ASSIGNED', 0)}"
    )
    print(
        f"- UNRESOLVED : "
        f"{final_status_counts.get('UNRESOLVED', 0)}"
    )
    print(
        f"- CONFLICT : "
        f"{final_status_counts.get('CONFLICT', 0)}"
    )
    print(
        f"- Cas non acceptés récupérés : "
        f"{total_rescued}"
    )

    print()
    print(f"Durée : {elapsed:.1f} secondes")
    print(f"Clusters : {CLUSTERS_CSV}")
    print(f"Affectations : {ASSIGNMENTS_CSV}")
    print(f"Rapport : {HTML_REPORT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())