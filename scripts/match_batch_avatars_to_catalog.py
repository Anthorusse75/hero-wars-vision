from __future__ import annotations

import csv
import html
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch

from cluster_hero_avatars import (
    SUPPORTED_EXTENSIONS,
    extract_embeddings,
)


REFERENCE_MANIFEST = Path(
    "data/catalog/hero_avatar_manifest.csv"
)

HERO_CATALOG = Path(
    "data/catalog/heroes.csv"
)

REFERENCE_AVATAR_DIR = Path(
    "data/crops/hero/avatars_inner"
)

QUERY_AVATAR_DIR = Path(
    "data/batches/hero_batch_001/crops/avatars_inner"
)

OUTPUT_DIR = Path(
    "data/batches/hero_batch_001/reports/visual_matching"
)

RESULTS_CSV = OUTPUT_DIR / "visual_match_results.csv"
HERO_COUNTS_CSV = OUTPUT_DIR / "predicted_hero_counts.csv"
HTML_REPORT = OUTPUT_DIR / "visual_match_review.html"


# Le seuil de regroupement validé précédemment était une distance
# cosinus de 0,10, soit une similarité de 0,90.
ACCEPT_SIMILARITY = 0.90
ACCEPT_MARGIN = 0.02
REVIEW_SIMILARITY = 0.82

QUERY_PATTERN = re.compile(
    r"^(?P<screenshot_id>\d+).*__"
    r"(?P<side>[LR])(?P<slot>[1-5])$"
)


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
    rows = read_csv(HERO_CATALOG)

    return {
        row["hero_uid"]: row["reference_name"]
        for row in rows
    }


def load_reference_records(
    hero_names: dict[str, str],
) -> list[dict[str, str]]:
    rows = read_csv(REFERENCE_MANIFEST)
    records: list[dict[str, str]] = []

    missing_images: list[str] = []

    for row in rows:
        avatar_file = row["avatar_file"]
        image_path = REFERENCE_AVATAR_DIR / avatar_file

        if not image_path.exists():
            missing_images.append(str(image_path))
            continue

        hero_uid = row["hero_uid"]

        records.append(
            {
                "avatar_file": avatar_file,
                "image_path": str(image_path),
                "hero_uid": hero_uid,
                "hero_name": hero_names.get(
                    hero_uid,
                    hero_uid,
                ),
                "appearance_id": row["appearance_id"],
            }
        )

    if missing_images:
        preview = "\n".join(missing_images[:10])

        raise RuntimeError(
            "Des avatars de référence sont absents :\n"
            f"{preview}"
        )

    if not records:
        raise RuntimeError(
            "Aucun avatar de référence exploitable."
        )

    return records


def find_query_images() -> list[Path]:
    if not QUERY_AVATAR_DIR.exists():
        raise RuntimeError(
            f"Dossier absent : {QUERY_AVATAR_DIR}"
        )

    image_paths = sorted(
        path
        for path in QUERY_AVATAR_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"Aucun avatar dans {QUERY_AVATAR_DIR}"
        )

    return image_paths


def parse_query_metadata(
    filename: str,
) -> tuple[str, str, int]:
    match = QUERY_PATTERN.match(Path(filename).stem)

    if match is None:
        return "", "", 0

    return (
        match.group("screenshot_id"),
        match.group("side"),
        int(match.group("slot")),
    )


def classify_status(
    similarity: float,
    margin: float,
) -> str:
    if (
        similarity >= ACCEPT_SIMILARITY
        and margin >= ACCEPT_MARGIN
    ):
        return "ACCEPTED"

    if similarity >= ACCEPT_SIMILARITY:
        return "AMBIGUOUS"

    if similarity >= REVIEW_SIMILARITY:
        return "REVIEW"

    return "UNKNOWN"


def build_hero_indices(
    reference_records: list[dict[str, str]],
) -> dict[str, list[int]]:
    hero_indices: dict[str, list[int]] = {}

    for index, record in enumerate(reference_records):
        hero_uid = record["hero_uid"]

        hero_indices.setdefault(
            hero_uid,
            [],
        ).append(index)

    return hero_indices


def ranked_hero_candidates(
    similarities: np.ndarray,
    hero_indices: dict[str, list[int]],
    reference_records: list[dict[str, str]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []

    for hero_uid, indices in hero_indices.items():
        hero_similarities = similarities[indices]

        local_best_position = int(
            np.argmax(hero_similarities)
        )

        best_reference_index = indices[
            local_best_position
        ]

        best_similarity = float(
            similarities[best_reference_index]
        )

        reference_record = reference_records[
            best_reference_index
        ]

        candidates.append(
            {
                "hero_uid": hero_uid,
                "hero_name": reference_record[
                    "hero_name"
                ],
                "similarity": best_similarity,
                "reference_index": best_reference_index,
                "reference_file": reference_record[
                    "avatar_file"
                ],
                "appearance_id": reference_record[
                    "appearance_id"
                ],
            }
        )

    return sorted(
        candidates,
        key=lambda candidate: float(
            candidate["similarity"]
        ),
        reverse=True,
    )


def calculate_reference_calibration(
    reference_embeddings: np.ndarray,
    reference_records: list[dict[str, str]],
    hero_indices: dict[str, list[int]],
) -> dict[str, object]:
    """
    Teste chaque avatar de référence contre tous les autres.

    Les héros qui ne possèdent qu’un seul avatar de référence sont
    exclus du calcul, car il n’existe aucun autre exemple de la même
    classe avec lequel les comparer.
    """

    similarities = (
        reference_embeddings
        @ reference_embeddings.T
    )

    np.fill_diagonal(
        similarities,
        -1.0,
    )

    evaluated = 0
    correct = 0

    same_class_similarities: list[float] = []
    other_class_similarities: list[float] = []
    margins: list[float] = []

    for reference_index, record in enumerate(
        reference_records
    ):
        expected_hero = record["hero_uid"]
        same_class_indices = [
            index
            for index in hero_indices[expected_hero]
            if index != reference_index
        ]

        if not same_class_indices:
            continue

        evaluated += 1

        same_best = float(
            np.max(
                similarities[
                    reference_index,
                    same_class_indices,
                ]
            )
        )

        other_indices = [
            index
            for index, other_record
            in enumerate(reference_records)
            if other_record["hero_uid"] != expected_hero
        ]

        other_best = float(
            np.max(
                similarities[
                    reference_index,
                    other_indices,
                ]
            )
        )

        same_class_similarities.append(same_best)
        other_class_similarities.append(other_best)
        margins.append(same_best - other_best)

        candidates = ranked_hero_candidates(
            similarities=similarities[
                reference_index
            ],
            hero_indices=hero_indices,
            reference_records=reference_records,
        )

        if candidates[0]["hero_uid"] == expected_hero:
            correct += 1

    def distribution(
        values: list[float],
    ) -> dict[str, float]:
        if not values:
            return {
                "minimum": 0.0,
                "p05": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "maximum": 0.0,
            }

        array = np.asarray(
            values,
            dtype=np.float32,
        )

        return {
            "minimum": float(np.min(array)),
            "p05": float(np.percentile(array, 5)),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "maximum": float(np.max(array)),
        }

    return {
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": (
            correct / evaluated
            if evaluated
            else 0.0
        ),
        "same_similarity": distribution(
            same_class_similarities
        ),
        "other_similarity": distribution(
            other_class_similarities
        ),
        "margin": distribution(margins),
    }


def relative_image_url(
    image_path: Path,
) -> str:
    relative_path = os.path.relpath(
        image_path,
        OUTPUT_DIR,
    ).replace("\\", "/")

    return quote(relative_path)


def create_html_report(
    result_rows: list[dict[str, object]],
    reference_records_by_file: dict[str, dict[str, str]],
) -> int:
    non_accepted = [
        row
        for row in result_rows
        if row["status"] != "ACCEPTED"
    ]

    accepted_audit = sorted(
        (
            row
            for row in result_rows
            if row["status"] == "ACCEPTED"
        ),
        key=lambda row: (
            float(row["margin"]),
            float(row["similarity"]),
        ),
    )[:50]

    report_rows = non_accepted + accepted_audit

    status_order = {
        "UNKNOWN": 0,
        "AMBIGUOUS": 1,
        "REVIEW": 2,
        "ACCEPTED": 3,
    }

    report_rows.sort(
        key=lambda row: (
            status_order.get(
                str(row["status"]),
                99,
            ),
            float(row["similarity"]),
            float(row["margin"]),
        )
    )

    cards: list[str] = []

    for row in report_rows:
        query_path = (
            QUERY_AVATAR_DIR
            / str(row["avatar_file"])
        )

        reference_file = str(
            row["best_reference_file"]
        )

        reference_record = (
            reference_records_by_file[
                reference_file
            ]
        )

        reference_path = Path(
            reference_record["image_path"]
        )

        status = str(row["status"])

        status_class = {
            "ACCEPTED": "accepted",
            "AMBIGUOUS": "ambiguous",
            "REVIEW": "review",
            "UNKNOWN": "unknown",
        }.get(status, "unknown")

        cards.append(
            f"""
            <article class="card {status_class}">
                <div class="images">
                    <div>
                        <div class="caption">Avatar analysé</div>
                        <img
                            src="{relative_image_url(query_path)}"
                            alt="{html.escape(str(row["avatar_file"]))}"
                        >
                    </div>

                    <div class="arrow">→</div>

                    <div>
                        <div class="caption">Référence la plus proche</div>
                        <img
                            src="{relative_image_url(reference_path)}"
                            alt="{html.escape(reference_file)}"
                        >
                    </div>
                </div>

                <h2>
                    {html.escape(str(row["predicted_name"]))}
                </h2>

                <p>
                    <strong>Statut :</strong>
                    {html.escape(status)}<br>

                    <strong>Similarité :</strong>
                    {float(row["similarity"]):.4f}<br>

                    <strong>Marge :</strong>
                    {float(row["margin"]):.4f}<br>

                    <strong>Apparence :</strong>
                    {html.escape(str(row["appearance_id"]))}<br>

                    <strong>Capture :</strong>
                    {html.escape(str(row["screenshot_id"]))}
                    — {html.escape(str(row["side"]))}
                    {html.escape(str(row["slot"]))}
                </p>

                <p class="alternatives">
                    <strong>Top 3 :</strong><br>
                    1. {html.escape(str(row["predicted_name"]))}
                    ({float(row["similarity"]):.4f})<br>

                    2. {html.escape(str(row["second_name"]))}
                    ({float(row["second_similarity"]):.4f})<br>

                    3. {html.escape(str(row["third_name"]))}
                    ({float(row["third_similarity"]):.4f})
                </p>

                <p class="filename">
                    {html.escape(str(row["avatar_file"]))}
                </p>
            </article>
            """
        )

    document = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8">

    <title>Contrôle des correspondances visuelles</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #eeeeee;
        }}

        .summary {{
            background: white;
            border: 1px solid #cccccc;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }}

        .grid {{
            display: grid;
            grid-template-columns:
                repeat(auto-fill, minmax(340px, 1fr));
            gap: 16px;
        }}

        .card {{
            background: white;
            border: 4px solid #cccccc;
            border-radius: 8px;
            padding: 12px;
        }}

        .card.accepted {{
            border-color: #43a047;
        }}

        .card.review {{
            border-color: #f9a825;
        }}

        .card.ambiguous {{
            border-color: #ef6c00;
        }}

        .card.unknown {{
            border-color: #c62828;
            background: #ffebee;
        }}

        .images {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }}

        .images img {{
            width: 125px;
            height: 125px;
            object-fit: contain;
            background: #222222;
        }}

        .caption {{
            font-size: 11px;
            text-align: center;
            margin-bottom: 4px;
        }}

        .arrow {{
            font-size: 28px;
            font-weight: bold;
        }}

        h2 {{
            margin-bottom: 6px;
        }}

        p {{
            line-height: 1.45;
        }}

        .alternatives {{
            background: #f5f5f5;
            padding: 8px;
        }}

        .filename {{
            color: #555555;
            font-size: 10px;
            overflow-wrap: anywhere;
        }}
    </style>
</head>

<body>
    <section class="summary">
        <h1>Contrôle des correspondances visuelles</h1>

        <p>
            Le rapport contient tous les résultats non acceptés,
            puis les 50 correspondances acceptées les plus fragiles.
        </p>

        <p>
            Seuil accepté :
            similarité ≥ {ACCEPT_SIMILARITY:.2f}
            et marge ≥ {ACCEPT_MARGIN:.2f}.<br>

            Seuil de vérification :
            similarité ≥ {REVIEW_SIMILARITY:.2f}.
        </p>

        <p>
            Cartes affichées : {len(report_rows)}
        </p>
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

    return len(report_rows)


def main() -> int:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        hero_names = load_hero_names()

        reference_records = load_reference_records(
            hero_names=hero_names,
        )

        query_paths = find_query_images()

    except RuntimeError as error:
        print(
            error,
            file=sys.stderr,
        )
        return 1

    reference_paths = [
        Path(record["image_path"])
        for record in reference_records
    ]

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

    print(
        f"Avatars de référence : "
        f"{len(reference_paths)}"
    )

    print(
        f"Avatars à identifier : "
        f"{len(query_paths)}"
    )

    print()
    print("Création des signatures visuelles...")

    started_at = time.perf_counter()

    try:
        reference_embeddings, reference_filenames = (
            extract_embeddings(
                image_paths=reference_paths,
                device=device,
            )
        )

        query_embeddings, query_filenames = (
            extract_embeddings(
                image_paths=query_paths,
                device=device,
            )
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

    reference_by_original_filename = {
        record["avatar_file"]: record
        for record in reference_records
    }

    reference_records_ordered = [
        reference_by_original_filename[filename]
        for filename in reference_filenames
    ]

    hero_indices = build_hero_indices(
        reference_records_ordered
    )

    calibration = calculate_reference_calibration(
        reference_embeddings=reference_embeddings,
        reference_records=reference_records_ordered,
        hero_indices=hero_indices,
    )

    similarities_matrix = (
        query_embeddings
        @ reference_embeddings.T
    )

    result_rows: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    hero_counts: Counter[str] = Counter()

    for query_index, query_filename in enumerate(
        query_filenames
    ):
        candidates = ranked_hero_candidates(
            similarities=similarities_matrix[
                query_index
            ],
            hero_indices=hero_indices,
            reference_records=reference_records_ordered,
        )

        best = candidates[0]
        second = candidates[1]
        third = candidates[2]

        similarity = float(best["similarity"])
        second_similarity = float(
            second["similarity"]
        )

        margin = similarity - second_similarity

        status = classify_status(
            similarity=similarity,
            margin=margin,
        )

        screenshot_id, side, slot = (
            parse_query_metadata(
                query_filename
            )
        )

        result_rows.append(
            {
                "screenshot_id": screenshot_id,
                "side": side,
                "slot": slot,
                "avatar_file": query_filename,
                "predicted_hero_uid": best["hero_uid"],
                "predicted_name": best["hero_name"],
                "appearance_id": best["appearance_id"],
                "best_reference_file": best[
                    "reference_file"
                ],
                "similarity": f"{similarity:.6f}",
                "second_hero_uid": second["hero_uid"],
                "second_name": second["hero_name"],
                "second_similarity": (
                    f"{second_similarity:.6f}"
                ),
                "third_hero_uid": third["hero_uid"],
                "third_name": third["hero_name"],
                "third_similarity": (
                    f"{float(third['similarity']):.6f}"
                ),
                "margin": f"{margin:.6f}",
                "status": status,
            }
        )

        status_counts[status] += 1
        hero_counts[str(best["hero_uid"])] += 1

    write_csv(
        RESULTS_CSV,
        [
            "screenshot_id",
            "side",
            "slot",
            "avatar_file",
            "predicted_hero_uid",
            "predicted_name",
            "appearance_id",
            "best_reference_file",
            "similarity",
            "second_hero_uid",
            "second_name",
            "second_similarity",
            "third_hero_uid",
            "third_name",
            "third_similarity",
            "margin",
            "status",
        ],
        result_rows,
    )

    hero_count_rows = [
        {
            "hero_uid": hero_uid,
            "hero_name": hero_names.get(
                hero_uid,
                hero_uid,
            ),
            "prediction_count": count,
        }
        for hero_uid, count in sorted(
            hero_counts.items(),
            key=lambda item: (
                -item[1],
                hero_names.get(
                    item[0],
                    item[0],
                ),
            ),
        )
    ]

    write_csv(
        HERO_COUNTS_CSV,
        [
            "hero_uid",
            "hero_name",
            "prediction_count",
        ],
        hero_count_rows,
    )

    reference_records_by_file = {
        record["avatar_file"]: record
        for record in reference_records_ordered
    }

    report_count = create_html_report(
        result_rows=result_rows,
        reference_records_by_file=(
            reference_records_by_file
        ),
    )

    elapsed = time.perf_counter() - started_at

    same_distribution = calibration[
        "same_similarity"
    ]

    other_distribution = calibration[
        "other_similarity"
    ]

    margin_distribution = calibration["margin"]

    print()
    print("Calibration sur le catalogue :")
    print(
        f"- Références évaluables : "
        f"{calibration['evaluated']}"
    )
    print(
        f"- Exactitude leave-one-out : "
        f"{calibration['accuracy'] * 100:.2f} %"
    )

    print(
        "- Similarité même héros : "
        f"min={same_distribution['minimum']:.4f}, "
        f"p05={same_distribution['p05']:.4f}, "
        f"médiane={same_distribution['median']:.4f}"
    )

    print(
        "- Meilleure similarité mauvais héros : "
        f"médiane={other_distribution['median']:.4f}, "
        f"p95={other_distribution['p95']:.4f}, "
        f"max={other_distribution['maximum']:.4f}"
    )

    print(
        "- Marge même héros / mauvais héros : "
        f"min={margin_distribution['minimum']:.4f}, "
        f"p05={margin_distribution['p05']:.4f}, "
        f"médiane={margin_distribution['median']:.4f}"
    )

    print()
    print("Résultats du lot :")

    for status in (
        "ACCEPTED",
        "AMBIGUOUS",
        "REVIEW",
        "UNKNOWN",
    ):
        print(
            f"- {status:<10} : "
            f"{status_counts.get(status, 0)}"
        )

    print()
    print(f"Héros prédits différents : {len(hero_counts)}")
    print(f"Cartes dans le rapport : {report_count}")
    print(f"Durée totale : {elapsed:.1f} secondes")
    print()

    print(f"Résultats : {RESULTS_CSV}")
    print(f"Répartition : {HERO_COUNTS_CSV}")
    print(f"Contrôle visuel : {HTML_REPORT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())