from __future__ import annotations

import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cluster_hero_avatars import (  # noqa: E402
    AVATAR_DIR,
    OCR_CSV,
    SUPPORTED_EXTENSIONS,
    create_clusters,
    extract_embeddings,
    group_cluster_indices,
)


DISTANCE_THRESHOLD = 0.10

OUTPUT_DIR = Path("data/manifests")
CLUSTERS_OUTPUT = OUTPUT_DIR / "hero_visual_clusters.csv"
AVATARS_OUTPUT = OUTPUT_DIR / "hero_avatar_manifest.csv"


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    value = value.replace("’", "'")
    value = value.replace("`", "'")
    value = " ".join(value.split())

    return value.strip()


def slugify(value: str) -> str:
    """
    Crée un identifiant technique provisoire.

    Exemple :
        Céleste          -> celeste
        Champi et Gnon   -> champi_et_gnon
        K'arkh           -> karkh
    """

    value = normalize_text(value)
    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    value = value.lower()
    value = value.replace("'", "")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")

    return value or "unknown_hero"


def load_ocr_records() -> dict[str, dict[str, str]]:
    if not OCR_CSV.exists():
        raise RuntimeError(f"Fichier OCR absent : {OCR_CSV}")

    records: dict[str, dict[str, str]] = {}

    with OCR_CSV.open(
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file,
            delimiter=";",
        )

        for row in reader:
            filename = row.get("filename", "").strip()

            if not filename:
                continue

            records[filename] = {
                "ocr_text": normalize_text(
                    row.get("ocr_text", "")
                ),
                "ocr_confidence": row.get(
                    "confidence",
                    "",
                ),
                "ocr_status": row.get("status", ""),
                "screenshot_id": row.get(
                    "screenshot_id",
                    "",
                ),
                "side": row.get("side", ""),
                "slot": row.get("slot", ""),
            }

    return records


def main() -> int:
    if not AVATAR_DIR.exists():
        print(
            f"Dossier absent : {AVATAR_DIR}",
            file=sys.stderr,
        )
        return 1

    image_paths = sorted(
        path
        for path in AVATAR_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        print(
            f"Aucun avatar trouvé dans {AVATAR_DIR}",
            file=sys.stderr,
        )
        return 1

    try:
        ocr_records = load_ocr_records()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Appareil utilisé : {device}")

    if device.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")

    print(f"Avatars à traiter : {len(image_paths)}")
    print(
        f"Seuil de regroupement : "
        f"{DISTANCE_THRESHOLD:.2f}"
    )
    print()

    try:
        embeddings, filenames = extract_embeddings(
            image_paths=image_paths,
            device=device,
        )
    except (RuntimeError, OSError, ValueError) as error:
        print(
            f"Échec de l'extraction des signatures : {error}",
            file=sys.stderr,
        )
        return 1

    cluster_labels = create_clusters(
        embeddings=embeddings,
        distance_threshold=DISTANCE_THRESHOLD,
    )

    grouped_clusters = group_cluster_indices(
        cluster_labels
    )

    # Compte les apparences différentes associées au même héros.
    appearance_number_by_hero: dict[str, int] = defaultdict(int)

    cluster_rows: list[dict[str, object]] = []
    avatar_rows: list[dict[str, object]] = []

    for display_number, (
        technical_cluster_id,
        indices,
    ) in enumerate(grouped_clusters, start=1):
        detected_names = [
            ocr_records.get(
                filenames[index],
                {},
            ).get("ocr_text", "")
            for index in indices
        ]

        detected_names = [
            name
            for name in detected_names
            if name
        ]

        name_counts = Counter(detected_names)

        if name_counts:
            suggested_name = name_counts.most_common(1)[0][0]
        else:
            suggested_name = "Unknown"

        hero_id = slugify(suggested_name)

        appearance_number_by_hero[hero_id] += 1

        appearance_id = (
            f"{hero_id}__appearance_"
            f"{appearance_number_by_hero[hero_id]:02d}"
        )

        ocr_aliases = sorted(name_counts)

        sample_files = [
            filenames[index]
            for index in indices[:5]
        ]

        cluster_rows.append(
            {
                "appearance_id": appearance_id,
                "technical_cluster_id": technical_cluster_id,
                "display_cluster_number": display_number,
                "hero_id": hero_id,
                "suggested_name": suggested_name,
                "ocr_aliases": " | ".join(ocr_aliases),
                "avatar_count": len(indices),
                "sample_files": " | ".join(sample_files),
                "reviewed": 0,
                "notes": "",
            }
        )

        for index in indices:
            filename = filenames[index]
            ocr_record = ocr_records.get(filename, {})

            avatar_rows.append(
                {
                    "avatar_file": filename,
                    "appearance_id": appearance_id,
                    "technical_cluster_id": technical_cluster_id,
                    "hero_id": hero_id,
                    "ocr_text": ocr_record.get(
                        "ocr_text",
                        "",
                    ),
                    "ocr_confidence": ocr_record.get(
                        "ocr_confidence",
                        "",
                    ),
                    "ocr_status": ocr_record.get(
                        "ocr_status",
                        "",
                    ),
                    "screenshot_id": ocr_record.get(
                        "screenshot_id",
                        "",
                    ),
                    "side": ocr_record.get("side", ""),
                    "slot": ocr_record.get("slot", ""),
                    "label_source": "visual_cluster_and_ocr",
                    "reviewed": 0,
                }
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CLUSTERS_OUTPUT.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            delimiter=";",
            fieldnames=[
                "appearance_id",
                "technical_cluster_id",
                "display_cluster_number",
                "hero_id",
                "suggested_name",
                "ocr_aliases",
                "avatar_count",
                "sample_files",
                "reviewed",
                "notes",
            ],
        )

        writer.writeheader()
        writer.writerows(cluster_rows)

    with AVATARS_OUTPUT.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            delimiter=";",
            fieldnames=[
                "avatar_file",
                "appearance_id",
                "technical_cluster_id",
                "hero_id",
                "ocr_text",
                "ocr_confidence",
                "ocr_status",
                "screenshot_id",
                "side",
                "slot",
                "label_source",
                "reviewed",
            ],
        )

        writer.writeheader()
        writer.writerows(avatar_rows)

    hero_to_appearances: dict[str, list[str]] = defaultdict(list)

    for row in cluster_rows:
        hero_to_appearances[str(row["hero_id"])].append(
            str(row["appearance_id"])
        )

    multiple_appearances = {
        hero_id: appearance_ids
        for hero_id, appearance_ids
        in hero_to_appearances.items()
        if len(appearance_ids) > 1
    }

    print(f"Apparences visuelles : {len(cluster_rows)}")
    print(f"Héros provisoires : {len(hero_to_appearances)}")
    print(f"Avatars étiquetés : {len(avatar_rows)}")
    print()

    print("Héros possédant plusieurs apparences ou groupes :")

    for hero_id, appearance_ids in sorted(
        multiple_appearances.items()
    ):
        print(
            f"- {hero_id}: "
            f"{', '.join(appearance_ids)}"
        )

    print()
    print(f"Clusters : {CLUSTERS_OUTPUT}")
    print(f"Avatars : {AVATARS_OUTPUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())