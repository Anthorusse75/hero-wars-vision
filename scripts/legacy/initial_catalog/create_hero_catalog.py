from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


CLUSTERS_INPUT = Path("data/manifests/hero_visual_clusters.csv")
AVATARS_INPUT = Path("data/manifests/hero_avatar_manifest.csv")

OUTPUT_DIR = Path("data/catalog")
CATALOG_OUTPUT = OUTPUT_DIR / "heroes.csv"
APPEARANCES_OUTPUT = OUTPUT_DIR / "hero_appearances.csv"
ALIASES_OUTPUT = OUTPUT_DIR / "hero_name_aliases.csv"
AVATARS_OUTPUT = OUTPUT_DIR / "hero_avatar_manifest.csv"


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


def main() -> int:
    try:
        cluster_rows = read_csv(CLUSTERS_INPUT)
        avatar_rows = read_csv(AVATARS_INPUT)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Le catalogue ne doit être créé qu’une seule fois afin que les
    # identifiants internes ne changent jamais.
    if CATALOG_OUTPUT.exists():
        print(
            f"Le catalogue existe déjà : {CATALOG_OUTPUT}",
            file=sys.stderr,
        )
        print(
            "Aucun fichier n’a été modifié.",
            file=sys.stderr,
        )
        return 1

    provisional_heroes: dict[str, dict[str, str]] = {}

    for row in cluster_rows:
        provisional_id = row["hero_id"]

        if provisional_id not in provisional_heroes:
            provisional_heroes[provisional_id] = {
                "reference_name": row["suggested_name"],
            }

    internal_ids: dict[str, str] = {}

    for index, provisional_id in enumerate(
        sorted(provisional_heroes),
        start=1,
    ):
        internal_ids[provisional_id] = f"HW_HERO_{index:04d}"

    catalog_output_rows: list[dict[str, object]] = []

    for provisional_id in sorted(provisional_heroes):
        hero_uid = internal_ids[provisional_id]

        catalog_output_rows.append(
            {
                "hero_uid": hero_uid,
                "reference_name": provisional_heroes[
                    provisional_id
                ]["reference_name"],
                "provisional_key": provisional_id,
                "reviewed": 0,
                "notes": "",
            }
        )

    appearances_output_rows: list[dict[str, object]] = []

    for row in cluster_rows:
        appearances_output_rows.append(
            {
                "appearance_id": row["appearance_id"],
                "hero_uid": internal_ids[row["hero_id"]],
                "technical_cluster_id": row[
                    "technical_cluster_id"
                ],
                "avatar_count": row["avatar_count"],
                "sample_files": row["sample_files"],
                "appearance_type": (
                    "skin_or_variant"
                    if row["hero_id"] == "dorian"
                    else "unknown_variant"
                ),
                "reviewed": row["reviewed"],
                "notes": row["notes"],
            }
        )

    aliases_by_key: dict[tuple[str, str], dict[str, object]] = {}

    for row in avatar_rows:
        alias = row["ocr_text"].strip()

        if not alias:
            continue

        hero_uid = internal_ids[row["hero_id"]]
        key = (hero_uid, alias)

        if key not in aliases_by_key:
            aliases_by_key[key] = {
                "hero_uid": hero_uid,
                "alias": alias,
                "language": "unknown",
                "source": "easyocr",
                "occurrences": 0,
                "reviewed": 0,
            }

        aliases_by_key[key]["occurrences"] = (
            int(aliases_by_key[key]["occurrences"]) + 1
        )

    aliases_output_rows = sorted(
        aliases_by_key.values(),
        key=lambda row: (
            str(row["hero_uid"]),
            str(row["alias"]),
        ),
    )

    updated_avatar_rows: list[dict[str, object]] = []

    for row in avatar_rows:
        updated_avatar_rows.append(
            {
                "avatar_file": row["avatar_file"],
                "hero_uid": internal_ids[row["hero_id"]],
                "appearance_id": row["appearance_id"],
                "ocr_text": row["ocr_text"],
                "ocr_confidence": row["ocr_confidence"],
                "ocr_status": row["ocr_status"],
                "screenshot_id": row["screenshot_id"],
                "side": row["side"],
                "slot": row["slot"],
                "label_source": row["label_source"],
                "reviewed": row["reviewed"],
            }
        )

    write_csv(
        CATALOG_OUTPUT,
        [
            "hero_uid",
            "reference_name",
            "provisional_key",
            "reviewed",
            "notes",
        ],
        catalog_output_rows,
    )

    write_csv(
        APPEARANCES_OUTPUT,
        [
            "appearance_id",
            "hero_uid",
            "technical_cluster_id",
            "avatar_count",
            "sample_files",
            "appearance_type",
            "reviewed",
            "notes",
        ],
        appearances_output_rows,
    )

    write_csv(
        ALIASES_OUTPUT,
        [
            "hero_uid",
            "alias",
            "language",
            "source",
            "occurrences",
            "reviewed",
        ],
        aliases_output_rows,
    )

    write_csv(
        AVATARS_OUTPUT,
        [
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
        ],
        updated_avatar_rows,
    )

    appearances_by_hero: dict[str, int] = defaultdict(int)

    for row in appearances_output_rows:
        appearances_by_hero[str(row["hero_uid"])] += 1

    print(f"Héros créés : {len(catalog_output_rows)}")
    print(f"Apparences créées : {len(appearances_output_rows)}")
    print(f"Alias enregistrés : {len(aliases_output_rows)}")
    print(f"Avatars référencés : {len(updated_avatar_rows)}")
    print()
    print("Fichiers créés :")
    print(f"- {CATALOG_OUTPUT}")
    print(f"- {APPEARANCES_OUTPUT}")
    print(f"- {ALIASES_OUTPUT}")
    print(f"- {AVATARS_OUTPUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())