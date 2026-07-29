from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


MANIFEST = Path("data/catalog/hero_avatar_manifest.csv")
SOURCE_DIR = Path(
    "data/batches/hero_batch_001/crops_dynamic_v1/avatars_inner"
)
DESTINATION_DIR = Path("data/crops/hero/avatars_inner")
REPORT = Path("data/catalog/reference_avatar_sync_report_v2.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copie dans le dossier canonique les avatars de référence "
            "manquants, en utilisant uniquement la source dynamique validée."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Effectue réellement les copies.",
    )
    return parser.parse_args()


def read_manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifeste absent : {MANIFEST}")

    with MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream, delimiter=";")
        rows = list(reader)

    if not rows:
        raise RuntimeError("Le manifeste est vide.")

    if "avatar_file" not in rows[0]:
        raise RuntimeError(
            "La colonne 'avatar_file' est absente du manifeste."
        )

    return rows


def write_report(rows: list[dict[str, str]]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "avatar_file",
        "hero_uid",
        "appearance_id",
        "status",
        "source_path",
        "destination_path",
    ]

    with REPORT.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    print("Synchronisation V2 des avatars de référence")
    print(f"Manifeste          : {MANIFEST}")
    print(f"Source unique      : {SOURCE_DIR}")
    print(f"Destination        : {DESTINATION_DIR}")
    print()

    if not SOURCE_DIR.exists():
        print(
            f"Source absente : {SOURCE_DIR}",
            file=sys.stderr,
        )
        return 1

    try:
        manifest_rows = read_manifest()
    except (FileNotFoundError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1

    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, str]] = []
    already_present = 0
    planned = 0
    copied = 0
    missing_source = 0
    duplicates = 0

    seen_names: set[str] = set()

    for row in manifest_rows:
        avatar_file = row.get("avatar_file", "").strip()

        if not avatar_file:
            continue

        if avatar_file in seen_names:
            duplicates += 1
        seen_names.add(avatar_file)

        source = SOURCE_DIR / avatar_file
        destination = DESTINATION_DIR / avatar_file

        if destination.exists():
            status = "ALREADY_PRESENT"
            already_present += 1

        elif not source.exists():
            status = "SOURCE_NOT_FOUND"
            missing_source += 1

        elif args.apply:
            shutil.copy2(source, destination)
            status = "COPIED"
            copied += 1

        else:
            status = "COPY_PLANNED"
            planned += 1

        report_rows.append(
            {
                "avatar_file": avatar_file,
                "hero_uid": row.get("hero_uid", ""),
                "appearance_id": row.get("appearance_id", ""),
                "status": status,
                "source_path": source.as_posix(),
                "destination_path": destination.as_posix(),
            }
        )

    write_report(report_rows)

    print("Résumé")
    print(f"- Avatars du manifeste       : {len(manifest_rows)}")
    print(f"- Déjà présents              : {already_present}")

    if args.apply:
        print(f"- Copiés                     : {copied}")
    else:
        print(f"- Copies prévues             : {planned}")

    print(f"- Sources introuvables       : {missing_source}")
    print(f"- Doublons dans le manifeste : {duplicates}")
    print(f"- Rapport                    : {REPORT}")

    if missing_source:
        print()
        print(
            "Certaines images n'existent pas dans la source dynamique.",
            file=sys.stderr,
        )
        return 2

    if not args.apply:
        print()
        print("Aucun fichier n'a été copié.")
        print(
            "Pour appliquer : "
            "python scripts/sync_catalog_reference_avatars_v2.py --apply"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
