from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path


MANIFEST_CSV = Path(
    "data/catalog/hero_avatar_manifest.csv"
)

REFERENCE_DIR = Path(
    "data/crops/hero/avatars_inner"
)

SEARCH_ROOTS = [
    Path("data/batches"),
    Path("data/crops"),
]

REPORT_CSV = Path(
    "data/catalog/reference_avatar_sync_report.csv"
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronise les images du manifeste maître "
            "dans le dossier canonique des avatars de référence."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Copie réellement les fichiers manquants. "
            "Sans cette option, seule une simulation est faite."
        ),
    )

    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_manifest() -> list[dict[str, str]]:
    if not MANIFEST_CSV.exists():
        raise RuntimeError(
            f"Manifeste absent : {MANIFEST_CSV}"
        )

    with MANIFEST_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        rows = list(
            csv.DictReader(
                csv_file,
                delimiter=";",
            )
        )

    if not rows:
        raise RuntimeError(
            "Le manifeste maître est vide."
        )

    if "avatar_file" not in rows[0]:
        raise RuntimeError(
            "La colonne avatar_file est absente du manifeste."
        )

    return rows


def build_source_index() -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                if path.resolve().parent == REFERENCE_DIR.resolve():
                    continue
            except OSError:
                pass

            index[path.name].append(path)

    return index


def select_source(
    avatar_file: str,
    candidates: list[Path],
) -> tuple[Path | None, str]:
    if not candidates:
        return None, "SOURCE_NOT_FOUND"

    if len(candidates) == 1:
        return candidates[0], "UNIQUE_SOURCE"

    hashes: dict[str, list[Path]] = defaultdict(list)

    for candidate in candidates:
        try:
            hashes[sha256(candidate)].append(candidate)
        except OSError:
            continue

    if len(hashes) == 1:
        selected = sorted(
            candidates,
            key=lambda path: (
                "crops_dynamic_v1" not in path.as_posix(),
                path.as_posix(),
            ),
        )[0]

        return selected, "MULTIPLE_IDENTICAL_SOURCES"

    return None, "AMBIGUOUS_DIFFERENT_SOURCES"


def write_report(
    rows: list[dict[str, object]],
) -> None:
    REPORT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "avatar_file",
        "hero_uid",
        "appearance_id",
        "status",
        "source_path",
        "destination_path",
        "details",
    ]

    with REPORT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            delimiter=";",
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    try:
        manifest_rows = read_manifest()
    except RuntimeError as error:
        print(
            error,
            file=sys.stderr,
        )
        return 1

    REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Avatars déclarés dans le manifeste : "
        f"{len(manifest_rows)}"
    )
    print(
        f"Dossier canonique : {REFERENCE_DIR}"
    )
    print()
    print("Indexation des sources disponibles...")

    source_index = build_source_index()

    report_rows: list[dict[str, object]] = []

    already_present = 0
    copied = 0
    copy_planned = 0
    unresolved = 0
    ambiguous = 0

    seen_manifest_files: set[str] = set()
    duplicate_manifest_files: set[str] = set()

    for manifest_row in manifest_rows:
        avatar_file = manifest_row[
            "avatar_file"
        ].strip()

        if not avatar_file:
            continue

        if avatar_file in seen_manifest_files:
            duplicate_manifest_files.add(
                avatar_file
            )

        seen_manifest_files.add(
            avatar_file
        )

        destination = (
            REFERENCE_DIR
            / avatar_file
        )

        if destination.exists():
            already_present += 1

            report_rows.append(
                {
                    "avatar_file": avatar_file,
                    "hero_uid": manifest_row.get(
                        "hero_uid",
                        "",
                    ),
                    "appearance_id": manifest_row.get(
                        "appearance_id",
                        "",
                    ),
                    "status": "ALREADY_PRESENT",
                    "source_path": "",
                    "destination_path": (
                        destination.as_posix()
                    ),
                    "details": "",
                }
            )

            continue

        source, source_status = select_source(
            avatar_file,
            source_index.get(
                avatar_file,
                [],
            ),
        )

        if source is None:
            if source_status == (
                "AMBIGUOUS_DIFFERENT_SOURCES"
            ):
                ambiguous += 1
            else:
                unresolved += 1

            report_rows.append(
                {
                    "avatar_file": avatar_file,
                    "hero_uid": manifest_row.get(
                        "hero_uid",
                        "",
                    ),
                    "appearance_id": manifest_row.get(
                        "appearance_id",
                        "",
                    ),
                    "status": source_status,
                    "source_path": " | ".join(
                        path.as_posix()
                        for path in source_index.get(
                            avatar_file,
                            [],
                        )
                    ),
                    "destination_path": (
                        destination.as_posix()
                    ),
                    "details": "",
                }
            )

            continue

        if args.apply:
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                source,
                destination,
            )

            copied += 1
            final_status = "COPIED"
        else:
            copy_planned += 1
            final_status = "COPY_PLANNED"

        report_rows.append(
            {
                "avatar_file": avatar_file,
                "hero_uid": manifest_row.get(
                    "hero_uid",
                    "",
                ),
                "appearance_id": manifest_row.get(
                    "appearance_id",
                    "",
                ),
                "status": final_status,
                "source_path": source.as_posix(),
                "destination_path": (
                    destination.as_posix()
                ),
                "details": source_status,
            }
        )

    write_report(report_rows)

    present_after = sum(
        1
        for avatar_file in seen_manifest_files
        if (
            REFERENCE_DIR
            / avatar_file
        ).exists()
    )

    print()
    print("Résumé :")
    print(
        f"- Déjà présents              : "
        f"{already_present}"
    )

    if args.apply:
        print(
            f"- Copiés                     : "
            f"{copied}"
        )
    else:
        print(
            f"- Copies prévues             : "
            f"{copy_planned}"
        )

    print(
        f"- Sources introuvables       : "
        f"{unresolved}"
    )
    print(
        f"- Sources ambiguës           : "
        f"{ambiguous}"
    )
    print(
        f"- Doublons dans le manifeste : "
        f"{len(duplicate_manifest_files)}"
    )

    if args.apply:
        print(
            f"- Références présentes après : "
            f"{present_after}"
        )

    print()
    print(f"Rapport : {REPORT_CSV}")

    if unresolved or ambiguous:
        print()
        print(
            "La synchronisation n'est pas complète.",
            file=sys.stderr,
        )
        return 2

    if not args.apply:
        print()
        print(
            "Aucun fichier n'a été copié."
        )
        print(
            "Pour appliquer :"
        )
        print(
            "python scripts/"
            "sync_catalog_reference_avatars.py --apply"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
