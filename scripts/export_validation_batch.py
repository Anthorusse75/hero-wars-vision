from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import pymysql
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from pymysql.connections import Connection
from pymysql.cursors import DictCursor


DEFAULT_BATCH_SIZE = 100
DEFAULT_TIME_BUCKETS = 10
DEFAULT_RANDOM_SEED = 20260729

BATCHES_ROOT = Path("data/batches")

EXISTING_SAMPLE_DIRECTORIES = (
    Path("data/sample"),
    Path("data/samples"),
    Path("data/archive/legacy_samples/data_sample"),
)

SUPPORTED_BATCH_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exporte depuis MySQL un lot diversifié de captures Hero Wars, "
            "en excluant toutes les captures déjà utilisées."
        )
    )

    parser.add_argument(
        "--batch",
        help=(
            "Nom du lot à créer, par exemple hero_batch_002. "
            "Sans cette option, le prochain numéro disponible est choisi."
        ),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Nombre de captures à exporter (défaut : {DEFAULT_BATCH_SIZE}).",
    )

    parser.add_argument(
        "--time-buckets",
        type=int,
        default=DEFAULT_TIME_BUCKETS,
        help=(
            "Nombre de tranches chronologiques utilisées pour diversifier "
            f"la sélection (défaut : {DEFAULT_TIME_BUCKETS})."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Graine aléatoire pour rendre la sélection reproductible "
            f"(défaut : {DEFAULT_RANDOM_SEED})."
        ),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise ValueError("--count doit être strictement supérieur à zéro.")

    if args.time_buckets <= 0:
        raise ValueError(
            "--time-buckets doit être strictement supérieur à zéro."
        )

    if args.batch and not SUPPORTED_BATCH_NAME.fullmatch(args.batch):
        raise ValueError(
            "--batch ne peut contenir que des lettres, chiffres, tirets "
            "et underscores."
        )


def next_batch_name() -> str:
    highest_number = 0

    if BATCHES_ROOT.exists():
        for path in BATCHES_ROOT.iterdir():
            if not path.is_dir():
                continue

            match = re.fullmatch(r"hero_batch_(\d+)", path.name)

            if match:
                highest_number = max(
                    highest_number,
                    int(match.group(1)),
                )

    return f"hero_batch_{highest_number + 1:03d}"


def required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Variable manquante dans .env : {name}"
        )

    return value


def create_connection() -> Connection:
    return pymysql.connect(
        host=required_env("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=required_env("DB_USER"),
        password=required_env("DB_PASSWORD"),
        database=required_env("DB_NAME"),
        charset="utf8mb4",
        cursorclass=DictCursor,
        connect_timeout=10,
        read_timeout=120,
        autocommit=True,
    )


def add_id_from_filename(
    filename: str,
    identifiers: set[int],
) -> None:
    match = re.match(r"^(\d+)", filename)

    if match:
        identifiers.add(int(match.group(1)))


def existing_usage() -> tuple[
    set[int],
    set[str],
    Counter[str],
]:
    """
    Recense toutes les captures déjà utilisées dans :
    - data/sample et data/samples ;
    - l'ancien échantillon archivé ;
    - tous les manifests de data/batches ;
    - tous les fichiers bruts de data/batches.

    Les identifiants et les screen_hash sont exclus pour éviter qu'une
    même image soit sélectionnée sous un autre identifiant.
    """

    identifiers: set[int] = set()
    screen_hashes: set[str] = set()
    sources: Counter[str] = Counter()

    for directory in EXISTING_SAMPLE_DIRECTORIES:
        if not directory.exists():
            continue

        before = len(identifiers)

        for path in directory.rglob("*"):
            if path.is_file():
                add_id_from_filename(
                    path.name,
                    identifiers,
                )

        sources[directory.as_posix()] += (
            len(identifiers) - before
        )

    if not BATCHES_ROOT.exists():
        return identifiers, screen_hashes, sources

    for batch_directory in sorted(BATCHES_ROOT.iterdir()):
        if not batch_directory.is_dir():
            continue

        manifest_path = batch_directory / "batch_manifest.csv"

        if manifest_path.exists():
            manifest_ids_before = len(identifiers)
            manifest_hashes_before = len(screen_hashes)

            with manifest_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as csv_file:
                reader = csv.DictReader(
                    csv_file,
                    delimiter=";",
                )

                for row in reader:
                    screenshot_id = str(
                        row.get("screenshot_id") or ""
                    ).strip()

                    if screenshot_id.isdigit():
                        identifiers.add(int(screenshot_id))

                    screen_hash = str(
                        row.get("screen_hash") or ""
                    ).strip()

                    if screen_hash:
                        screen_hashes.add(screen_hash)

            sources[
                f"{batch_directory.name}/manifest_ids"
            ] += len(identifiers) - manifest_ids_before

            sources[
                f"{batch_directory.name}/manifest_hashes"
            ] += len(screen_hashes) - manifest_hashes_before

        raw_directory = batch_directory / "raw"

        if raw_directory.exists():
            raw_ids_before = len(identifiers)

            for path in raw_directory.rglob("*"):
                if path.is_file():
                    add_id_from_filename(
                        path.name,
                        identifiers,
                    )

            sources[
                f"{batch_directory.name}/raw"
            ] += len(identifiers) - raw_ids_before

    return identifiers, screen_hashes, sources


def user_key(row: dict[str, Any]) -> str:
    user_id = row.get("user_id")

    if user_id is not None:
        return f"id:{user_id}"

    username = str(row.get("username") or "").strip()

    if username:
        return f"name:{username}"

    return "unknown"


def select_diverse_rows(
    rows: list[dict[str, Any]],
    target_size: int,
    time_buckets: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """
    Divise l'historique en tranches chronologiques, puis sélectionne
    dans chaque tranche des captures provenant d'utilisateurs variés.
    """

    if len(rows) <= target_size:
        return rows.copy()

    rng = random.Random(random_seed)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    total_rows = len(rows)
    effective_buckets = min(
        time_buckets,
        target_size,
        total_rows,
    )

    for bucket_index in range(effective_buckets):
        start = round(
            bucket_index * total_rows / effective_buckets
        )
        end = round(
            (bucket_index + 1) * total_rows / effective_buckets
        )

        bucket = rows[start:end]

        quota = target_size // effective_buckets

        if bucket_index < target_size % effective_buckets:
            quota += 1

        rows_by_user: dict[str, list[dict[str, Any]]] = (
            defaultdict(list)
        )

        for row in bucket:
            rows_by_user[user_key(row)].append(row)

        for user_rows in rows_by_user.values():
            rng.shuffle(user_rows)

        active_users = list(rows_by_user)
        rng.shuffle(active_users)

        selected_in_bucket = 0

        while (
            selected_in_bucket < quota
            and active_users
        ):
            users_still_available: list[str] = []

            for current_user in active_users:
                user_rows = rows_by_user[current_user]

                if not user_rows:
                    continue

                row = user_rows.pop()
                screenshot_id = int(row["id"])

                if screenshot_id not in selected_ids:
                    selected.append(row)
                    selected_ids.add(screenshot_id)
                    selected_in_bucket += 1

                if user_rows:
                    users_still_available.append(
                        current_user
                    )

                if selected_in_bucket >= quota:
                    break

            active_users = users_still_available

    if len(selected) < target_size:
        remaining_rows = [
            row
            for row in rows
            if int(row["id"]) not in selected_ids
        ]

        rng.shuffle(remaining_rows)

        for row in remaining_rows:
            selected.append(row)
            selected_ids.add(int(row["id"]))

            if len(selected) >= target_size:
                break

    return sorted(
        selected,
        key=lambda row: (
            row["date"],
            int(row["id"]),
        ),
    )


def extension_from_format(
    image_format: str | None,
) -> str:
    extensions = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
    }

    return extensions.get(image_format or "", ".bin")


def sanitize_filename(filename: str) -> str:
    filename = Path(filename).stem
    filename = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        filename,
    )
    filename = filename.strip(" .")

    return filename or "screenshot"


def fetch_metadata(
    connection: Connection,
    excluded_ids: set[int],
    excluded_hashes: set[str],
) -> list[dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                `id`,
                `filename`,
                `date`,
                `user_id`,
                `username`,
                `guild_id`,
                `message_id`,
                `screen_hash`
            FROM `screenshots`
            WHERE `TypeDeCombat` = 'hero'
              AND `image_data` IS NOT NULL
            ORDER BY `date`, `id`
            """
        )

        rows = list(cursor.fetchall())

    filtered_rows: list[dict[str, Any]] = []

    for row in rows:
        screenshot_id = int(row["id"])

        if screenshot_id in excluded_ids:
            continue

        screen_hash = str(row.get("screen_hash") or "").strip()

        if screen_hash and screen_hash in excluded_hashes:
            continue

        filtered_rows.append(row)

    return filtered_rows


def fetch_images(
    connection: Connection,
    selected_ids: list[int],
) -> dict[int, dict[str, Any]]:
    rows_by_id: dict[int, dict[str, Any]] = {}

    chunk_size = 25

    for start in range(
        0,
        len(selected_ids),
        chunk_size,
    ):
        chunk = selected_ids[start:start + chunk_size]
        placeholders = ", ".join(["%s"] * len(chunk))

        query = f"""
            SELECT
                `id`,
                `filename`,
                `image_data`
            FROM `screenshots`
            WHERE `id` IN ({placeholders})
        """

        with connection.cursor() as cursor:
            cursor.execute(query, chunk)

            for row in cursor.fetchall():
                rows_by_id[int(row["id"])] = row

    return rows_by_id


def write_manifest(
    manifest_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "screenshot_id",
        "filename",
        "original_filename",
        "date",
        "user_id",
        "username",
        "guild_id",
        "message_id",
        "screen_hash",
        "width",
        "height",
        "format",
        "file_size_bytes",
    ]

    with manifest_path.open(
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
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 2

    load_dotenv()

    batch_name = args.batch or next_batch_name()
    output_root = BATCHES_ROOT / batch_name
    raw_directory = output_root / "raw"
    manifest_path = output_root / "batch_manifest.csv"

    if output_root.exists():
        print(
            f"Le lot existe déjà : {output_root}",
            file=sys.stderr,
        )
        print(
            "Aucun fichier n'a été modifié.",
            file=sys.stderr,
        )
        return 1

    connection: Connection | None = None

    try:
        (
            excluded_ids,
            excluded_hashes,
            exclusion_sources,
        ) = existing_usage()

        print(f"Lot à créer : {batch_name}")
        print(f"Taille demandée : {args.count}")
        print(f"Graine : {args.seed}")
        print()
        print(
            f"Identifiants déjà utilisés et exclus : "
            f"{len(excluded_ids)}"
        )
        print(
            f"Hashes déjà utilisés et exclus : "
            f"{len(excluded_hashes)}"
        )

        if exclusion_sources:
            print("Sources d'exclusion :")

            for source, count in exclusion_sources.most_common():
                print(f"- {source}: {count}")

        print()

        connection = create_connection()

        metadata_rows = fetch_metadata(
            connection=connection,
            excluded_ids=excluded_ids,
            excluded_hashes=excluded_hashes,
        )

        print(
            f"Captures de héros disponibles après exclusions : "
            f"{len(metadata_rows)}"
        )

        if len(metadata_rows) < args.count:
            print(
                "Nombre insuffisant de captures disponibles : "
                f"{len(metadata_rows)} pour {args.count} demandées.",
                file=sys.stderr,
            )
            return 1

        selected_rows = select_diverse_rows(
            rows=metadata_rows,
            target_size=args.count,
            time_buckets=args.time_buckets,
            random_seed=args.seed,
        )

        selected_ids = [
            int(row["id"])
            for row in selected_rows
        ]

        print(
            f"Captures sélectionnées : "
            f"{len(selected_ids)}"
        )
        print("Téléchargement des images sélectionnées...")

        image_rows = fetch_images(
            connection=connection,
            selected_ids=selected_ids,
        )

        raw_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        manifest_rows: list[dict[str, Any]] = []
        resolution_counts: Counter[str] = Counter()
        format_counts: Counter[str] = Counter()
        user_counts: Counter[str] = Counter()

        total_bytes = 0
        errors = 0

        metadata_by_id = {
            int(row["id"]): row
            for row in selected_rows
        }

        for index, screenshot_id in enumerate(
            selected_ids,
            start=1,
        ):
            metadata = metadata_by_id[screenshot_id]
            image_row = image_rows.get(screenshot_id)

            if image_row is None:
                print(
                    f"[ERREUR] Image absente pour "
                    f"id={screenshot_id}",
                    file=sys.stderr,
                )
                errors += 1
                continue

            image_bytes = bytes(image_row["image_data"])

            try:
                with Image.open(
                    BytesIO(image_bytes)
                ) as image:
                    image_format = image.format
                    width, height = image.size
                    image.verify()

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
            ) as error:
                print(
                    f"[ERREUR] id={screenshot_id} : "
                    f"{error}",
                    file=sys.stderr,
                )
                errors += 1
                continue

            extension = extension_from_format(
                image_format
            )

            original_name = sanitize_filename(
                str(metadata["filename"])
            )

            output_name = (
                f"{screenshot_id}_{original_name}"
                f"{extension}"
            )

            output_path = raw_directory / output_name
            output_path.write_bytes(image_bytes)

            resolution = f"{width}x{height}"

            resolution_counts[resolution] += 1
            format_counts[str(image_format)] += 1
            user_counts[user_key(metadata)] += 1

            total_bytes += len(image_bytes)

            manifest_rows.append(
                {
                    "screenshot_id": screenshot_id,
                    "filename": output_name,
                    "original_filename": metadata[
                        "filename"
                    ],
                    "date": metadata["date"],
                    "user_id": metadata["user_id"],
                    "username": metadata["username"],
                    "guild_id": metadata["guild_id"],
                    "message_id": metadata["message_id"],
                    "screen_hash": metadata["screen_hash"],
                    "width": width,
                    "height": height,
                    "format": image_format,
                    "file_size_bytes": len(image_bytes),
                }
            )

            print(
                f"[{index:03}/{len(selected_ids):03}] "
                f"id={screenshot_id} | "
                f"{resolution} | "
                f"{image_format} | "
                f"{metadata['username'] or 'inconnu'}"
            )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_manifest(
            manifest_path=manifest_path,
            rows=manifest_rows,
        )

        dates = [
            row["date"]
            for row in selected_rows
        ]

        print()
        print("Résumé du lot :")
        print(
            f"- Lot : {batch_name}"
        )
        print(
            f"- Images exportées : "
            f"{len(manifest_rows)}"
        )
        print(f"- Erreurs : {errors}")
        print(
            f"- Utilisateurs différents : "
            f"{len(user_counts)}"
        )

        if dates:
            print(f"- Première date : {min(dates)}")
            print(f"- Dernière date : {max(dates)}")

        print(
            f"- Volume total : "
            f"{total_bytes / 1024 / 1024:.1f} Mo"
        )

        print()
        print("Résolutions :")

        for resolution, count in (
            resolution_counts.most_common()
        ):
            print(f"- {resolution}: {count}")

        print()
        print("Formats :")

        for image_format, count in (
            format_counts.most_common()
        ):
            print(f"- {image_format}: {count}")

        print()
        print(f"Images : {raw_directory}")
        print(f"Manifeste : {manifest_path}")

        if errors:
            print()
            print(
                "Attention : le lot contient moins d'images que demandé "
                "à cause d'erreurs de lecture.",
                file=sys.stderr,
            )

        return 0 if manifest_rows else 1

    except (
        pymysql.MySQLError,
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
        csv.Error,
    ) as error:
        print(
            "Échec de la création du lot.",
            file=sys.stderr,
        )
        print(
            f"Erreur : {error}",
            file=sys.stderr,
        )
        return 1

    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
