from __future__ import annotations

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


BATCH_NAME = "hero_batch_001"
BATCH_SIZE = 100
TIME_BUCKETS = 10
RANDOM_SEED = 20260728

OUTPUT_ROOT = Path("data/batches") / BATCH_NAME
RAW_DIRECTORY = OUTPUT_ROOT / "raw"
MANIFEST_PATH = OUTPUT_ROOT / "batch_manifest.csv"

EXISTING_SAMPLE_DIRECTORIES = (
    Path("data/sample"),
    Path("data/samples"),
)


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


def existing_screenshot_ids() -> set[int]:
    """
    Détecte les captures déjà exportées dans data/sample
    ou data/samples grâce à l'identifiant placé au début du nom.
    """

    identifiers: set[int] = set()

    for directory in EXISTING_SAMPLE_DIRECTORIES:
        if not directory.exists():
            continue

        for path in directory.rglob("*"):
            if not path.is_file():
                continue

            match = re.match(r"^(\d+)", path.name)

            if match:
                identifiers.add(int(match.group(1)))

    return identifiers


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
) -> list[dict[str, Any]]:
    """
    Divise l'historique en tranches chronologiques, puis sélectionne
    dans chaque tranche des captures provenant d'utilisateurs variés.
    """

    if len(rows) <= target_size:
        return rows.copy()

    rng = random.Random(RANDOM_SEED)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    total_rows = len(rows)

    for bucket_index in range(TIME_BUCKETS):
        start = round(
            bucket_index * total_rows / TIME_BUCKETS
        )
        end = round(
            (bucket_index + 1) * total_rows / TIME_BUCKETS
        )

        bucket = rows[start:end]

        quota = target_size // TIME_BUCKETS

        if bucket_index < target_size % TIME_BUCKETS:
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

    return [
        row
        for row in rows
        if int(row["id"]) not in excluded_ids
    ]


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


def main() -> int:
    load_dotenv()

    if MANIFEST_PATH.exists():
        print(
            f"Le lot existe déjà : {OUTPUT_ROOT}",
            file=sys.stderr,
        )
        print(
            "Aucun fichier n'a été modifié.",
            file=sys.stderr,
        )
        return 1

    connection: Connection | None = None

    try:
        excluded_ids = existing_screenshot_ids()

        print(
            f"Captures déjà utilisées et exclues : "
            f"{len(excluded_ids)}"
        )

        connection = create_connection()

        metadata_rows = fetch_metadata(
            connection=connection,
            excluded_ids=excluded_ids,
        )

        print(
            f"Captures de héros disponibles : "
            f"{len(metadata_rows)}"
        )

        selected_rows = select_diverse_rows(
            rows=metadata_rows,
            target_size=BATCH_SIZE,
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

        RAW_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
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

            output_path = RAW_DIRECTORY / output_name
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

        OUTPUT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        with MANIFEST_PATH.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                delimiter=";",
                fieldnames=[
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
                ],
            )

            writer.writeheader()
            writer.writerows(manifest_rows)

        dates = [
            row["date"]
            for row in selected_rows
        ]

        print()
        print("Résumé du lot :")
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
        print(f"Images : {RAW_DIRECTORY}")
        print(f"Manifeste : {MANIFEST_PATH}")

        return 0 if manifest_rows else 1

    except (
        pymysql.MySQLError,
        RuntimeError,
        ValueError,
        TypeError,
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