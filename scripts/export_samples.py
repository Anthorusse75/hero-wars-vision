from __future__ import annotations

import os
import re
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import pymysql
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from pymysql.connections import Connection
from pymysql.cursors import DictCursor


EXPORT_ROOT = Path("data/samples")


def required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Variable manquante dans .env : {name}")

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


def sanitize_filename(filename: str) -> str:
    filename = Path(filename).stem
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    filename = filename.strip(" .")

    return filename or "screenshot"


def extension_from_format(image_format: str | None) -> str:
    extensions = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
        "GIF": ".gif",
    }

    return extensions.get(image_format or "", ".bin")


def fetch_screenshots(
    connection: Connection,
    combat_type: str,
    order: str,
    limit: int,
) -> list[dict[str, Any]]:
    if order not in {"ASC", "DESC"}:
        raise ValueError(f"Ordre SQL non autorisé : {order}")

    query = f"""
        SELECT
            `id`,
            `filename`,
            `image_data`,
            `date`,
            `TypeDeCombat`
        FROM `screenshots`
        WHERE `TypeDeCombat` = %s
          AND `image_data` IS NOT NULL
        ORDER BY `id` {order}
        LIMIT %s
    """

    with connection.cursor() as cursor:
        cursor.execute(query, (combat_type, limit))
        return list(cursor.fetchall())


def export_rows(
    rows: list[dict[str, Any]],
    resolution_counter: Counter[str],
) -> tuple[int, int]:
    exported = 0
    errors = 0

    for row in rows:
        screenshot_id = int(row["id"])
        combat_type = str(row["TypeDeCombat"])
        image_bytes = bytes(row["image_data"])

        destination_directory = EXPORT_ROOT / combat_type
        destination_directory.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image_format = image.format
                width, height = image.size
                image.verify()

            extension = extension_from_format(image_format)
            original_name = sanitize_filename(str(row["filename"]))

            output_path = destination_directory / (
                f"{screenshot_id}_{original_name}{extension}"
            )

            output_path.write_bytes(image_bytes)

            resolution_counter[
                f"{combat_type} — {width}x{height} — {image_format}"
            ] += 1

            print(
                f"[OK] {combat_type:<5} "
                f"id={screenshot_id:<6} "
                f"{width}x{height:<5} "
                f"format={image_format:<5} "
                f"taille={len(image_bytes):>8,} octets"
            )

            exported += 1

        except (UnidentifiedImageError, OSError, ValueError) as error:
            print(
                f"[ERREUR] id={screenshot_id} : {error}",
                file=sys.stderr,
            )
            errors += 1

    return exported, errors


def main() -> int:
    load_dotenv()

    connection: Connection | None = None
    resolution_counter: Counter[str] = Counter()

    try:
        connection = create_connection()

        # Dix anciennes et dix récentes pour couvrir plusieurs périodes.
        oldest_heroes = fetch_screenshots(
            connection=connection,
            combat_type="hero",
            order="ASC",
            limit=10,
        )

        newest_heroes = fetch_screenshots(
            connection=connection,
            combat_type="hero",
            order="DESC",
            limit=10,
        )

        # Il n’existe actuellement que douze captures de titans.
        titan_rows = fetch_screenshots(
            connection=connection,
            combat_type="titan",
            order="DESC",
            limit=100,
        )

        # Évite tout doublon éventuel.
        all_rows_by_id = {
            int(row["id"]): row
            for row in oldest_heroes + newest_heroes + titan_rows
        }

        rows = list(all_rows_by_id.values())

        print(f"Captures à exporter : {len(rows)}")
        print(f"Destination : {EXPORT_ROOT.resolve()}")
        print()

        exported, errors = export_rows(
            rows=rows,
            resolution_counter=resolution_counter,
        )

        print()
        print("Résumé des résolutions et formats :")

        for description, count in resolution_counter.most_common():
            print(f"- {description} : {count}")

        print()
        print(f"Images exportées : {exported}")
        print(f"Erreurs : {errors}")

        return 0 if exported > 0 else 1

    except (
        pymysql.MySQLError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as error:
        print("Échec de l’export.", file=sys.stderr)
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())