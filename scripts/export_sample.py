from __future__ import annotations

import hashlib
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pymysql
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from pymysql.cursors import DictCursor


EXPORT_DIRECTORY = Path("data/sample")
SAMPLE_SIZE = 20


def required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Variable manquante dans .env : {name}")

    return value


def create_connection():
    return pymysql.connect(
        host=required_env("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=required_env("DB_USER"),
        password=required_env("DB_PASSWORD"),
        database=required_env("DB_NAME"),
        charset="utf8mb4",
        cursorclass=DictCursor,
        connect_timeout=10,
        read_timeout=60,
        autocommit=True,
    )


def sanitize_filename(filename: str) -> str:
    """
    Supprime les caractères interdits dans les noms de fichiers Windows.
    """
    filename = Path(filename).name
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def extension_from_format(image_format: str | None) -> str:
    extensions = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
        "GIF": ".gif",
    }

    return extensions.get(image_format or "", ".bin")


def main() -> int:
    load_dotenv()
    EXPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    connection = None

    try:
        connection = create_connection()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    filename,
                    image_data,
                    screen_hash,
                    date,
                    TypeDeCombat
                FROM screenshots
                WHERE image_data IS NOT NULL
                  AND OCTET_LENGTH(image_data) > 0
                  AND TypeDeCombat = 'hero'
                ORDER BY id DESC
                LIMIT %s
                """,
                (SAMPLE_SIZE,),
            )

            rows: list[dict[str, Any]] = cursor.fetchall()

        if not rows:
            print("Aucune capture exploitable trouvée.")
            return 1

        print(f"{len(rows)} captures trouvées.")
        print(f"Répertoire : {EXPORT_DIRECTORY.resolve()}")
        print()

        exported_count = 0
        error_count = 0

        for row in rows:
            screenshot_id = row["id"]
            image_bytes: bytes = row["image_data"]

            try:
                with Image.open(BytesIO(image_bytes)) as image:
                    image_format = image.format
                    width, height = image.size

                    # Vérifie que Pillow peut réellement décoder l’image.
                    image.verify()

                extension = extension_from_format(image_format)

                original_stem = Path(
                    sanitize_filename(row["filename"])
                ).stem

                output_name = (
                    f"{screenshot_id}_{original_stem}{extension}"
                )

                output_path = EXPORT_DIRECTORY / output_name
                output_path.write_bytes(image_bytes)

                calculated_hash = hashlib.sha256(image_bytes).hexdigest()
                stored_hash = row["screen_hash"]

                hash_matches = calculated_hash == stored_hash

                print(
                    f"[OK] id={screenshot_id} | "
                    f"{width}x{height} | "
                    f"format={image_format} | "
                    f"taille={len(image_bytes):,} octets | "
                    f"hash_identique={hash_matches}"
                )

                exported_count += 1

            except (UnidentifiedImageError, OSError, ValueError) as error:
                print(
                    f"[ERREUR] id={screenshot_id} : {error}",
                    file=sys.stderr,
                )
                error_count += 1

        print()
        print(f"Images exportées : {exported_count}")
        print(f"Erreurs : {error_count}")

        return 0 if exported_count > 0 else 1

    except (pymysql.MySQLError, RuntimeError, ValueError) as error:
        print("Échec de l’export.", file=sys.stderr)
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())