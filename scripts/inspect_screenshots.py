from __future__ import annotations

import os
import sys
from typing import Any

import pymysql
from dotenv import load_dotenv
from pymysql.connections import Connection
from pymysql.cursors import DictCursor


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


def escape_identifier(name: str) -> str:
    return f"`{name.replace('`', '``')}`"


def main() -> int:
    load_dotenv()

    connection: Connection | None = None

    try:
        connection = create_connection()

        with connection.cursor() as cursor:
            cursor.execute("SHOW FULL COLUMNS FROM `screenshots`")
            columns: list[dict[str, Any]] = list(cursor.fetchall())

            cursor.execute(
                """
                SELECT COUNT(*) AS row_count
                FROM `screenshots`
                """
            )
            total_row = cursor.fetchone()

            if total_row is None:
                raise RuntimeError(
                    "Impossible de récupérer le nombre de lignes."
                )

            cursor.execute(
                """
                SELECT
                    `TypeDeCombat` AS combat_type,
                    COUNT(*) AS row_count,
                    COUNT(`image_data`) AS image_count
                FROM `screenshots`
                GROUP BY `TypeDeCombat`
                ORDER BY row_count DESC
                """
            )
            combat_types: list[dict[str, Any]] = list(
                cursor.fetchall()
            )

        print(
            f"Nombre total de lignes : "
            f"{total_row['row_count']}"
        )
        print()

        print("Répartition par type de combat :")

        for combat_type in combat_types:
            print(
                f"- {combat_type['combat_type']} : "
                f"{combat_type['row_count']} lignes, "
                f"{combat_type['image_count']} images disponibles"
            )

        print()
        print("Structure de la table screenshots :")
        print("-" * 100)

        heavy_columns: list[str] = []
        safe_columns: list[str] = []

        for column in columns:
            column_name = str(column["Field"])
            column_type = str(column["Type"]).lower()
            null_value = str(column["Null"])
            key_value = str(column["Key"] or "-")

            print(
                f"{column_name:<30} "
                f"type={column_type:<25} "
                f"null={null_value:<3} "
                f"clé={key_value:<3}"
            )

            if any(
                data_type in column_type
                for data_type in (
                    "blob",
                    "binary",
                    "text",
                    "json",
                )
            ):
                heavy_columns.append(column_name)
            else:
                safe_columns.append(column_name)

        print()
        print("Colonnes potentiellement volumineuses :")

        if heavy_columns:
            for column_name in heavy_columns:
                print(f"- {column_name}")
        else:
            print("- aucune")

        if safe_columns:
            selected_columns = ", ".join(
                escape_identifier(column_name)
                for column_name in safe_columns
            )

            sample_query = f"""
                SELECT {selected_columns}
                FROM `screenshots`
                ORDER BY `id` DESC
                LIMIT 5
            """

            with connection.cursor() as cursor:
                cursor.execute(sample_query)
                sample_rows: list[dict[str, Any]] = list(
                    cursor.fetchall()
                )

            print()
            print(
                "Exemple de 5 lignes, "
                "sans les colonnes volumineuses :"
            )
            print("-" * 100)

            for index, row in enumerate(sample_rows, start=1):
                print(f"Ligne {index} :")

                for key, value in row.items():
                    print(f"  {key}: {value}")

                print()

        if heavy_columns:
            length_expressions = ", ".join(
                (
                    f"OCTET_LENGTH("
                    f"{escape_identifier(column_name)}) "
                    f"AS "
                    f"{escape_identifier(column_name + '_length')}"
                )
                for column_name in heavy_columns
            )

            length_query = f"""
                SELECT
                    `id`,
                    {length_expressions}
                FROM `screenshots`
                ORDER BY `id` DESC
                LIMIT 5
            """

            with connection.cursor() as cursor:
                cursor.execute(length_query)
                length_rows: list[dict[str, Any]] = list(
                    cursor.fetchall()
                )

            print("Taille des colonnes volumineuses :")
            print("-" * 100)

            for row in length_rows:
                print(row)

        return 0

    except (
        pymysql.MySQLError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            "Échec de l'inspection.",
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