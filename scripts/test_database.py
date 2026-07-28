from __future__ import annotations

import os
import sys

import pymysql
from dotenv import load_dotenv
from pymysql.cursors import DictCursor


def required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Variable manquante dans .env : {name}")

    return value


def main() -> int:
    load_dotenv()

    connection = None

    try:
        connection = pymysql.connect(
            host=required_env("DB_HOST"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=required_env("DB_USER"),
            password=required_env("DB_PASSWORD"),
            database=required_env("DB_NAME"),
            charset="utf8mb4",
            cursorclass=DictCursor,
            connect_timeout=10,
            autocommit=True,
        )

        with connection.cursor() as cursor:
            cursor.execute("SELECT DATABASE() AS database_name")
            database = cursor.fetchone()

            cursor.execute("SELECT CURRENT_USER() AS mysql_account")
            user = cursor.fetchone()

            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()

        print("Connexion MySQL réussie.")
        print(f"Base : {database['database_name']}")
        print(f"Compte : {user['mysql_account']}")
        print(f"Nombre de tables visibles : {len(tables)}")
        print()

        for table in tables:
            print(f"- {next(iter(table.values()))}")

        return 0

    except (pymysql.MySQLError, RuntimeError, ValueError) as error:
        print("Échec de la connexion.", file=sys.stderr)
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())