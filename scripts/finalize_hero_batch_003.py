from __future__ import annotations

import argparse
import csv
import shutil
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path


BATCH_NAME = "hero_batch_003"

BATCH_ROOT = Path("data/batches") / BATCH_NAME
RECONCILIATION = (
    BATCH_ROOT
    / "reports"
    / "reconciliation_v2"
    / "reconciliation_results.csv"
)
VALIDATED_DIR = BATCH_ROOT / "validated"

HEROES_CSV = Path("data/catalog/heroes.csv")
ALIASES_CSV = Path("data/catalog/hero_name_aliases.csv")

OUTPUT_MANIFEST = VALIDATED_DIR / "slot_identity_manifest.csv"
OUTPUT_DECISIONS = VALIDATED_DIR / "manual_decisions.csv"
OUTPUT_SUMMARY = VALIDATED_DIR / "independent_evaluation_summary.txt"

MANUAL_DECISIONS = {
    ("2820", "R", "5"): {
        "decision": "EMPTY_SLOT",
        "hero_name": "",
        "reason": "Emplacement visuellement vide sur la capture complète.",
    },
    ("6900", "R", "3"): {
        "decision": "MANUAL_VISUAL_REVIEW",
        "hero_name": "Elmir",
        "reason": "Nom Elmir lisible sur la capture complète.",
    },
    ("30831", "R", "2"): {
        "decision": "MANUAL_VISUAL_REVIEW",
        "hero_name": "Astaroth",
        "reason": "Avatar et nom Astaroth confirmés sur la capture complète.",
    },
    ("30831", "R", "3"): {
        "decision": "MANUAL_VISUAL_REVIEW",
        "hero_name": "Jorgen",
        "reason": "Avatar et nom Jorgen confirmés sur la capture complète.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalise les quatre cas d'identité restant à revoir "
            "dans hero_batch_003."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Écrit les fichiers validés. Sans cette option, "
            "le script effectue une simulation."
        ),
    )
    return parser.parse_args()


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        first_line = stream.readline()

    candidates = [";", ",", "\t"]
    return max(candidates, key=first_line.count)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    if not path.exists():
        raise RuntimeError(f"Fichier absent : {path}")

    delimiter = detect_delimiter(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(f"En-tête CSV absent : {path}")

    return fieldnames, rows, delimiter


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )

    return "".join(
        character
        for character in without_accents.casefold()
        if character.isalnum()
    )


def find_uid_field(fieldnames: list[str]) -> str | None:
    preferred = [
        "hero_uid",
        "uid",
        "hero_id",
    ]

    normalized = {
        field.casefold(): field
        for field in fieldnames
    }

    for candidate in preferred:
        if candidate in normalized:
            return normalized[candidate]

    for field in fieldnames:
        lowered = field.casefold()

        if "hero" in lowered and "uid" in lowered:
            return field

    return None


def candidate_name_fields(fieldnames: list[str]) -> list[str]:
    candidates = [
        field
        for field in fieldnames
        if any(
            token in field.casefold()
            for token in (
                "name",
                "alias",
                "label",
                "display",
                "provisional",
                "canonical",
            )
        )
    ]

    return candidates or list(fieldnames)


def add_catalog_entries(
    index: dict[str, set[str]],
    path: Path,
) -> None:
    fieldnames, rows, _ = read_csv(path)
    uid_field = find_uid_field(fieldnames)

    if uid_field is None:
        raise RuntimeError(
            f"Impossible de trouver la colonne hero_uid dans {path}."
        )

    name_fields = candidate_name_fields(fieldnames)

    for row in rows:
        hero_uid = str(row.get(uid_field) or "").strip()

        if not hero_uid:
            continue

        for field in name_fields:
            value = str(row.get(field) or "").strip()

            if not value:
                continue

            key = normalize_text(value)

            if key:
                index.setdefault(key, set()).add(hero_uid)


def build_catalog_index() -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}

    add_catalog_entries(index, HEROES_CSV)

    if ALIASES_CSV.exists():
        add_catalog_entries(index, ALIASES_CSV)

    return index


def resolve_hero_uid(
    hero_name: str,
    catalog_index: dict[str, set[str]],
    source_row: dict[str, str],
) -> str:
    target = normalize_text(hero_name)
    catalog_matches = catalog_index.get(target, set())

    visual_name = normalize_text(
        str(source_row.get("visual_hero_name") or "")
    )
    visual_uid = str(
        source_row.get("visual_hero_uid") or ""
    ).strip()

    if visual_name == target and visual_uid:
        if catalog_matches and visual_uid not in catalog_matches:
            raise RuntimeError(
                f"Incohérence de catalogue pour {hero_name} : "
                f"visuel={visual_uid}, catalogue={sorted(catalog_matches)}."
            )

        return visual_uid

    if len(catalog_matches) == 1:
        return next(iter(catalog_matches))

    if not catalog_matches:
        raise RuntimeError(
            f"Héros introuvable dans le catalogue : {hero_name}."
        )

    raise RuntimeError(
        f"Plusieurs UID correspondent à {hero_name} : "
        f"{sorted(catalog_matches)}."
    )


def review_required(row: dict[str, str]) -> bool:
    value = str(row.get("review_required") or "").strip().casefold()

    return value in {
        "1",
        "true",
        "yes",
        "oui",
    }


def backup_existing(path: Path, timestamp: str) -> None:
    if not path.exists():
        return

    backup = path.with_name(
        f"{path.stem}_before_finalize_{timestamp}{path.suffix}"
    )
    shutil.copy2(path, backup)


def main() -> int:
    args = parse_args()

    try:
        fieldnames, rows, _ = read_csv(RECONCILIATION)
        catalog_index = build_catalog_index()
    except (RuntimeError, OSError, csv.Error) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    required_fields = {
        "screenshot_id",
        "side",
        "slot",
        "decision",
        "final_hero_uid",
        "final_hero_name",
        "review_required",
    }

    missing_fields = sorted(required_fields - set(fieldnames))

    if missing_fields:
        print(
            "Erreur : colonnes absentes de la réconciliation : "
            + ", ".join(missing_fields),
            file=sys.stderr,
        )
        return 1

    rows_by_key: dict[tuple[str, str, str], dict[str, str]] = {}

    for row in rows:
        key = (
            str(row.get("screenshot_id") or "").strip(),
            str(row.get("side") or "").strip().upper(),
            str(row.get("slot") or "").strip(),
        )

        if key in rows_by_key:
            print(
                f"Erreur : emplacement dupliqué : {key}",
                file=sys.stderr,
            )
            return 1

        rows_by_key[key] = row

    missing_keys = [
        key
        for key in MANUAL_DECISIONS
        if key not in rows_by_key
    ]

    if missing_keys:
        print(
            f"Erreur : cas manuels absents : {missing_keys}",
            file=sys.stderr,
        )
        return 1

    existing_review_keys = {
        key
        for key, row in rows_by_key.items()
        if review_required(row)
    }

    expected_review_keys = set(MANUAL_DECISIONS)

    if existing_review_keys != expected_review_keys:
        print(
            "Erreur : les cas restant à revoir ne correspondent pas "
            "exactement aux quatre décisions attendues.",
            file=sys.stderr,
        )
        print(
            f"Attendus : {sorted(expected_review_keys)}",
            file=sys.stderr,
        )
        print(
            f"Trouvés : {sorted(existing_review_keys)}",
            file=sys.stderr,
        )
        return 1

    manual_rows: list[dict[str, str]] = []

    try:
        for key, manual in MANUAL_DECISIONS.items():
            row = rows_by_key[key]
            decision = manual["decision"]
            hero_name = manual["hero_name"]

            if decision == "EMPTY_SLOT":
                hero_uid = ""
            else:
                hero_uid = resolve_hero_uid(
                    hero_name,
                    catalog_index,
                    row,
                )

            row["decision"] = decision
            row["final_hero_uid"] = hero_uid
            row["final_hero_name"] = hero_name
            row["review_required"] = "0"

            manual_rows.append(
                {
                    "screenshot_id": key[0],
                    "side": key[1],
                    "slot": key[2],
                    "decision": decision,
                    "hero_uid": hero_uid,
                    "hero_name": hero_name,
                    "reason": manual["reason"],
                }
            )

    except RuntimeError as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    remaining_review = [
        row
        for row in rows
        if review_required(row)
    ]

    empty_rows = [
        row
        for row in rows
        if str(row.get("decision") or "").strip().upper()
        == "EMPTY_SLOT"
    ]

    identified_rows = [
        row
        for row in rows
        if str(row.get("final_hero_uid") or "").strip()
    ]

    unidentified_non_empty = [
        row
        for row in rows
        if (
            str(row.get("decision") or "").strip().upper()
            != "EMPTY_SLOT"
            and not str(row.get("final_hero_uid") or "").strip()
        )
    ]

    decision_counts = Counter(
        str(row.get("decision") or "").strip()
        for row in rows
    )

    errors: list[str] = []

    if len(rows) != 1000:
        errors.append(
            f"1000 lignes attendues, {len(rows)} trouvées."
        )

    if remaining_review:
        errors.append(
            f"{len(remaining_review)} cas restent en revue."
        )

    if len(empty_rows) != 1:
        errors.append(
            f"1 emplacement vide attendu, {len(empty_rows)} trouvé(s)."
        )

    if len(identified_rows) != 999:
        errors.append(
            f"999 identités attendues, {len(identified_rows)} trouvées."
        )

    if unidentified_non_empty:
        errors.append(
            f"{len(unidentified_non_empty)} emplacement(s) non vide(s) "
            "sans identité."
        )

    print("FINALISATION DES IDENTITÉS — HERO_BATCH_003")
    print("=" * 72)
    print(f"Emplacements analysés : {len(rows)}")
    print(f"Identités attribuées : {len(identified_rows)}")
    print(f"Emplacements vides : {len(empty_rows)}")
    print(f"Décisions manuelles : {len(manual_rows)}")
    print(f"Cas restant à revoir : {len(remaining_review)}")
    print()

    print("Décisions manuelles :")

    for row in manual_rows:
        label = (
            row["hero_name"]
            if row["hero_name"]
            else "EMPTY_SLOT"
        )
        print(
            f"- {row['screenshot_id']} "
            f"{row['side']}{row['slot']} : {label}"
        )

    print()

    if errors:
        print("VALIDATION REFUSÉE :", file=sys.stderr)

        for error in errors:
            print(f"- {error}", file=sys.stderr)

        return 1

    if not args.apply:
        print("MODE SIMULATION : aucun fichier n'a été écrit.")
        print()
        print("Pour appliquer :")
        print(
            "python scripts/finalize_hero_batch_003.py --apply"
        )
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    VALIDATED_DIR.mkdir(parents=True, exist_ok=True)

    for path in (
        OUTPUT_MANIFEST,
        OUTPUT_DECISIONS,
        OUTPUT_SUMMARY,
    ):
        backup_existing(path, timestamp)

    output_fields = list(fieldnames)

    write_csv(
        OUTPUT_MANIFEST,
        output_fields,
        rows,
    )

    manual_fields = [
        "screenshot_id",
        "side",
        "slot",
        "decision",
        "hero_uid",
        "hero_name",
        "reason",
    ]

    write_csv(
        OUTPUT_DECISIONS,
        manual_fields,
        manual_rows,
    )

    summary_lines = [
        "ÉVALUATION INDÉPENDANTE DES IDENTITÉS — HERO_BATCH_003",
        "=" * 72,
        "",
        f"Emplacements analysés : {len(rows)}",
        f"Emplacements héros : {len(identified_rows)}",
        f"Emplacements vides : {len(empty_rows)}",
        f"Identités finales attribuées : {len(identified_rows)}",
        f"Décisions manuelles : {len(manual_rows)}",
        f"Cas restant à revoir : {len(remaining_review)}",
        "",
        "Origine des décisions finales :",
    ]

    for decision, count in sorted(
        decision_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        summary_lines.append(
            f"- {decision or 'SANS_DECISION'} : {count}"
        )

    summary_lines.extend(
        [
            "",
            "Décisions manuelles :",
        ]
    )

    for row in manual_rows:
        label = (
            f"{row['hero_name']} ({row['hero_uid']})"
            if row["hero_name"]
            else "EMPTY_SLOT"
        )
        summary_lines.append(
            f"- {row['screenshot_id']} "
            f"{row['side']}{row['slot']} : {label}"
        )

    summary_lines.extend(
        [
            "",
            "Remarque : le taux final inclut quatre décisions "
            "issues d'une revue visuelle manuelle.",
            "",
        ]
    )

    OUTPUT_SUMMARY.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("Finalisation appliquée.")
    print(f"Manifeste final : {OUTPUT_MANIFEST}")
    print(f"Décisions manuelles : {OUTPUT_DECISIONS}")
    print(f"Résumé : {OUTPUT_SUMMARY}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
