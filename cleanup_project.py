from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path.cwd().resolve()

# Sorties anciennes, invalides ou purement diagnostiques.
# Elles sont toutes régénérables à partir des données conservées.
DELETE_PATHS = [
    Path("scripts/__pycache__"),
    Path("data/batches/hero_batch_001/crops"),
    Path(
        "data/batches/hero_batch_001/"
        "reports/anchored_clusters"
    ),
    Path(
        "data/batches/hero_batch_001/"
        "reports/visual_matching"
    ),
    Path(
        "data/batches/hero_batch_001/"
        "reports/crop_layout_check"
    ),
    Path(
        "data/batches/hero_batch_001/"
        "reports/crop_layout_check_v2"
    ),
    Path(
        "data/batches/hero_batch_001/"
        "reports/catalog_update_preview"
    ),
    Path(
        "data/batches/hero_batch_001/"
        "crops_dynamic_v1/fallback_debug"
    ),
    Path(
        "data/batches/hero_batch_001/"
        "reports/frame_detection_v1/debug"
    ),
]

# Scripts historiques qui ne font plus partie du pipeline actif.
# Ils sont déplacés, jamais supprimés.
LEGACY_SCRIPT_GROUPS = {
    "initial_catalog": [
        "build_cluster_manifest.py",
        "create_contact_sheets.py",
        "create_hero_catalog.py",
        "crop_hero_samples.py",
        "export_sample.py",
        "export_samples.py",
        "ocr_hero_names.py",
    ],
    "obsolete_layout": [
        "check_batch_crop_layout.py",
        "check_batch_crop_layout_v2.py",
        "cluster_batch_with_catalog_anchors.py",
        "extract_validation_batch_crops.py",
    ],
}

# Éléments administratifs ou exploratoires déplacés hors de la racine.
MOVE_PATHS = {
    Path("data/sample"): Path(
        "data/archive/legacy_samples/data_sample"
    ),
    Path("hero_catalog_update_inputs.zip"): Path(
        "data/archive/admin/hero_catalog_update_inputs.zip"
    ),
    Path("project_audit_report.zip"): Path(
        "data/archive/admin/project_audit_report.zip"
    ),
    Path("_project_audit"): Path(
        "data/archive/audits/project_audit_20260729"
    ),
    Path("audit_project_structure.py"): Path(
        "scripts/tools/audit_project_structure.py"
    ),
}

PROJECT_MAP = """# Carte du projet Hero Wars Vision

## Données importantes

- `data/catalog/` : catalogue maître des héros, apparences et alias.
- `data/crops/hero/avatars_inner/` : 200 références initiales utilisées par le catalogue.
- `data/batches/hero_batch_001/raw/` : 100 captures brutes du lot de validation.
- `data/batches/hero_batch_001/crops_dynamic_v1/avatars_inner/` : 1 000 avatars correctement découpés.
- `data/batches/hero_batch_001/validated/hero_identity_manifest.csv` : identité validée des 1 000 avatars.
- `data/curation/hero_batch_001_review_decisions.csv` : décisions humaines du lot.
- `data/catalog/backups/` : sauvegardes du catalogue avant mise à jour.

## Pipeline actif

1. `scripts/test_database.py`
   Vérifie la connexion MySQL.

2. `scripts/inspect_screenshots.py`
   Inspecte les captures présentes en base.

3. `scripts/export_validation_batch.py`
   Exporte un lot de captures depuis MySQL.

4. `scripts/create_validation_contact_sheets.py`
   Produit des planches de contrôle du lot.

5. `scripts/detect_avatar_frames_batch.py`
   Détecte dynamiquement les cadres des avatars.

6. `scripts/extract_dynamic_frame_crops.py`
   Extrait les portraits et zones de noms.

7. `scripts/match_dynamic_crops_to_catalog.py`
   Lance la reconnaissance visuelle sur les découpes dynamiques.

8. `scripts/ocr_dynamic_hero_names.py`
   Lit les noms des héros avec EasyOCR.

9. `scripts/reconcile_visual_ocr.py`
   Croise reconnaissance visuelle, OCR et catalogue.

10. `scripts/group_reconciliation_review.py`
    Regroupe les cas nécessitant une revue humaine.

11. `scripts/apply_hero_batch_001_review.py`
    Applique les décisions validées au catalogue.

## Dépendances internes à conserver

- `scripts/match_dynamic_crops_to_catalog.py`
  dépend de `scripts/match_batch_avatars_to_catalog.py`.
- `scripts/match_batch_avatars_to_catalog.py`
  dépend de `scripts/cluster_hero_avatars.py`.

Ces trois fichiers doivent donc rester ensemble dans `scripts/`.

## Scripts historiques

Les anciens essais sont rangés dans :

- `scripts/legacy/initial_catalog/`
- `scripts/legacy/obsolete_layout/`

Ils ne font plus partie du pipeline actif.

## Rapports

Les rapports actifs du lot se trouvent dans :

- `data/batches/hero_batch_001/reports/visual_matching_dynamic_v1/`
- `data/batches/hero_batch_001/reports/ocr_dynamic_v1/`
- `data/batches/hero_batch_001/reports/reconciliation_v1/`
- `data/batches/hero_batch_001/reports/reconciliation_groups_v1/`
- `data/batches/hero_batch_001/reports/catalog_update_applied/`

## Nettoyage effectué

Le nettoyage supprime uniquement :

- les anciennes découpes incorrectes ;
- les rapports produits à partir de ces mauvaises découpes ;
- les contrôles de mise en page devenus obsolètes ;
- les caches Python ;
- les images de diagnostic régénérables.

Le catalogue, les références visuelles valides, les captures brutes et les manifestes validés sont conservés.
"""


def human_size(size: int) -> str:
    units = ("o", "Ko", "Mo", "Go", "To")
    value = float(size)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "o":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{size} o"


def path_size(path: Path) -> int:
    if not path.exists():
        return 0

    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0

    total = 0

    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass

    return total


def git_status() -> str:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip()
    except OSError:
        return ""


def validate_project_root() -> None:
    required = [
        ROOT / "scripts",
        ROOT / "data" / "catalog",
        ROOT / "data" / "batches" / "hero_batch_001",
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        print(
            "Ce script doit être exécuté à la racine "
            "du projet hero-wars-vision.",
            file=sys.stderr,
        )
        print(
            "Éléments absents :",
            file=sys.stderr,
        )

        for path in missing:
            print(
                f"- {path}",
                file=sys.stderr,
            )

        raise SystemExit(1)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def move_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        raise RuntimeError(
            f"Destination déjà présente : {destination}"
        )

    shutil.move(
        str(source),
        str(destination),
    )


def build_plan() -> tuple[
    list[tuple[Path, int]],
    list[tuple[Path, Path, int]],
    list[tuple[Path, Path, int]],
]:
    deletions: list[tuple[Path, int]] = []
    moves: list[tuple[Path, Path, int]] = []
    legacy_moves: list[
        tuple[Path, Path, int]
    ] = []

    for relative_path in DELETE_PATHS:
        source = ROOT / relative_path

        if source.exists():
            deletions.append(
                (
                    relative_path,
                    path_size(source),
                )
            )

    for source_relative, destination_relative in (
        MOVE_PATHS.items()
    ):
        source = ROOT / source_relative

        if source.exists():
            moves.append(
                (
                    source_relative,
                    destination_relative,
                    path_size(source),
                )
            )

    for group, filenames in (
        LEGACY_SCRIPT_GROUPS.items()
    ):
        for filename in filenames:
            source_relative = (
                Path("scripts") / filename
            )

            destination_relative = (
                Path("scripts")
                / "legacy"
                / group
                / filename
            )

            source = ROOT / source_relative

            if source.exists():
                legacy_moves.append(
                    (
                        source_relative,
                        destination_relative,
                        path_size(source),
                    )
                )

    return deletions, moves, legacy_moves


def print_plan(
    deletions: list[tuple[Path, int]],
    moves: list[tuple[Path, Path, int]],
    legacy_moves: list[
        tuple[Path, Path, int]
    ],
) -> None:
    reclaimed = sum(
        size
        for _, size in deletions
    )

    print("PLAN DE NETTOYAGE")
    print("=" * 80)
    print(f"Projet : {ROOT}")
    print()

    print("Suppressions régénérables :")

    if not deletions:
        print("- aucune")

    for path, size in deletions:
        print(
            f"- {path.as_posix()} "
            f"({human_size(size)})"
        )

    print()
    print(
        "Espace libéré estimé : "
        f"{human_size(reclaimed)}"
    )
    print()

    print("Déplacements administratifs :")

    if not moves:
        print("- aucun")

    for source, destination, size in moves:
        print(
            f"- {source.as_posix()} "
            f"→ {destination.as_posix()} "
            f"({human_size(size)})"
        )

    print()
    print("Scripts déplacés vers legacy :")

    if not legacy_moves:
        print("- aucun")

    for source, destination, size in legacy_moves:
        print(
            f"- {source.as_posix()} "
            f"→ {destination.as_posix()} "
            f"({human_size(size)})"
        )

    print()
    print(
        "Éléments explicitement conservés :"
    )
    print(
        "- data/catalog/"
    )
    print(
        "- data/catalog/backups/"
    )
    print(
        "- data/crops/hero/avatars_inner/"
    )
    print(
        "- data/batches/hero_batch_001/raw/"
    )
    print(
        "- data/batches/hero_batch_001/"
        "crops_dynamic_v1/avatars_inner/"
    )
    print(
        "- data/batches/hero_batch_001/"
        "validated/"
    )
    print(
        "- tous les rapports dynamiques "
        "et de réconciliation"
    )


def apply_plan(
    deletions: list[tuple[Path, int]],
    moves: list[tuple[Path, Path, int]],
    legacy_moves: list[
        tuple[Path, Path, int]
    ],
) -> None:
    log_lines = [
        "Nettoyage du projet Hero Wars Vision",
        f"Date : {datetime.now().isoformat(timespec='seconds')}",
        f"Racine : {ROOT}",
        "",
    ]

    for relative_path, size in deletions:
        absolute_path = ROOT / relative_path

        remove_path(absolute_path)

        log_lines.append(
            "SUPPRIMÉ ; "
            f"{relative_path.as_posix()} ; "
            f"{size}"
        )

    for (
        source_relative,
        destination_relative,
        size,
    ) in moves:
        move_path(
            ROOT / source_relative,
            ROOT / destination_relative,
        )

        log_lines.append(
            "DÉPLACÉ ; "
            f"{source_relative.as_posix()} ; "
            f"{destination_relative.as_posix()} ; "
            f"{size}"
        )

    for (
        source_relative,
        destination_relative,
        size,
    ) in legacy_moves:
        move_path(
            ROOT / source_relative,
            ROOT / destination_relative,
        )

        log_lines.append(
            "SCRIPT_LEGACY ; "
            f"{source_relative.as_posix()} ; "
            f"{destination_relative.as_posix()} ; "
            f"{size}"
        )

    project_map_path = ROOT / "PROJECT_MAP.md"
    project_map_path.write_text(
        PROJECT_MAP,
        encoding="utf-8",
    )

    log_lines.append(
        "CRÉÉ ; PROJECT_MAP.md"
    )

    logs_dir = (
        ROOT
        / "data"
        / "archive"
        / "cleanup_logs"
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = (
        logs_dir
        / f"cleanup_{timestamp}.txt"
    )

    log_path.write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("Nettoyage appliqué.")
    print(f"Journal : {log_path}")
    print(
        "Carte du projet : PROJECT_MAP.md"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Nettoie et organise le projet "
            "Hero Wars Vision."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Applique réellement le plan. "
            "Sans cette option, aucune modification."
        ),
    )

    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Autorise l'application même si Git "
            "contient des modifications non validées."
        ),
    )

    args = parser.parse_args()

    validate_project_root()

    deletions, moves, legacy_moves = (
        build_plan()
    )

    print_plan(
        deletions=deletions,
        moves=moves,
        legacy_moves=legacy_moves,
    )

    if not args.apply:
        print()
        print(
            "MODE SIMULATION : aucun fichier "
            "n'a été modifié."
        )
        print()
        print(
            "Après avoir validé ou sauvegardé "
            "le travail Git :"
        )
        print(
            "python cleanup_project.py --apply"
        )
        return 0

    status = git_status()

    if status and not args.allow_dirty:
        print()
        print(
            "Nettoyage annulé : le dépôt Git "
            "contient des modifications.",
            file=sys.stderr,
        )
        print(
            "Crée d'abord un commit de sauvegarde, "
            "puis relance la commande.",
            file=sys.stderr,
        )
        print(
            "Pour ignorer volontairement cette "
            "protection :",
            file=sys.stderr,
        )
        print(
            "python cleanup_project.py "
            "--apply --allow-dirty",
            file=sys.stderr,
        )
        return 2

    try:
        apply_plan(
            deletions=deletions,
            moves=moves,
            legacy_moves=legacy_moves,
        )

    except (
        OSError,
        RuntimeError,
        shutil.Error,
    ) as error:
        print(
            f"Erreur pendant le nettoyage : {error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
