from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path.cwd().resolve()
SCRIPTS_DIR = ROOT / "scripts"

PIPELINE = {
    "detect": "detect_avatar_frames_batch.py",
    "extract": "extract_dynamic_frame_crops.py",
    "visual": "match_dynamic_crops_to_catalog.py",
    "ocr": "ocr_dynamic_hero_names.py",
    "reconcile": "reconcile_visual_ocr.py",
    "group": "group_reconciliation_review.py",
}

ORDER = list(PIPELINE)

BATCH_PATTERN = re.compile(r"^hero_batch_\d{3}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lance les scripts existants du pipeline Hero Wars sur "
            "n'importe quel lot, sans modifier leurs fichiers."
        )
    )

    parser.add_argument(
        "--batch",
        required=True,
        help="Nom du lot, par exemple hero_batch_002.",
    )

    parser.add_argument(
        "--step",
        choices=ORDER,
        action="append",
        help=(
            "Étape à exécuter. L'option peut être répétée. "
            "Exemple : --step extract --step visual."
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Exécute toutes les étapes dans l'ordre.",
    )

    return parser.parse_args()


def validate(args: argparse.Namespace) -> list[str]:
    if not BATCH_PATTERN.fullmatch(args.batch):
        raise ValueError(
            "--batch doit avoir la forme hero_batch_002."
        )

    batch_dir = ROOT / "data" / "batches" / args.batch

    if not batch_dir.exists():
        raise ValueError(
            f"Lot absent : {batch_dir}"
        )

    if args.all and args.step:
        raise ValueError(
            "Utilise soit --all, soit --step, pas les deux."
        )

    if args.all:
        return ORDER.copy()

    if not args.step:
        raise ValueError(
            "Indique au moins une étape avec --step, ou utilise --all."
        )

    selected = set(args.step)

    return [
        step
        for step in ORDER
        if step in selected
    ]


def patch_source(
    source: str,
    batch_name: str,
    step: str,
) -> str:
    patched = source.replace(
        "hero_batch_001",
        batch_name,
    )

    # Le lot 001 a montré qu'une largeur de 2,10 tronquait parfois
    # les noms longs comme "Mushy and Shroom".
    if step == "extract":
        patched = re.sub(
            r"NAME_WIDTH_IN_FRAME_WIDTHS\s*=\s*2\.10",
            "NAME_WIDTH_IN_FRAME_WIDTHS = 2.40",
            patched,
        )

    return patched


def run_step(
    batch_name: str,
    step: str,
) -> int:
    script_name = PIPELINE[step]
    source_path = SCRIPTS_DIR / script_name

    if not source_path.exists():
        print(
            f"[ERREUR] Script absent : {source_path}",
            file=sys.stderr,
        )
        return 1

    source = source_path.read_text(
        encoding="utf-8",
        errors="strict",
    )

    patched = patch_source(
        source=source,
        batch_name=batch_name,
        step=step,
    )

    runtime_path = (
        SCRIPTS_DIR
        / (
            f".__runtime_{source_path.stem}_"
            f"{os.getpid()}.py"
        )
    )

    runtime_path.write_text(
        patched,
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["HERO_BATCH"] = batch_name

    print()
    print("=" * 80)
    print(
        f"ÉTAPE : {step.upper()} — {batch_name}"
    )
    print(
        f"Script source : scripts/{script_name}"
    )
    print("=" * 80)
    print()

    try:
        completed = subprocess.run(
            [sys.executable, str(runtime_path)],
            cwd=ROOT,
            env=environment,
            check=False,
        )
    finally:
        try:
            runtime_path.unlink()
        except FileNotFoundError:
            pass

    # group_reconciliation_review.py renvoie actuellement 1 lorsqu'il
    # n'existe aucun cas à regrouper. Pour un lot de validation, ce cas
    # est un succès fonctionnel.
    if step == "group" and completed.returncode == 1:
        reconciliation = (
            ROOT
            / "data"
            / "batches"
            / batch_name
            / "reports"
            / "reconciliation_v1"
            / "reconciliation_results.csv"
        )

        if reconciliation.exists():
            print()
            print(
                "L'étape GROUP n'a créé aucun groupe. "
                "Cela peut simplement signifier qu'aucune revue n'est requise."
            )
            return 0

    return completed.returncode


def main() -> int:
    args = parse_args()

    try:
        steps = validate(args)
    except ValueError as error:
        print(
            f"Erreur : {error}",
            file=sys.stderr,
        )
        return 2

    print(
        f"Pipeline demandé pour : {args.batch}"
    )
    print(
        "Étapes : " + " → ".join(steps)
    )

    for step in steps:
        return_code = run_step(
            batch_name=args.batch,
            step=step,
        )

        if return_code != 0:
            print()
            print(
                f"Arrêt du pipeline : l'étape {step} "
                f"a renvoyé le code {return_code}.",
                file=sys.stderr,
            )
            return return_code

    print()
    print("=" * 80)
    print("ÉTAPES TERMINÉES")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
