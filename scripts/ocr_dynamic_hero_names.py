from __future__ import annotations

import csv
import html
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import easyocr
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


BATCH_DIR = Path("data/batches/hero_batch_001")

NAMES_DIR = (
    BATCH_DIR
    / "crops_dynamic_v1"
    / "names"
)

OCR_PROFILE = os.getenv(
    "OCR_PROFILE",
    "ocr_dynamic_v1",
).strip() or "ocr_dynamic_v1"

REPORTS_DIR = (
    BATCH_DIR
    / "reports"
    / OCR_PROFILE
)

CSV_OUTPUT = REPORTS_DIR / "hero_names_ocr.csv"
HTML_OUTPUT = REPORTS_DIR / "hero_names_ocr_review.html"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

# Première passe optimisée pour les langues utilisant
# principalement l'alphabet latin.
#
# Cette liste peut être désactivée avec :
# OCR_USE_ALLOWLIST=0
LATIN_ALLOWLIST = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "ÀÁÂÃÄÅÆÇÉÈÊËÍÌÎÏÑÓÒÔÕÖØŒÚÙÛÜÝŸ"
    "àáâãäåæçéèêëíìîïñóòôõöøœúùûüýÿ"
    "ŠŽČĆĐŁŃŚŹŻ"
    "šžčćđłńśźż"
    "'’- "
)

SLOT_PATTERN = re.compile(
    r"__(?P<side>[LR])(?P<slot>[1-5])$"
)

ID_PATTERN = re.compile(r"^(?P<id>\d+)")


def env_flag(
    name: str,
    default: bool,
) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return raw_value.strip().lower() not in {
        "0",
        "false",
        "no",
        "non",
        "off",
    }


def get_ocr_languages() -> list[str]:
    """
    Exemples dans Git Bash :

    OCR_LANGUAGES=fr,en \
    python scripts/ocr_dynamic_hero_names.py

    OCR_LANGUAGES=ru,en OCR_USE_ALLOWLIST=0 \
    python scripts/ocr_dynamic_hero_names.py
    """

    raw_value = os.getenv(
        "OCR_LANGUAGES",
        "fr,en",
    )

    languages = [
        language.strip()
        for language in raw_value.split(",")
        if language.strip()
    ]

    if not languages:
        raise RuntimeError(
            "OCR_LANGUAGES ne contient aucune langue."
        )

    return languages


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)

    value = value.replace("’", "'")
    value = value.replace("`", "'")

    value = " ".join(value.split())

    return value.strip(" -")


def extract_metadata(
    path: Path,
) -> tuple[str, str, int]:
    screenshot_match = ID_PATTERN.match(path.stem)
    slot_match = SLOT_PATTERN.search(path.stem)

    screenshot_id = (
        screenshot_match.group("id")
        if screenshot_match
        else ""
    )

    side = (
        slot_match.group("side")
        if slot_match
        else ""
    )

    slot = (
        int(slot_match.group("slot"))
        if slot_match
        else 0
    )

    return screenshot_id, side, slot


def prepare_image(path: Path) -> np.ndarray:
    """
    Agrandit la zone de nom sans appliquer de
    binarisation destructive.
    """

    with Image.open(path) as source:
        image = source.convert("RGB")

    image = image.resize(
        (
            image.width * 3,
            image.height * 3,
        ),
        Image.Resampling.LANCZOS,
    )

    image = ImageOps.autocontrast(
        image,
        cutoff=1,
    )

    return np.asarray(image)


def combine_results(
    detections: list,
) -> tuple[str, float, int]:
    """
    Regroupe les éventuelles détections de gauche
    à droite.
    """

    if not detections:
        return "", 0.0, 0

    sorted_detections = sorted(
        detections,
        key=lambda detection: min(
            point[0]
            for point in detection[0]
        ),
    )

    text_parts: list[str] = []
    weighted_confidence = 0.0
    total_weight = 0

    for _, detected_text, confidence in sorted_detections:
        detected_text = normalize_text(
            str(detected_text)
        )

        if not detected_text:
            continue

        text_parts.append(detected_text)

        weight = max(
            len(detected_text),
            1,
        )

        weighted_confidence += (
            float(confidence) * weight
        )

        total_weight += weight

    final_text = normalize_text(
        " ".join(text_parts)
    )

    if total_weight == 0:
        return "", 0.0, len(detections)

    confidence = (
        weighted_confidence / total_weight
    )

    return (
        final_text,
        confidence,
        len(detections),
    )


def confidence_status(
    text: str,
    confidence: float,
) -> str:
    if not text:
        return "NO_DETECTION"

    if confidence >= 0.85:
        return "HIGH"

    if confidence >= 0.60:
        return "MEDIUM"

    return "LOW"


def create_html_report(
    rows: list[dict[str, object]],
    languages: list[str],
    use_allowlist: bool,
) -> None:
    """
    Affiche d'abord les résultats les moins fiables.
    """

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            float(row["confidence"]),
            str(row["filename"]),
        ),
    )

    table_rows: list[str] = []

    for row in sorted_rows:
        source_path = Path(
            str(row["path"])
        )

        relative_path = os.path.relpath(
            source_path,
            REPORTS_DIR,
        ).replace("\\", "/")

        image_url = quote(relative_path)

        status = str(row["status"])

        if status == "HIGH":
            background = "#e8f5e9"
        elif status == "MEDIUM":
            background = "#fff8e1"
        else:
            background = "#ffebee"

        table_rows.append(
            f"""
            <tr style="background: {background}">
                <td>{html.escape(str(row["screenshot_id"]))}</td>
                <td>{html.escape(str(row["side"]))}</td>
                <td>{html.escape(str(row["slot"]))}</td>
                <td>
                    <img
                        src="{image_url}"
                        alt="{html.escape(str(row["filename"]))}"
                    >
                </td>
                <td class="ocr">
                    {html.escape(str(row["ocr_text"]))}
                </td>
                <td>{float(row["confidence"]):.4f}</td>
                <td>{html.escape(status)}</td>
                <td>{int(row["detection_count"])}</td>
                <td>{html.escape(str(row["filename"]))}</td>
            </tr>
            """
        )

    language_text = ", ".join(languages)

    document = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <title>Contrôle OCR dynamique des noms</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}

        .summary {{
            background: white;
            border: 1px solid #cccccc;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 18px;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            background: white;
        }}

        th,
        td {{
            border: 1px solid #cccccc;
            padding: 7px;
            text-align: left;
            vertical-align: middle;
        }}

        th {{
            position: sticky;
            top: 0;
            background: #222222;
            color: white;
        }}

        img {{
            width: 300px;
            max-height: 80px;
            object-fit: contain;
            image-rendering: auto;
        }}

        .ocr {{
            font-size: 20px;
            font-weight: bold;
        }}
    </style>
</head>

<body>
    <section class="summary">
        <h1>Contrôle OCR dynamique des noms</h1>

        <p>
            Les résultats les moins fiables sont affichés
            en premier.
        </p>

        <p>
            Langues chargées : {html.escape(language_text)}<br>
            Allowlist latine :
            {"activée" if use_allowlist else "désactivée"}<br>
            Images analysées : {len(rows)}
        </p>
    </section>

    <table>
        <thead>
            <tr>
                <th>Capture</th>
                <th>Côté</th>
                <th>Position</th>
                <th>Image</th>
                <th>Texte OCR</th>
                <th>Confiance</th>
                <th>Statut</th>
                <th>Détections</th>
                <th>Fichier</th>
            </tr>
        </thead>

        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>
</body>
</html>
"""

    HTML_OUTPUT.write_text(
        document,
        encoding="utf-8",
    )


def main() -> int:
    try:
        languages = get_ocr_languages()
    except RuntimeError as error:
        print(
            f"Configuration OCR invalide : {error}",
            file=sys.stderr,
        )
        return 1

    use_allowlist = env_flag(
        "OCR_USE_ALLOWLIST",
        default=True,
    )

    if not NAMES_DIR.exists():
        print(
            f"Dossier absent : {NAMES_DIR}",
            file=sys.stderr,
        )
        return 1

    image_paths = sorted(
        path
        for path in NAMES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        print(
            f"Aucune zone de nom dans {NAMES_DIR}",
            file=sys.stderr,
        )
        return 1

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Chargement d'EasyOCR sur le GPU...")
    print(f"Profil de sortie : {OCR_PROFILE}")
    print(
        "Langues : "
        + ", ".join(languages)
    )
    print(
        "Allowlist latine : "
        + (
            "activée"
            if use_allowlist
            else "désactivée"
        )
    )

    reader = easyocr.Reader(
        languages,
        gpu=True,
        verbose=False,
    )

    print(f"Images à analyser : {len(image_paths)}")
    print()

    started_at = time.perf_counter()
    rows: list[dict[str, object]] = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        screenshot_id, side, slot = extract_metadata(
            image_path
        )

        try:
            prepared_image = prepare_image(
                image_path
            )

            readtext_options = {
                "detail": 1,
                "paragraph": False,
                "decoder": "beamsearch",
                "batch_size": 1,
                "workers": 0,
                "min_size": 5,
                "text_threshold": 0.45,
                "low_text": 0.25,
                "link_threshold": 0.25,
                "contrast_ths": 0.10,
                "adjust_contrast": 0.60,
                "width_ths": 0.70,
                "add_margin": 0.05,
            }

            if use_allowlist:
                readtext_options["allowlist"] = (
                    LATIN_ALLOWLIST
                )

            detections = reader.readtext(
                prepared_image,
                **readtext_options,
            )

            (
                ocr_text,
                confidence,
                detection_count,
            ) = combine_results(detections)

            status = confidence_status(
                ocr_text,
                confidence,
            )

        except (
            UnidentifiedImageError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            ocr_text = ""
            confidence = 0.0
            detection_count = 0
            status = "ERROR"

            print(
                f"[ERREUR] {image_path.name} : "
                f"{error}",
                file=sys.stderr,
            )

        rows.append(
            {
                "screenshot_id": screenshot_id,
                "side": side,
                "slot": slot,
                "filename": image_path.name,
                "path": str(image_path),
                "ocr_text": ocr_text,
                "confidence": confidence,
                "status": status,
                "detection_count": detection_count,
                "ocr_languages": ",".join(languages),
                "allowlist_enabled": int(
                    use_allowlist
                ),
            }
        )

        print(
            f"[{index:04}/{len(image_paths):04}] "
            f"{screenshot_id} {side}{slot} -> "
            f"{ocr_text or '(rien)'} "
            f"({confidence:.3f}, {status})"
        )

    fieldnames = [
        "screenshot_id",
        "side",
        "slot",
        "filename",
        "ocr_text",
        "confidence",
        "status",
        "detection_count",
        "ocr_languages",
        "allowlist_enabled",
    ]

    with CSV_OUTPUT.open(
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

        for row in rows:
            writer.writerow(
                {
                    key: row[key]
                    for key in fieldnames
                }
            )

    create_html_report(
        rows=rows,
        languages=languages,
        use_allowlist=use_allowlist,
    )

    elapsed = (
        time.perf_counter() - started_at
    )

    status_counts: dict[str, int] = {}

    for row in rows:
        status = str(row["status"])

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    print()
    print("Résumé :")

    for status in (
        "HIGH",
        "MEDIUM",
        "LOW",
        "NO_DETECTION",
        "ERROR",
    ):
        print(
            f"- {status:<12} : "
            f"{status_counts.get(status, 0)}"
        )

    print()
    print(f"Durée totale : {elapsed:.1f} secondes")
    print(
        "Moyenne : "
        f"{elapsed / len(image_paths):.3f} "
        "seconde/image"
    )
    print()
    print(f"CSV : {CSV_OUTPUT}")
    print(f"Contrôle visuel : {HTML_OUTPUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
