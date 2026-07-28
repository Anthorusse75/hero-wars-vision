from __future__ import annotations

import csv
import html
import os
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, UnidentifiedImageError
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


AVATAR_DIR = Path("data/crops/hero/avatars_inner")
OCR_CSV = Path("data/reports/hero_names_ocr.csv")
OUTPUT_DIR = Path("data/reports/avatar_clusters")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

# Distance cosinus :
# 0,06 = très strict
# 0,18 = plus permissif
DISTANCE_THRESHOLDS = (
    0.06,
    0.10,
    0.14,
    0.18,
)

BATCH_SIZE = 32


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    value = value.replace("’", "'")
    value = value.replace("`", "'")
    value = " ".join(value.split())

    return value.strip()


def load_ocr_names() -> dict[str, str]:
    if not OCR_CSV.exists():
        raise RuntimeError(f"Fichier absent : {OCR_CSV}")

    names: dict[str, str] = {}

    with OCR_CSV.open(
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file,
            delimiter=";",
        )

        for row in reader:
            filename = row.get("filename", "").strip()
            ocr_text = normalize_name(
                row.get("ocr_text", "")
            )

            if filename:
                names[filename] = ocr_text

    return names


class AvatarDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        transform,
    ) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, str]:
        image_path = self.image_paths[index]

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise RuntimeError(
                f"Impossible d'ouvrir {image_path}: {error}"
            ) from error

        width, height = image.size

        # Deux vues du même avatar :
        # - image complète ;
        # - centre légèrement resserré pour réduire l’influence
        #   du cadre, du niveau et des bordures.
        center_crop = image.crop(
            (
                round(width * 0.08),
                round(height * 0.12),
                round(width * 0.92),
                round(height * 0.95),
            )
        )

        views = torch.stack(
            (
                self.transform(image),
                self.transform(center_crop),
            )
        )

        return views, image_path.name


def create_feature_model(
    device: torch.device,
) -> tuple[nn.Module, object]:
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    # On retire la couche de classification ImageNet.
    # Le réseau renvoie alors un vecteur visuel de 512 valeurs.
    model.fc = nn.Identity()

    model.eval()
    model.to(device)

    return model, weights.transforms()


def extract_embeddings(
    image_paths: list[Path],
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    model, transform = create_feature_model(device)

    dataset = AvatarDataset(
        image_paths=image_paths,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    embedding_batches: list[np.ndarray] = []
    ordered_filenames: list[str] = []

    with torch.inference_mode():
        for views, filenames in loader:
            # Shape initiale :
            # batch × 2 vues × 3 couleurs × hauteur × largeur
            batch_size, view_count, channels, height, width = (
                views.shape
            )

            views = views.reshape(
                batch_size * view_count,
                channels,
                height,
                width,
            )

            views = views.to(
                device,
                non_blocking=True,
            )

            features = model(views)
            features = functional.normalize(
                features,
                dim=1,
            )

            # On moyenne les deux représentations du même avatar.
            features = features.reshape(
                batch_size,
                view_count,
                -1,
            ).mean(dim=1)

            features = functional.normalize(
                features,
                dim=1,
            )

            embedding_batches.append(
                features.cpu().numpy()
            )

            ordered_filenames.extend(filenames)

    embeddings = np.concatenate(
        embedding_batches,
        axis=0,
    )

    return embeddings, ordered_filenames


def create_clusters(
    embeddings: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    if len(embeddings) == 1:
        return np.array([1], dtype=int)

    pairwise_distances = pdist(
        embeddings,
        metric="cosine",
    )

    hierarchy = linkage(
        pairwise_distances,
        method="average",
    )

    cluster_labels = fcluster(
        hierarchy,
        t=distance_threshold,
        criterion="distance",
    )

    return cluster_labels


def group_cluster_indices(
    cluster_labels: np.ndarray,
) -> list[tuple[int, list[int]]]:
    clusters: dict[int, list[int]] = defaultdict(list)

    for image_index, cluster_id in enumerate(cluster_labels):
        clusters[int(cluster_id)].append(image_index)

    return sorted(
        clusters.items(),
        key=lambda item: (
            -len(item[1]),
            item[0],
        ),
    )


def analyse_clusters(
    cluster_labels: np.ndarray,
    filenames: list[str],
    ocr_names: dict[str, str],
) -> dict[str, object]:
    clusters = group_cluster_indices(cluster_labels)

    mixed_cluster_ids: list[int] = []
    singletons = 0

    name_to_clusters: dict[str, set[int]] = defaultdict(set)

    for cluster_id, indices in clusters:
        if len(indices) == 1:
            singletons += 1

        names_in_cluster = {
            ocr_names.get(filenames[index], "")
            for index in indices
            if ocr_names.get(filenames[index], "")
        }

        if len(names_in_cluster) > 1:
            mixed_cluster_ids.append(cluster_id)

        for name in names_in_cluster:
            name_to_clusters[name].add(cluster_id)

    split_names = {
        name: sorted(cluster_ids)
        for name, cluster_ids in name_to_clusters.items()
        if len(cluster_ids) > 1
    }

    return {
        "cluster_count": len(clusters),
        "singleton_count": singletons,
        "mixed_cluster_ids": mixed_cluster_ids,
        "mixed_cluster_count": len(mixed_cluster_ids),
        "split_names": split_names,
        "split_name_count": len(split_names),
    }


def calculate_cluster_similarity(
    embeddings: np.ndarray,
    indices: list[int],
) -> tuple[float, float]:
    if len(indices) < 2:
        return 1.0, 1.0

    cluster_embeddings = embeddings[indices]
    similarities = cluster_embeddings @ cluster_embeddings.T

    upper_triangle = similarities[
        np.triu_indices(
            len(indices),
            k=1,
        )
    ]

    return (
        float(np.mean(upper_triangle)),
        float(np.min(upper_triangle)),
    )


def create_html_report(
    distance_threshold: float,
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    filenames: list[str],
    ocr_names: dict[str, str],
    analysis: dict[str, object],
) -> Path:
    clusters = group_cluster_indices(cluster_labels)

    threshold_name = str(distance_threshold).replace(
        ".",
        "_",
    )

    output_path = (
        OUTPUT_DIR
        / f"clusters_distance_{threshold_name}.html"
    )

    cluster_sections: list[str] = []

    for display_number, (
        cluster_id,
        indices,
    ) in enumerate(clusters, start=1):
        ocr_values = sorted(
            {
                ocr_names.get(
                    filenames[index],
                    "",
                )
                for index in indices
                if ocr_names.get(
                    filenames[index],
                    "",
                )
            }
        )

        mixed = len(ocr_values) > 1

        mean_similarity, minimum_similarity = (
            calculate_cluster_similarity(
                embeddings,
                indices,
            )
        )

        cards: list[str] = []

        for index in indices:
            filename = filenames[index]
            image_path = AVATAR_DIR / filename

            relative_path = os.path.relpath(
                image_path,
                OUTPUT_DIR,
            ).replace("\\", "/")

            image_url = quote(relative_path)
            ocr_text = ocr_names.get(filename, "")

            cards.append(
                f"""
                <div class="card">
                    <img
                        src="{image_url}"
                        alt="{html.escape(filename)}"
                    >
                    <div class="name">
                        {html.escape(ocr_text or "(sans OCR)")}
                    </div>
                    <div class="filename">
                        {html.escape(filename)}
                    </div>
                </div>
                """
            )

        cluster_class = "cluster mixed" if mixed else "cluster"

        cluster_sections.append(
            f"""
            <section class="{cluster_class}">
                <h2>
                    Groupe {display_number}
                    — {len(indices)} avatar(s)
                </h2>

                <p>
                    Groupe technique : {cluster_id}<br>
                    Nom(s) OCR :
                    <strong>
                        {html.escape(", ".join(ocr_values) or "aucun")}
                    </strong><br>
                    Similarité moyenne :
                    {mean_similarity:.4f}<br>
                    Similarité minimale :
                    {minimum_similarity:.4f}
                </p>

                <div class="cards">
                    {''.join(cards)}
                </div>
            </section>
            """
        )

    mixed_cluster_ids = analysis["mixed_cluster_ids"]
    split_names = analysis["split_names"]

    split_names_html = "".join(
        f"""
        <li>
            {html.escape(name)} :
            groupes techniques
            {html.escape(", ".join(map(str, cluster_ids)))}
        </li>
        """
        for name, cluster_ids in split_names.items()
    )

    document = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8">

    <title>
        Regroupement visuel des avatars
    </title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #eeeeee;
        }}

        .summary,
        .cluster {{
            background: white;
            border: 1px solid #cccccc;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 18px;
        }}

        .cluster.mixed {{
            border: 4px solid #d32f2f;
            background: #ffebee;
        }}

        .cards {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .card {{
            width: 135px;
            background: #fafafa;
            border: 1px solid #bbbbbb;
            padding: 6px;
            text-align: center;
        }}

        .card img {{
            width: 120px;
            height: 120px;
            object-fit: contain;
        }}

        .name {{
            font-weight: bold;
            margin-top: 5px;
        }}

        .filename {{
            font-size: 10px;
            color: #555555;
            overflow-wrap: anywhere;
            margin-top: 4px;
        }}

        .warning {{
            color: #b71c1c;
            font-weight: bold;
        }}
    </style>
</head>

<body>
    <h1>Regroupement visuel des avatars</h1>

    <section class="summary">
        <h2>Résumé</h2>

        <p>
            Distance maximale :
            <strong>{distance_threshold:.2f}</strong>
        </p>

        <ul>
            <li>
                Groupes :
                {analysis["cluster_count"]}
            </li>

            <li>
                Groupes d’un seul avatar :
                {analysis["singleton_count"]}
            </li>

            <li>
                Groupes mélangeant plusieurs noms OCR :
                <strong>
                    {analysis["mixed_cluster_count"]}
                </strong>
            </li>

            <li>
                Noms répartis dans plusieurs groupes :
                <strong>
                    {analysis["split_name_count"]}
                </strong>
            </li>
        </ul>

        <p>
            Groupes mélangés :
            {html.escape(str(mixed_cluster_ids))}
        </p>

        <h3>Noms éclatés entre plusieurs groupes</h3>

        <ul>
            {split_names_html or "<li>Aucun</li>"}
        </ul>

        <p>
            Les noms OCR servent uniquement au contrôle.
            Ils ne sont pas utilisés pour créer les groupes.
        </p>
    </section>

    {''.join(cluster_sections)}
</body>
</html>
"""

    output_path.write_text(
        document,
        encoding="utf-8",
    )

    return output_path


def main() -> int:
    if not AVATAR_DIR.exists():
        print(
            f"Dossier absent : {AVATAR_DIR}",
            file=sys.stderr,
        )
        return 1

    image_paths = sorted(
        path
        for path in AVATAR_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        print(
            f"Aucun avatar dans {AVATAR_DIR}",
            file=sys.stderr,
        )
        return 1

    try:
        ocr_names = load_ocr_names()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Appareil utilisé : {device}")

    if device.type == "cuda":
        print(
            "GPU :",
            torch.cuda.get_device_name(0),
        )

    print(f"Avatars à analyser : {len(image_paths)}")
    print()
    print(
        "Chargement du modèle ResNet18 "
        "et création des signatures visuelles..."
    )

    started_at = time.perf_counter()

    try:
        embeddings, filenames = extract_embeddings(
            image_paths=image_paths,
            device=device,
        )
    except (
        RuntimeError,
        OSError,
        ValueError,
    ) as error:
        print(
            f"Échec de l'extraction : {error}",
            file=sys.stderr,
        )
        return 1

    extraction_duration = time.perf_counter() - started_at

    print(
        f"Signatures créées en "
        f"{extraction_duration:.1f} secondes."
    )
    print()

    summary_rows: list[dict[str, object]] = []

    for threshold in DISTANCE_THRESHOLDS:
        cluster_labels = create_clusters(
            embeddings=embeddings,
            distance_threshold=threshold,
        )

        analysis = analyse_clusters(
            cluster_labels=cluster_labels,
            filenames=filenames,
            ocr_names=ocr_names,
        )

        report_path = create_html_report(
            distance_threshold=threshold,
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            filenames=filenames,
            ocr_names=ocr_names,
            analysis=analysis,
        )

        summary_rows.append(
            {
                "distance_threshold": threshold,
                **analysis,
                "report": str(report_path),
            }
        )

        print(
            f"Distance {threshold:.2f} : "
            f"{analysis['cluster_count']} groupes | "
            f"{analysis['mixed_cluster_count']} mélangés | "
            f"{analysis['split_name_count']} noms éclatés | "
            f"{analysis['singleton_count']} singletons"
        )

    summary_path = OUTPUT_DIR / "summary.csv"

    with summary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            delimiter=";",
            fieldnames=[
                "distance_threshold",
                "cluster_count",
                "singleton_count",
                "mixed_cluster_count",
                "split_name_count",
                "mixed_cluster_ids",
                "split_names",
                "report",
            ],
        )

        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    print()
    print(f"Résumé CSV : {summary_path}")
    print(f"Rapports HTML : {OUTPUT_DIR}")
    print()
    print(
        "Le premier lancement peut télécharger "
        "les poids préentraînés de ResNet18."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())