from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path.cwd().resolve()
AUDIT_DIR = ROOT / "_project_audit"
ZIP_OUTPUT = ROOT / "project_audit_report.zip"

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".vscode",
    "node_modules",
    "_project_audit",
}

TRANSIENT_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".env",
    ".gitignore",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
}

MAX_HASH_FILE_SIZE = 100 * 1024 * 1024
COMPACT_FILE_LIMIT_PER_DIR = 25


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


def normalize_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def run_git_command(arguments: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        return completed.returncode, completed.stdout

    except OSError:
        return 1, ""


def load_git_information() -> tuple[set[str], str]:
    tracked: set[str] = set()

    return_code, output = run_git_command(
        ["ls-files", "-z"]
    )

    if return_code == 0:
        tracked = {
            path
            for path in output.split("\0")
            if path
        }

    _, status_output = run_git_command(
        ["status", "--short", "--branch"]
    )

    return tracked, status_output


def categorize_path(relative_path: str) -> str:
    path = relative_path.replace("\\", "/")

    if path.startswith("scripts/"):
        return "scripts"

    if path.startswith("src/"):
        return "source"

    if path.startswith("tests/"):
        return "tests"

    if path.startswith("data/catalog/backups/"):
        return "catalog_backups"

    if path.startswith("data/catalog/"):
        return "catalog"

    if path.startswith("data/manifests/"):
        return "manifests"

    if "/validated/" in path:
        return "validated"

    if "/reports/" in path or path.startswith("data/reports/"):
        return "reports"

    if "/crops" in path or path.startswith("data/crops/"):
        return "generated_crops"

    if "/raw/" in path or path.endswith("/raw"):
        return "raw_data"

    if path.startswith("data/batches/"):
        return "batch_data"

    if path.startswith("data/"):
        return "data_other"

    if path.startswith("docs/"):
        return "documentation"

    if path.startswith(".github/"):
        return "github"

    return "project_root_or_other"


def walk_project() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    file_rows: list[dict[str, object]] = []
    transient_rows: list[dict[str, object]] = []

    for current_root, dir_names, file_names in os.walk(ROOT):
        current_path = Path(current_root)

        for dir_name in list(dir_names):
            candidate = current_path / dir_name
            relative = normalize_relative(candidate)

            if dir_name in TRANSIENT_DIR_NAMES:
                try:
                    size = sum(
                        item.stat().st_size
                        for item in candidate.rglob("*")
                        if item.is_file()
                    )
                except OSError:
                    size = 0

                transient_rows.append(
                    {
                        "path": relative,
                        "type": "directory",
                        "size_bytes": size,
                        "size_human": human_size(size),
                        "cleanup_level": "SAFE",
                        "reason": "Cache temporaire recréé automatiquement.",
                    }
                )

            if dir_name in EXCLUDED_DIR_NAMES:
                dir_names.remove(dir_name)

        for file_name in file_names:
            path = current_path / file_name

            try:
                stat = path.stat()
            except OSError:
                continue

            relative = normalize_relative(path)

            file_rows.append(
                {
                    "path": relative,
                    "directory": str(
                        Path(relative).parent
                    ).replace("\\", "/"),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "size_human": human_size(stat.st_size),
                    "modified": datetime.fromtimestamp(
                        stat.st_mtime
                    ).isoformat(timespec="seconds"),
                    "category": categorize_path(relative),
                }
            )

    return file_rows, transient_rows


def build_directory_rows(
    file_rows: list[dict[str, object]],
    tracked_files: set[str],
) -> list[dict[str, object]]:
    aggregates: dict[str, dict[str, object]] = {}

    for file_row in file_rows:
        relative_path = str(file_row["path"])
        path = Path(relative_path)

        directories = [
            Path("."),
            *[
                Path(*path.parts[:index])
                for index in range(
                    1,
                    len(path.parts),
                )
            ],
        ]

        for directory in directories:
            key = (
                "."
                if str(directory) == "."
                else directory.as_posix()
            )

            aggregate = aggregates.setdefault(
                key,
                {
                    "path": key,
                    "file_count": 0,
                    "size_bytes": 0,
                    "tracked_file_count": 0,
                    "untracked_file_count": 0,
                    "categories": Counter(),
                },
            )

            aggregate["file_count"] = (
                int(aggregate["file_count"]) + 1
            )

            aggregate["size_bytes"] = (
                int(aggregate["size_bytes"])
                + int(file_row["size_bytes"])
            )

            if relative_path in tracked_files:
                aggregate["tracked_file_count"] = (
                    int(
                        aggregate[
                            "tracked_file_count"
                        ]
                    )
                    + 1
                )
            else:
                aggregate["untracked_file_count"] = (
                    int(
                        aggregate[
                            "untracked_file_count"
                        ]
                    )
                    + 1
                )

            aggregate["categories"][
                str(file_row["category"])
            ] += 1

    rows: list[dict[str, object]] = []

    for key, aggregate in aggregates.items():
        categories: Counter = aggregate["categories"]
        dominant_category = (
            categories.most_common(1)[0][0]
            if categories
            else ""
        )

        rows.append(
            {
                "path": key,
                "file_count": aggregate["file_count"],
                "size_bytes": aggregate["size_bytes"],
                "size_human": human_size(
                    int(aggregate["size_bytes"])
                ),
                "tracked_file_count": aggregate[
                    "tracked_file_count"
                ],
                "untracked_file_count": aggregate[
                    "untracked_file_count"
                ],
                "dominant_category": dominant_category,
                "category_distribution": " | ".join(
                    f"{name}:{count}"
                    for name, count
                    in categories.most_common()
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            str(row["path"]).count("/"),
            str(row["path"]),
        ),
    )


def analyse_python_script(
    relative_path: str,
    tracked: bool,
) -> dict[str, object]:
    path = ROOT / relative_path

    imports: set[str] = set()
    functions: list[str] = []
    classes: list[str] = []
    path_literals: set[str] = set()
    has_main = False
    syntax_error = ""

    try:
        source = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        tree = ast.parse(
            source,
            filename=relative_path,
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(
                    alias.name
                    for alias in node.names
                )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)

            elif isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                functions.append(node.name)

                if node.name == "main":
                    has_main = True

            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

            elif isinstance(node, ast.Constant):
                if isinstance(node.value, str):
                    value = node.value

                    if (
                        "/" in value
                        or "\\" in value
                        or value.endswith(
                            (
                                ".csv",
                                ".json",
                                ".html",
                                ".txt",
                                ".png",
                                ".jpg",
                                ".zip",
                            )
                        )
                    ):
                        if len(value) <= 300:
                            path_literals.add(value)

    except SyntaxError as error:
        syntax_error = (
            f"{error.msg} ligne {error.lineno}"
        )

    except OSError as error:
        syntax_error = str(error)

    return {
        "path": relative_path,
        "tracked": int(tracked),
        "size_bytes": path.stat().st_size,
        "size_human": human_size(
            path.stat().st_size
        ),
        "has_main_function": int(has_main),
        "function_count": len(functions),
        "class_count": len(classes),
        "imports": " | ".join(
            sorted(imports)
        ),
        "path_literals": " | ".join(
            sorted(path_literals)
        ),
        "syntax_error": syntax_error,
    }


def build_script_rows(
    file_rows: list[dict[str, object]],
    tracked_files: set[str],
) -> list[dict[str, object]]:
    script_paths = sorted(
        str(row["path"])
        for row in file_rows
        if str(row["extension"]) == ".py"
    )

    return [
        analyse_python_script(
            relative_path=relative_path,
            tracked=relative_path in tracked_files,
        )
        for relative_path in script_paths
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def find_duplicates(
    file_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_size: dict[
        int,
        list[dict[str, object]],
    ] = defaultdict(list)

    for row in file_rows:
        size = int(row["size_bytes"])

        if 0 < size <= MAX_HASH_FILE_SIZE:
            by_size[size].append(row)

    duplicate_rows: list[dict[str, object]] = []
    group_number = 0

    for size, candidate_rows in by_size.items():
        if len(candidate_rows) < 2:
            continue

        by_hash: dict[
            str,
            list[dict[str, object]],
        ] = defaultdict(list)

        for row in candidate_rows:
            path = ROOT / str(row["path"])

            try:
                digest = file_sha256(path)
            except OSError:
                continue

            by_hash[digest].append(row)

        for digest, matching_rows in by_hash.items():
            if len(matching_rows) < 2:
                continue

            group_number += 1

            for row in matching_rows:
                duplicate_rows.append(
                    {
                        "duplicate_group": (
                            f"DUP_{group_number:04d}"
                        ),
                        "sha256": digest,
                        "size_bytes": size,
                        "size_human": human_size(size),
                        "path": row["path"],
                        "category": row["category"],
                    }
                )

    return duplicate_rows


def build_cleanup_candidates(
    file_rows: list[dict[str, object]],
    transient_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    candidates = list(transient_rows)

    existing_directories = {
        str(row["directory"])
        for row in file_rows
    }

    known_rules = [
        (
            "data/batches/hero_batch_001/crops",
            "LIKELY_SAFE",
            "Anciennes découpes remplacées par crops_dynamic_v1.",
        ),
        (
            "data/batches/hero_batch_001/reports/visual_matching",
            "LIKELY_SAFE",
            "Rapport calculé avec les anciennes découpes incorrectes.",
        ),
        (
            "data/batches/hero_batch_001/reports/anchored_clusters",
            "LIKELY_SAFE",
            "Clustering calculé avec les anciennes découpes incorrectes.",
        ),
        (
            "data/batches/hero_batch_001/reports/catalog_update_preview",
            "LIKELY_SAFE",
            "Aperçu de simulation conservable seulement à titre de diagnostic.",
        ),
        (
            "data/batches/hero_batch_001/crops_dynamic_v1/fallback_debug",
            "REVIEW",
            "Image de diagnostic du fallback, non nécessaire au pipeline final.",
        ),
        (
            "data/batches/hero_batch_001/reports/frame_detection_v1",
            "REVIEW",
            "Rapport de validation de la détection des cadres.",
        ),
        (
            "data/catalog/backups",
            "KEEP_OR_ARCHIVE",
            "Sauvegardes du catalogue : conserver au moins la dernière sauvegarde validée.",
        ),
        (
            "data/crops/hero",
            "REVIEW",
            "Jeu initial de 200 références ; peut encore être référencé par le catalogue.",
        ),
        (
            "data/reports",
            "REVIEW",
            "Rapports historiques du catalogue initial.",
        ),
    ]

    for directory, level, reason in known_rules:
        if any(
            path == directory
            or path.startswith(
                directory + "/"
            )
            for path in existing_directories
        ):
            size = sum(
                int(row["size_bytes"])
                for row in file_rows
                if str(row["path"]).startswith(
                    directory + "/"
                )
            )

            candidates.append(
                {
                    "path": directory,
                    "type": "directory",
                    "size_bytes": size,
                    "size_human": human_size(size),
                    "cleanup_level": level,
                    "reason": reason,
                }
            )

    return sorted(
        candidates,
        key=lambda row: (
            {
                "SAFE": 0,
                "LIKELY_SAFE": 1,
                "REVIEW": 2,
                "KEEP_OR_ARCHIVE": 3,
            }.get(
                str(row["cleanup_level"]),
                99,
            ),
            str(row["path"]),
        ),
    )


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    with path.open(
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
        writer.writerows(rows)


def build_full_tree(
    file_rows: list[dict[str, object]],
) -> str:
    children: dict[str, list[str]] = defaultdict(list)
    sizes: dict[str, int] = defaultdict(int)
    file_sizes: dict[str, int] = {}

    for row in file_rows:
        path = Path(str(row["path"]))
        file_sizes[path.as_posix()] = int(
            row["size_bytes"]
        )

        parent = (
            "."
            if str(path.parent) == "."
            else path.parent.as_posix()
        )

        children[parent].append(path.name)

        current = path.parent

        while str(current) != ".":
            sizes[current.as_posix()] += int(
                row["size_bytes"]
            )
            current = current.parent

        sizes["."] += int(row["size_bytes"])

    directory_paths = {
        str(Path(str(row["path"])).parent).replace(
            "\\",
            "/",
        )
        for row in file_rows
    }

    for directory in sorted(directory_paths):
        if directory == ".":
            continue

        parent_path = Path(directory).parent
        parent = (
            "."
            if str(parent_path) == "."
            else parent_path.as_posix()
        )

        children[parent].append(
            Path(directory).name + "/"
        )

    lines = [
        f"{ROOT.name}/  [{human_size(sizes['.'])}]"
    ]

    def render(
        directory: str,
        prefix: str,
    ) -> None:
        entries = sorted(
            set(children.get(directory, [])),
            key=lambda name: (
                not name.endswith("/"),
                name.casefold(),
            ),
        )

        for index, entry in enumerate(entries):
            is_last = index == len(entries) - 1
            connector = (
                "└── "
                if is_last
                else "├── "
            )

            if entry.endswith("/"):
                child_name = entry[:-1]
                child_path = (
                    child_name
                    if directory == "."
                    else f"{directory}/{child_name}"
                )

                lines.append(
                    f"{prefix}{connector}"
                    f"{entry} "
                    f"[{human_size(sizes[child_path])}]"
                )

                render(
                    child_path,
                    prefix
                    + (
                        "    "
                        if is_last
                        else "│   "
                    ),
                )

            else:
                file_path = (
                    entry
                    if directory == "."
                    else f"{directory}/{entry}"
                )

                lines.append(
                    f"{prefix}{connector}"
                    f"{entry} "
                    f"[{human_size(file_sizes[file_path])}]"
                )

    render(".", "")

    return "\n".join(lines) + "\n"


def build_compact_tree(
    directory_rows: list[dict[str, object]],
    file_rows: list[dict[str, object]],
) -> str:
    direct_files: dict[
        str,
        list[dict[str, object]],
    ] = defaultdict(list)

    for row in file_rows:
        direct_files[str(row["directory"])].append(row)

    directories = {
        str(row["path"]): row
        for row in directory_rows
    }

    child_directories: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for directory in directories:
        if directory == ".":
            continue

        path = Path(directory)
        parent = (
            "."
            if str(path.parent) == "."
            else path.parent.as_posix()
        )

        child_directories[parent].append(
            directory
        )

    lines = [
        (
            f"{ROOT.name}/ "
            f"[{directories.get('.', {}).get('size_human', '0 o')}]"
        )
    ]

    def render(
        directory: str,
        prefix: str,
    ) -> None:
        child_dirs = sorted(
            child_directories.get(
                directory,
                [],
            ),
            key=str.casefold,
        )

        files = sorted(
            direct_files.get(
                directory,
                [],
            ),
            key=lambda row: str(
                row["filename"]
            ).casefold(),
        )

        entries: list[
            tuple[str, object]
        ] = [
            ("directory", child)
            for child in child_dirs
        ]

        if len(files) <= COMPACT_FILE_LIMIT_PER_DIR:
            entries.extend(
                ("file", file_row)
                for file_row in files
            )
        elif files:
            entries.append(
                (
                    "summary",
                    {
                        "count": len(files),
                        "size": sum(
                            int(row["size_bytes"])
                            for row in files
                        ),
                    },
                )
            )

        for index, (
            entry_type,
            value,
        ) in enumerate(entries):
            is_last = index == len(entries) - 1
            connector = (
                "└── "
                if is_last
                else "├── "
            )

            if entry_type == "directory":
                child = str(value)
                row = directories[child]

                lines.append(
                    f"{prefix}{connector}"
                    f"{Path(child).name}/ "
                    f"[{row['file_count']} fichiers, "
                    f"{row['size_human']}]"
                )

                render(
                    child,
                    prefix
                    + (
                        "    "
                        if is_last
                        else "│   "
                    ),
                )

            elif entry_type == "file":
                row = value

                lines.append(
                    f"{prefix}{connector}"
                    f"{row['filename']} "
                    f"[{row['size_human']}]"
                )

            else:
                summary = value

                lines.append(
                    f"{prefix}{connector}"
                    f"… {summary['count']} fichiers "
                    f"[{human_size(summary['size'])}]"
                )

    render(".", "")

    return "\n".join(lines) + "\n"


def build_summary(
    file_rows: list[dict[str, object]],
    directory_rows: list[dict[str, object]],
    script_rows: list[dict[str, object]],
    duplicate_rows: list[dict[str, object]],
    cleanup_rows: list[dict[str, object]],
    tracked_files: set[str],
) -> str:
    total_size = sum(
        int(row["size_bytes"])
        for row in file_rows
    )

    category_counts = Counter(
        str(row["category"])
        for row in file_rows
    )

    category_sizes: dict[str, int] = defaultdict(int)

    for row in file_rows:
        category_sizes[
            str(row["category"])
        ] += int(row["size_bytes"])

    top_directories = sorted(
        (
            row
            for row in directory_rows
            if str(row["path"]) != "."
        ),
        key=lambda row: int(
            row["size_bytes"]
        ),
        reverse=True,
    )[:20]

    duplicate_groups = len(
        {
            str(row["duplicate_group"])
            for row in duplicate_rows
        }
    )

    lines = [
        "AUDIT DU PROJET",
        "=" * 80,
        f"Racine : {ROOT}",
        f"Date : {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Résumé général",
        "-" * 80,
        f"Fichiers analysés : {len(file_rows)}",
        f"Taille analysée : {human_size(total_size)}",
        f"Scripts Python : {len(script_rows)}",
        f"Fichiers suivis par Git présents : "
        f"{sum(1 for row in file_rows if str(row['path']) in tracked_files)}",
        f"Fichiers non suivis par Git présents : "
        f"{sum(1 for row in file_rows if str(row['path']) not in tracked_files)}",
        f"Groupes de doublons exacts : {duplicate_groups}",
        f"Candidats de nettoyage : {len(cleanup_rows)}",
        "",
        "Répartition par catégorie",
        "-" * 80,
    ]

    for category, count in category_counts.most_common():
        lines.append(
            f"- {category:<24} : "
            f"{count:>6} fichiers — "
            f"{human_size(category_sizes[category])}"
        )

    lines.extend(
        [
            "",
            "20 répertoires les plus volumineux",
            "-" * 80,
        ]
    )

    for row in top_directories:
        lines.append(
            f"- {str(row['path']):<70} "
            f"{str(row['size_human']):>12} "
            f"({row['file_count']} fichiers)"
        )

    lines.extend(
        [
            "",
            "Fichiers produits",
            "-" * 80,
            "- tree_compact.txt : arborescence lisible et résumée",
            "- tree_full.txt : arborescence complète",
            "- files.csv : inventaire détaillé de chaque fichier",
            "- directories.csv : taille et contenu de chaque répertoire",
            "- scripts.csv : inventaire et dépendances des scripts Python",
            "- duplicate_files.csv : doublons exacts détectés",
            "- cleanup_candidates.csv : propositions sans suppression",
            "- git_status.txt : état Git actuel",
            "- report.json : version structurée du résumé",
            "",
            "Aucun fichier du projet n'a été supprimé ou modifié.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    if not (ROOT / ".git").exists():
        print(
            "Attention : aucun dossier .git détecté dans "
            f"{ROOT}",
            file=sys.stderr,
        )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracked_files, git_status = (
        load_git_information()
    )

    print("Analyse de l'arborescence...")

    file_rows, transient_rows = (
        walk_project()
    )

    for row in file_rows:
        relative_path = str(row["path"])

        row["git_tracked"] = int(
            relative_path in tracked_files
        )

    directory_rows = build_directory_rows(
        file_rows=file_rows,
        tracked_files=tracked_files,
    )

    print("Analyse des scripts Python...")

    script_rows = build_script_rows(
        file_rows=file_rows,
        tracked_files=tracked_files,
    )

    print("Recherche des doublons exacts...")

    duplicate_rows = find_duplicates(
        file_rows
    )

    cleanup_rows = build_cleanup_candidates(
        file_rows=file_rows,
        transient_rows=transient_rows,
    )

    write_csv(
        AUDIT_DIR / "files.csv",
        file_rows,
        [
            "path",
            "directory",
            "filename",
            "extension",
            "size_bytes",
            "size_human",
            "modified",
            "category",
            "git_tracked",
        ],
    )

    write_csv(
        AUDIT_DIR / "directories.csv",
        directory_rows,
        [
            "path",
            "file_count",
            "size_bytes",
            "size_human",
            "tracked_file_count",
            "untracked_file_count",
            "dominant_category",
            "category_distribution",
        ],
    )

    write_csv(
        AUDIT_DIR / "scripts.csv",
        script_rows,
        [
            "path",
            "tracked",
            "size_bytes",
            "size_human",
            "has_main_function",
            "function_count",
            "class_count",
            "imports",
            "path_literals",
            "syntax_error",
        ],
    )

    write_csv(
        AUDIT_DIR / "duplicate_files.csv",
        duplicate_rows,
        [
            "duplicate_group",
            "sha256",
            "size_bytes",
            "size_human",
            "path",
            "category",
        ],
    )

    write_csv(
        AUDIT_DIR / "cleanup_candidates.csv",
        cleanup_rows,
        [
            "path",
            "type",
            "size_bytes",
            "size_human",
            "cleanup_level",
            "reason",
        ],
    )

    compact_tree = build_compact_tree(
        directory_rows=directory_rows,
        file_rows=file_rows,
    )

    full_tree = build_full_tree(
        file_rows=file_rows,
    )

    (AUDIT_DIR / "tree_compact.txt").write_text(
        compact_tree,
        encoding="utf-8",
    )

    (AUDIT_DIR / "tree_full.txt").write_text(
        full_tree,
        encoding="utf-8",
    )

    (AUDIT_DIR / "git_status.txt").write_text(
        git_status,
        encoding="utf-8",
    )

    summary = build_summary(
        file_rows=file_rows,
        directory_rows=directory_rows,
        script_rows=script_rows,
        duplicate_rows=duplicate_rows,
        cleanup_rows=cleanup_rows,
        tracked_files=tracked_files,
    )

    (AUDIT_DIR / "summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    report = {
        "root": str(ROOT),
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "file_count": len(file_rows),
        "total_size_bytes": sum(
            int(row["size_bytes"])
            for row in file_rows
        ),
        "python_script_count": len(script_rows),
        "duplicate_group_count": len(
            {
                str(row["duplicate_group"])
                for row in duplicate_rows
            }
        ),
        "cleanup_candidate_count": len(
            cleanup_rows
        ),
        "audit_directory": str(AUDIT_DIR),
        "archive": str(ZIP_OUTPUT),
    }

    (AUDIT_DIR / "report.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if ZIP_OUTPUT.exists():
        ZIP_OUTPUT.unlink()

    with ZipFile(
        ZIP_OUTPUT,
        "w",
        ZIP_DEFLATED,
    ) as archive:
        for path in sorted(
            AUDIT_DIR.rglob("*")
        ):
            if path.is_file():
                archive.write(
                    path,
                    path.relative_to(ROOT),
                )

    print()
    print(summary)
    print(f"Rapport : {AUDIT_DIR}")
    print(f"Archive à transmettre : {ZIP_OUTPUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
