from __future__ import annotations

from pathlib import Path

import match_batch_avatars_to_catalog as matcher


# Nouvelles découpes obtenues par détection dynamique des cadres.
matcher.QUERY_AVATAR_DIR = Path(
    "data/batches/hero_batch_001/"
    "crops_dynamic_v1/avatars_inner"
)

# Nouveau dossier pour ne pas mélanger ces résultats
# avec ceux issus des anciennes découpes incorrectes.
matcher.OUTPUT_DIR = Path(
    "data/batches/hero_batch_001/"
    "reports/visual_matching_dynamic_v1"
)

matcher.RESULTS_CSV = (
    matcher.OUTPUT_DIR
    / "visual_match_results.csv"
)

matcher.HERO_COUNTS_CSV = (
    matcher.OUTPUT_DIR
    / "predicted_hero_counts.csv"
)

matcher.HTML_REPORT = (
    matcher.OUTPUT_DIR
    / "visual_match_review.html"
)


if __name__ == "__main__":
    raise SystemExit(matcher.main())