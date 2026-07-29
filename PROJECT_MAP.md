# Carte du projet Hero Wars Vision

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
