# Arbitrage OCR des statistiques — V4

## Données de calibration

- Décisions humaines analysées : **366**
- V1 correcte : **251**
- V2 correcte : **28**
- Ni V1 ni V2 correcte : **87**

## Résultat rétrospectif

- Cas supplémentaires résolus automatiquement : **214**
- Choix V1 : **206**
- Choix V2 : **8**
- Décisions correctes sur l'historique : **214**
- Erreurs rétrospectives : **0**
- Précision rétrospective observée : **100,00 %**
- Réduction des revues historiques : **58,47 %**

### Par lot

| Lot | Cas historiques | Résolus par V4 | Corrects |
|---|---:|---:|---:|
| hero_batch_002 | 46 | 21 | 21 |
| hero_batch_003 | 57 | 45 | 45 |
| hero_batch_004 | 263 | 148 | 148 |
| **Total** | **366** | **214** | **214** |

## Simulation sur le batch 004

Avec les mêmes 20 000 valeurs :

- décisions automatiques V3 : **19 737**
- décisions automatiques V4 : **19 885**
- valeurs à revoir V3 : **263**
- valeurs à revoir V4 : **115**
- taux de revue : **1,315 % → 0,575 %**

## Principe

La V4 conserve les décisions sûres de la V3 et ajoute des règles
calibrées sur les lectures déjà vérifiées. Les nouvelles décisions sont
traçables avec :

- `arbitration_reason`
- `calibration_rule`

Le format de sortie reste compatible avec le dossier `stat_ocr_v3` et
avec le finaliseur générique existant.

## Limite

La précision de 100 % est une mesure rétrospective sur les batches 002 à
004. Elle ne garantit pas qu'aucune erreur n'apparaisse sur de nouvelles
captures. Les cas ne satisfaisant pas exactement les règles restent en
revue humaine.
