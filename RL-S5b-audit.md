# Décision RL-S5b

## Décision validée

- `mlp-compact-v2` est l’architecture officielle V1.
- La limite réglementaire reste de 30 manches. Moins de 1 % des parties des
  agents unité 197 l’atteignent ; les parties limitées sont principalement
  passives.
- Le self-play fonctionne mais n’a pas démontré d’avantage sur le pool fixe.
- Les exploiters S5b ne produisent aucune victoire : leur comportement de nul
  est classé `temporisatrice`, non contre-stratégie gagnante.
- Les matchs officiels restent aller-retour, avec même graine et inversion du
  premier joueur, car le biais premier/second est mesuré.

## Corrections de traçabilité appliquées

- Le plan complet d’audit compte 43 200 parties : 180 tâches × 3 seeds × 5
  confrontations × 4 positions × 4 limites.
- Les snapshots de self-play conservent leur unité physique réelle (`147…197`)
  et déclarent séparément leur catégorie historique.
- Les évaluations contrefactuelles dérivent les flux de tirage des politiques à
  partir du `game_id`, indépendamment de la limite ou des options de rapport.
- Les compteurs effectivement tirés de premier/second et de côté A/B sont
  sauvegardés par cycle et cumulés dans les artefacts de calibration.

## Protocole des futurs exploiters

Chaque exploiter part du même checkpoint généraliste compétent, avec un
optimiseur neuf. Il s’entraîne contre une cible gelée et vérifiée par hash,
avec les mêmes seeds et budgets pour toutes les cibles. Les rapports séparent
victoires, nuls et défaites ; au moins une victoire est nécessaire pour
qualifier une contre-stratégie gagnante. Une politique qui ne produit que des
nuls est `temporisatrice`. Toute adaptation d’une cible invalide les exploiters
vus : une nouvelle mesure exige des exploiters neufs depuis la même base.
