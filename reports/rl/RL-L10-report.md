# RL-L10 — calibration RL-S7

État : `ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S7`.

La campagne n'a pas été lancée localement. Le protocole canonique est dans
`configs/rl/s7.json` : 300 unités, seeds 20/21/23, seuil visible 0,40,
seuil de gate 0,55 et delta de plateau 0,01.

Le preflight vérifie le résultat immuable `RL_S6_V3_PASSED`, les hashes des
checkpoints sélectionnés, l'architecture `mlp-compact-v2` et le fichier de
configuration RL-S6. Il crée ensuite, dans un nouveau répertoire RL-S7, un état
PPO de reprise indépendant pour chaque seed : le répertoire RL-S6 n'est jamais
modifié. La reprise est idempotente au niveau de l'unité via `AtomicRecoveryStore`
et `UnitLedger`.

Mesure appliquée à chaque évaluation (toutes les 5 unités) : moyenne des scores
déterministes contre `aggressive-v1`, `prudent-v1` et `opportunist-v1`.

- `T_visible` : première unité de deux évaluations successives à au moins 0,40 ;
- `T_gate` : première unité à au moins 0,55 ;
- `T_plateau` : première unité après laquelle trois améliorations successives sont
  chacune au plus 0,01.

Après import des résultats serveur, `s7 analyze` écrit un rapport de capacité
distinct (`capacity-report.json`) et propose `D` seulement s'il satisfait pour
toutes les seeds `max(T_visible) <= D < min(T_plateau) / 2`. Les valeurs
provisoires ci-dessus sont alors remplacées dans ce nouveau rapport, sans
réécrire les décisions RL-S6 antérieures.

## Reprise opérationnelle

Relancer exactement la même commande `s7 run --resume` pour une seed interrompue.
Elle restaure le dernier état PPO validé, y compris RNG et état vectoriel. Ne pas
supprimer les dossiers `recovery`, `ledger.sqlite3` ou `curve.json`; ils forment
l'évidence de reprise et de calibration.
