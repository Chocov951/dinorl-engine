# Audit RL-S6 v2 — correction de mesure

## Périmètre immuable

`artifacts/rl/s6/rl-s6-v1/campaign-result.json` reste la décision historique
`FAILED_RL_S6` (SHA-256 `9f5938adce727fa75da129c1559dc189e1711b1e24662007505eaf6a84ac53ba`).
RL-S6 v2 ne régénère ni PPO, ni checkpoint, ni poids ; il mesure seulement les cinq
candidats v1 existants : 19/u110, 20/u90, 21/u125, 22/u95 et 23/u105.

## Déséquilibre établi

L'évaluation déterministe v1 utilisait quatre parties par bot scripté, soit une
résolution de score de 0,25, mais la publication stochastique en utilisait 400 par
adversaire. Les deux modes utilisaient également des seeds et des positions différents.
Les échecs 19 et 22 sont précisément dus au seuil de chute contre `prudent-v1`
(0,24375 et 0,16375), calculé depuis le score déterministe discret. Il est donc
impossible d'interpréter ce seul écart comme une chute stochastique mesurée à précision
égale. Le verdict v1 n'est pas réécrit : le diagnostic justifie uniquement une mesure
corrective séparée.

## Protocole v2 figé

Pour chaque seed et chaque adversaire, v2 exécute 200 confrontations × 4 positions :
`(A,A)`, `(A,B)`, `(B,A)`, `(B,B)` pour (côté apprenant, premier joueur). Chaque
confrontation a la même carte et le même `pair_id` dans les modes déterministe et
stochastique. Cela représente 800 parties par mode et par adversaire, 6 400 par seed.
Les observations sont agrégées par confrontation puis comparées par bootstrap apparié
reproductible (10 000 réplications, seed 20260924 + seed d'entraînement, IC à 95 %).

La première mesure est acceptée seulement si les bornes favorables de tous les seuils
absolus et de chute passent. Elle est rejetée seulement si les bornes contraires
établissent un échec. Tout autre résultat reçoit exactement une extension identique,
avec répétitions 200–399, avant une unique décision. Le gate reste 4 seeds sur 5 :
random ≥ 0,85 ; moyenne des trois bots > 0,55 ; chaque bot > 0,40 ; chute moyenne
≤ 0,10 ; chute individuelle ≤ 0,15.

Les empreintes du code du protocole, de l'évaluation, des bots, de la configuration,
de la campagne v1 et des cinq checkpoints sont écrites dans le manifeste v2. Les
métriques de modes de victoire et `ROUND_LIMIT` sont conservées dans chaque record.
