# RL-S6 V3 — décision sans nouvelle mesure

V3 est une agrégation versionnée des intervalles V2 déjà produits. Elle ne charge aucun
PPO, ne régénère aucun poids et n'exécute aucune partie. Les décisions historiques
`FAILED_RL_S6` (V1) et `FAILED_RL_S6_V2` restent immuables.

## Justification

Les bots scriptés ont des trajectoires déterministes ; les répétitions V2 ne créent donc
pas une population déterministe indépendante pour la chute individuelle. Cette chute
compare principalement argmax et politique échantillonnée, alors que le déploiement
utilise la seconde. Elle demeure un signal diagnostique mais ne constitue pas une preuve
d'effondrement lorsque les scores stochastiques absolus passent leurs IC.

## Gate V3

Pour chaque seed, les bornes IC95 V2 doivent satisfaire : random ≥ 0,85 ; moyenne des
trois bots > 0,55 ; chaque bot > 0,40 ; borne haute de chute moyenne ≤ 0,10. La
publication requiert quatre seeds conformes sur cinq. Les chutes individuelles restent
calculées : un point > 0,15 ou un IC traversant 0,15 produit un avertissement explicite,
jamais un `failed_criteria`.

V3 source et hache V1, le manifeste V2, le résultat V2, chaque résultat V2 par seed et
chaque fichier de records V2. Elle est donc entièrement traçable sans modifier ces
sources.
