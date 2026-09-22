# RL-S5c-A3 — décision de récompenses

## Scavenger

`scavenger-a2-control` conserve exactement le Reward DSL A2 et reste exclusivement contre
`random-legal` jusqu'à 197 unités, sans arrêt sur le gate historique.

`scavenger-a3-curriculum` conserve la validation effective d'un point et exclut présence,
tentative interrompue et distance continue. `safe_feed` vaut 0,03 par opportunité de
ressource, au plus une fois par opportunité, avec un plafond explicite de **0,30 par
épisode**.

## Predator / Agressif

`predator-a2-control` conserve exactement le Reward DSL A2 avec le curriculum A3.

`predator-a3-balanced` conserve le terminal `+1/-1/0`, réduit les dégâts infligés de 0,10
à **0,06 par PV**, pénalise les dégâts reçus de **0,04 par PV**, et accorde **0,04** à une
interruption effective de nutrition adverse. Cette réduction empêche les dégâts de dominer
presque tout le retour auxiliaire. Aucune proximité, distance, approche ou morsure sans
effet n'est récompensée.
