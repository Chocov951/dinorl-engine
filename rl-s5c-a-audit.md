# Audit RL-S5c-A — calibration des récompenses

**Décision globale : NO-GO.** RL-S5c-B n'est pas lancé.

Cet audit a lu les artefacts bruts de `artifacts/rl/s5c/rl-s5c-calibration-v1/` : manifest, 1 167 snapshots de récupération, états, cycles, évaluations et ledger.

## Périmètre et intégrité

| Contrôle | Constat | Verdict |
|---|---|---|
| Campagne | `scavenger`, `predator` et `controller`, chacun pour les seeds 19, 20 et 21 | Conforme |
| Architecture et limite | Manifest : `mlp-compact-v2`, 30 manches, budget 147 | Conforme |
| Opposant d'entraînement | Les 1 167 états de récupération déclarent uniquement `random-legal-v1` | Conforme sur les traces disponibles |
| Parent / reprise | Chaque état déclare `parent_checkpoint: null`; aucun pool de ligue ou de self-play n'est déclaré | Conforme déclaré, mais provenance initiale incomplète |
| Snapshots | Les hashes de chaque fichier référencé par les 1 167 manifests de récupération ont été recalculés : 0 divergence, 0 fichier manquant | Conforme |
| Arrêts | 75, 95 et 115 unités pour `scavenger`; 147 unités pour les six autres seeds | Voir gate |
| Rôles d'entraînement | Comptages réalisés archivés par cycle et par run; global : premier 20 911, second 20 274, A 20 599, B 20 586 | Conforme (écarts 1,55 % et 0,03 %) |
| Versions | Commit enregistré : `e02670f23d371aa37595fb3790382529a104a771` | Non reproductible en l'état : ni hash des versions moteur/règles/observation/actions/PPO, ni hash des poids initiaux; le worktree contenant le code RL-S5c est modifié et comporte des fichiers non suivis, que le seul hash Git ne décrit pas |

Les mêmes trois seeds sont bien passées au constructeur PPO pour les trois archétypes. En revanche, les artefacts ne contiennent pas le hash des poids **avant** la première unité : l'égalité réelle des initialisations communes et l'absence de poids préentraînés ne peuvent donc pas être prouvées a posteriori. Les `ppo_state.npz` et les snapshots post-unité ne remplacent pas cette preuve.

La configuration fige trois seeds d'évaluation (`101`, `102`, `103`), mais RL-S5c-A n'a exécuté que `101` : l'appel de calibration utilise `evaluation_seeds[0]`. Les résultats ne sont donc pas une mesure sur les trois seeds annoncées. Les sorties agrégées d'évaluation ne ventilent pas non plus premier/second ni A/B.

Enfin, le flux de politique de `run_evaluation_game` est encore dérivé de la graine et de l'adversaire, sans `game_id` persistant dans `EvaluationGameSpec`; `derive_game_stream_seed()` n'est pas employé sur ce chemin. La garantie de préfixes RNG identiques pour des contrefactuels n'est pas démontrée. C'est une violation de reproductibilité bloquante.

## Gate de compétence et arrêt

Les évaluations sont présentes tous les cinq unités. Les six seeds qui n'ont pas passé le gate possèdent un snapshot final à l'unité 147 : elles n'ont pas été prolongées. Leur état final reste néanmoins `training`, plutôt qu'un état explicite `failed_budget`.

| Archétype | Seed | Deux passages de seuil | `first_stable_gate` | Dernière unité | Gate |
|---|---:|---|---:|---:|---|
| scavenger | 19 | 70, 75 | 75 | 75 | Conforme : arrêt à la seconde réussite |
| scavenger | 20 | 90, 95 | 95 | 95 | Conforme : arrêt à la seconde réussite |
| scavenger | 21 | 110, 115 | 115 | 115 | Conforme : arrêt à la seconde réussite |
| predator | 19 | aucune | — | 147 | Échec budget |
| predator | 20 | aucune | — | 147 | Échec budget |
| predator | 21 | aucune | — | 147 | Échec budget |
| controller | 19 | aucune | — | 147 | Échec budget |
| controller | 20 | aucune | — | 147 | Échec budget |
| controller | 21 | aucune | — | 147 | Échec budget |

À leur dernier contrôle (unité 145), les six échecs ne satisfont pas tous les seuils :

| Archétype | Seed | Random-legal | Moyenne déterministe | Scores agressif / opportuniste / prudent |
|---|---:|---:|---:|---|
| predator | 19 | 93,38 % | 50,00 % | 100,00 % / 50,00 % / 0,00 % |
| predator | 20 | 92,63 % | 50,00 % | 100,00 % / 50,00 % / 0,00 % |
| predator | 21 | 89,13 % | 50,00 % | 100,00 % / 50,00 % / 0,00 % |
| controller | 19 | 83,13 % | 50,00 % | 100,00 % / 50,00 % / 0,00 % |
| controller | 20 | 80,63 % | 50,00 % | 100,00 % / 50,00 % / 0,00 % |
| controller | 21 | 94,50 % | 58,33 % | 100,00 % / 50,00 % / 25,00 % |

## Télémetrie de style disponible

La seule télémétrie disponible est celle de la suite de gate déterministe (412 parties par checkpoint : 400 contre `random-legal` et 4 par bot déterministe). Elle fournit des raisons terminales globales, mais ne les joint pas à l'issue du joueur apprenant. Elle ne permet donc pas de calculer les signatures **conditionnées à la victoire**, ni les bousculades « utiles ».

| Archétype | Seed | Checkpoint | W / N / D | `ROUND_LIMIT` | Signal brut sans conditionnement |
|---|---:|---:|---:|---:|---|
| scavenger | 19 | 75 | 404 / 8 / 0 | 1,9 % | 1 210 points de carcasse; 1 214 nutritions validées; 399 fins aux points et 5 KO |
| scavenger | 20 | 95 | 403 / 9 / 0 | 2,2 % | 1 200 points; 1 205 nutritions validées; 396 fins aux points et 7 KO |
| scavenger | 21 | 115 | 402 / 10 / 0 | 2,4 % | 1 210 points; 1 213 nutritions validées; 398 fins aux points et 4 KO |
| predator | 19 | 145 | 359 / 41 / 12 | 10,0 % | 5,58 dégâts infligés/partie; 367 fins KO; 4 fins aux points |
| predator | 20 | 145 | 349 / 55 / 8 | 13,3 % | 5,50 dégâts/partie; 353 KO; 4 fins aux points |
| predator | 21 | 145 | 320 / 85 / 7 | 20,6 % | 5,12 dégâts/partie; 322 KO; 5 fins aux points |
| controller | 19 | 145 | 271 / 135 / 6 | 32,8 % | 0 bousculade tentée, réussie, mur ou boue |
| controller | 20 | 145 | 253 / 151 / 8 | 36,7 % | 0 bousculade tentée, réussie, mur ou boue |
| controller | 21 | 145 | 361 / 48 / 3 | 11,7 % | 0 bousculade tentée, réussie, mur ou boue |

Les trois `scavenger` satisfont le gate et présentent un signal brut de collecte net. Comme ils ne perdent aucune partie de cette suite, les routes observées permettent d'inférer 399/404 (98,8 %), 396/403 (98,3 %) et 398/402 (99,0 %) de victoires aux points, respectivement. Cela reste insuffisant pour geler le DSL : la suite de style séparée, les agrégats par rôle et la force contre RL-S5b sont absents.

`predator` inflige davantage de dégâts que `scavenger` dans cette télémétrie, mais ne passe aucun gate; les données ne permettent pas d'attribuer ses KO à ses seules victoires. Le taux `ROUND_LIMIT` dépasse 10 % pour les seeds 20 et 21, sans replay à inspecter.

`controller` ne présente aucune bousculade dans les 1 236 parties de ses trois dernières évaluations. Son identité est opposée au critère attendu et elle est invalidée indépendamment de son échec de gate.

## Reward hacking, force et réglementation

Les sources DSL sont statiquement événementielles : `terminal_score`, dégâts, points de carcasse, début de nutrition et effets de bousculade. Cette lecture ne démontre toutefois pas leur effet observé.

| Contrôle requis | Artefact trouvé | Conclusion |
|---|---|---|
| Retour terminal / auxiliaire par W/N/D | Aucun | Non vérifiable |
| Cas auxiliaires extrêmes et replays | 0 replay, 0 retour enregistré | Non vérifiable |
| Boucles d'événements / actions sans effet | Aucun journal de transition ou replay | Non vérifiable |
| Pool fort RL-S5b, score 25–50 % | Aucun | Non mesuré |
| W/N/D, causes, durée, rôles et A/B contre RL-S5b | Aucun | Non mesuré |
| Inspection des `ROUND_LIMIT` > 10 % | Aucun replay | Impossible |

L'absence de ces données est bloquante : elle interdit de conclure à une identité non temporisatrice, à l'absence de reward hacking et à une bande de force acceptable.

## Décision par archétype

| Archétype | Décision | Motifs bloquants |
|---|---|---|
| scavenger | **NO-GO** | Gate et signal brut favorables, mais provenance non rejouable, style conditionné, retours, replays et force RL-S5b absents |
| predator | **NO-GO** | 0/3 gate au budget 147; données de style insuffisantes; `ROUND_LIMIT` > 10 % sur 2/3 seeds sans replay |
| controller | **NO-GO** | 0/3 gate au budget 147; 0 bousculade sur les trois évaluations finales, donc style absent; données de validation manquantes |

La décision globale est donc **NO-GO**. Aucun manifeste de production n'est gelé, aucun checkpoint de calibration n'est réutilisable pour une production, et **RL-S5c-B ne doit pas être lancé**.

## Révision minimale proposée et prochaine calibration

Ce ne sont que des candidats de calibration : ils ne sont pas appliqués dans ce dépôt et ne constituent pas un gel de production.

| DSL | Révision minimale proposée | Justification |
|---|---|---|
| scavenger | Inchangé | Le signal collecte est déjà net; aucune mesure ne justifie de modifier son coefficient |
| predator | dégâts infligés `+0,10 → +0,12`; dégâts reçus `-0,03 → -0,05` | Rend le poids de dégâts effectivement supérieur à celui de `scavenger` et restaure une pénalisation de survie sans introduire de récompense de poursuite |
| controller | bousculade utile `+0,08 → +0,12` | Cible uniquement l'événement qui n'apparaît jamais, sans récompenser une bousculade sans effet |

La prochaine calibration doit repartir de zéro sur les trois archétypes et les mêmes seeds 19, 20 et 21, sans changement de moteur, règles, architecture ou PPO. Avant son exécution, elle doit également sceller un worktree propre (ou un hash de contenu complet), les poids initiaux, l'optimiseur neuf, les trois seeds d'évaluation effectivement employées, les rôles d'évaluation et les flux RNG indexés par `game_id`. Elle doit produire les retours terminal/auxiliaire, les événements nécessaires aux signatures avec et sans conditionnement sur la victoire, des replays ciblés et l'évaluation appariée contre le pool fort RL-S5b.

## Compte rendu de ticket

```text
Ticket : Audit serveur RL-S5c-A et décision de passage vers RL-S5c-B.
Fichiers modifiés : rl-s5c-a-audit.md.
Test rouge observé : sans objet — audit d'artefacts existants, aucune implémentation modifiée.
Commandes exécutées : inspection des manifests/configurations/états/cycles/évaluations; recalcul SHA-256 des 1 167 snapshots; contrôle des rôles, du gate et de l'absence des catégories d'artefacts requises.
Résultats : hashes intègres; scavenger 3/3 gate; predator 0/3; controller 0/3 et zéro bousculade; validation finale incomplète; NO-GO.
Couverture concernée : sans objet.
Décisions ou hypothèses : aucune mesure de performance supplémentaire n'a été lancée; les raisons terminales ne sont pas assimilées à une route de victoire sans jointure issue/route.
Risques ou travail restant : fiabiliser la provenance et les flux RNG, enregistrer les métriques/replays manquants, recalibrer depuis zéro, puis refaire cet audit avant toute production.
```
