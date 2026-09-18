# RL-S5b — Comparaison robuste, reprise, self-play et exploiters

## 1. Objet

Ce ticket prolonge `RL-S5` lorsque l'évaluation contre `random-legal` et les bots scriptés est saturée et ne permet plus de sélectionner une architecture.

Il doit ajouter, en restant test-driven :

- la reprise exacte depuis les checkpoints produits par `RL-S5` ;
- un pool d'évaluation figé et traçable ;
- une évaluation croisée complète des politiques ;
- une continuation d'entraînement à budget identique contre des adversaires communs ;
- un mode de self-play avec historique ;
- l'entraînement d'exploiters contre une politique figée ;
- un rapport permettant de choisir l'architecture la plus robuste, et pas seulement la première à battre les bots simples.

Ce ticket ne modifie ni les règles du jeu, ni l'observation, ni les neuf actions, ni le Reward DSL, ni l'intégration MyBlog. Il ne remplace pas `RL-S6` : il détermine le candidat qui y sera envoyé.

## 2. Entrées de référence

Le comportement par défaut utilise les checkpoints finaux de `RL-S5` :

- architectures : `mlp-v1`, `mlp-compact-v2`, `mlp-balanced-v2`, `mlp-deep-v2` ;
- seeds d'entraînement : `19`, `20`, `21` ;
- checkpoint à `147` unités, soit `301 056` transitions apprenant ;
- hyperparamètres PPO et programme de récompense inchangés.

Les architectures, seeds et chemins restent configurables. Aucune politique ne doit être codée en dur dans le moteur d'évaluation.

Pour le prolongement coûteux, les finalistes par défaut sont `mlp-compact-v2` et `mlp-deep-v2`. `mlp-v1` reste un contrôle ; `mlp-balanced-v2` reste dans le benchmark croisé, mais n'est pas prolongé par défaut. Ce choix doit pouvoir être changé dans la configuration sans modifier le code.

## 3. Principes non négociables

1. Le pool de benchmark est immuable pendant toute une campagne `RL-S5b`.
2. Le pool de benchmark, le pool d'entraînement et la ligue de self-play sont trois objets distincts.
3. Une politique évaluée ne met jamais à jour ses poids, son optimiseur, ses normalisations ou ses RNG.
4. Toute partie officielle d'évaluation est appariée : même graine et même configuration, en deux manches avec inversion du premier joueur.
5. Les performances sont conservées par checkpoint et par seed. Une moyenne globale ne doit jamais masquer l'effondrement d'une seed.
6. Les mesures de performance, de débit et de durée destinées à une décision sont exécutées sur le serveur. Le local sert aux tests et aux smoke tests.
7. Les poids, optimiseurs et grosses archives ne sont pas commités. Les manifestes, résultats JSON légers, rapports et décisions le sont.

## 4. Reprise exacte d'un checkpoint

Ajouter une commande de continuation acceptant un checkpoint parent et un nombre d'unités supplémentaires.

Une reprise exacte restaure au minimum :

- poids de politique et de valeur ;
- état de l'optimiseur et éventuel scheduler ;
- compteurs d'unités, transitions, epochs et pas d'optimisation ;
- RNG Python, NumPy, PyTorch, environnements et sélection d'adversaires ;
- états des environnements vectorisés et épisodes inachevés ;
- programme de récompense compilé et son hash ;
- versions du moteur, des règles, de l'observation et des actions ;
- configuration PPO, architecture et manifeste du pool d'adversaires.

La commande échoue explicitement en cas d'incompatibilité. Si seul le modèle peut être chargé, l'opération doit être nommée `warm-start` et ne doit jamais être présentée comme une reprise exacte ni servir de preuve `RL-S5b`.

Test d'acceptation : interrompre une exécution à une frontière de checkpoint, la reprendre, puis vérifier que les transitions, poids et métriques obtenus sont identiques à une exécution non interrompue de même seed.

## 5. Pool de benchmark figé

Créer un manifeste versionné `s5b-benchmark-pool-v1` contenant :

- `random-legal` ;
- tous les bots scriptés déjà présents dans la suite déterministe ;
- les douze checkpoints finaux `RL-S5` définis à la section 2 ;
- les checkpoints historiques aux unités `32`, `64`, `96` et `147` lorsqu'ils existent déjà.

Un historique absent n'est pas régénéré : il est déclaré `missing` dans le manifeste. Le gel du pool calcule un SHA-256 sur le manifeste canonique et sur chaque artefact de politique. Toute modification crée une nouvelle version de pool et interdit de comparer directement ses résultats à la campagne précédente.

Chaque entrée indique : identifiant, type d'adversaire, architecture, seed, unité, hash, versions de compatibilité et provenance.

Le benchmark ne reçoit jamais automatiquement un nouveau snapshot issu du self-play ou un exploiter. Une nouvelle campagne peut décider de les ajouter en créant `s5b-benchmark-pool-v2`.

## 6. Évaluation croisée

### 6.1 Matrice

Évaluer chaque checkpoint final contre tous les autres checkpoints finaux, y compris entre seeds différentes. Avec douze politiques, la matrice attendue est de `12 × 12` et couvre les 66 paires non ordonnées ; la diagonale est également mesurée ou explicitement définie comme auto-jeu.

Réutiliser l'évaluateur apparié existant :

- mode stochastique comme mesure principale ;
- mode déterministe comme diagnostic seulement ;
- 100 confrontations par paire par défaut ;
- les quatre configurations de position existantes ;
- deux manches aller-retour avec la même graine et inversion du premier joueur.

Le nombre exact de manches réellement joué doit apparaître dans le résultat afin d'éviter toute ambiguïté entre confrontation, match et manche.

### 6.2 Mesures obligatoires

Pour chaque paire, seed et architecture :

- victoires, nuls, défaites et score échecs moyen ;
- intervalle de confiance à 95 % sur le score apparié ;
- score par rôle de premier et second joueur ;
- écart lié au premier joueur ;
- actions et tours moyens ;
- résultats par configuration de position ;
- matrice brute par checkpoint ;
- agrégats d'architecture : moyenne, médiane, IQR, pire seed et moyenne du quartile inférieur.

Produire au minimum `crossplay.json`, `crossplay.csv` et `crossplay-report.md`. Le rapport doit rendre visibles les cycles éventuels du type `A > B > C > A` ; un classement Elo seul est insuffisant.

## 7. Continuation commune à budget fixé

Avant le self-play, comparer la capacité d'apprentissage des finalistes dans une expérience contrôlée.

À partir de chaque checkpoint à 147 unités :

1. prolonger chaque finaliste de 50 unités ;
2. utiliser exactement le même pool d'entraînement figé, les mêmes poids de tirage et les mêmes seeds de sélection d'adversaire ;
3. évaluer tous les 10 unités contre le benchmark figé ;
4. prolonger de 97 unités supplémentaires uniquement si les intervalles restent trop proches ou si les courbes montent encore ;
5. si une prolongation est décidée, l'appliquer à tous les finalistes encore comparés.

Pool d'entraînement commun par défaut :

| Catégorie | Poids |
| --- | ---: |
| `random-legal` | 10 % |
| bots scriptés | 20 % |
| checkpoints finaux `RL-S5` | 50 % |
| checkpoints historiques disponibles | 20 % |

Les adversaires d'une catégorie sont tirés uniformément en V1. Le pool est identique pour toutes les architectures ; il ne contient aucun snapshot créé pendant cette continuation.

Conserver un contrôle `random-only` pour au moins une seed et un budget de 50 unités. Il sert à vérifier si les gains proviennent réellement des adversaires plus forts plutôt que de la simple poursuite de l'entraînement.

## 8. Self-play avec historique

Le self-play est une expérience séparée de la continuation commune. Il doit être implémenté comme un mode configurable et ne doit pas modifier le benchmark figé.

### 8.1 Règles de la ligue

- Le learner ne joue jamais contre ses poids vivants.
- Son adversaire est toujours un snapshot immuable et validé.
- Créer un snapshot initial à l'unité 147, puis un snapshot tous les 10 unités.
- Chaque run et chaque seed possèdent leur propre historique afin de conserver l'indépendance statistique.
- Aucun snapshot issu d'une autre seed n'entre silencieusement dans la ligue.
- Conserver les snapshots historiques même lorsqu'ils deviennent faibles : ils limitent l'oubli et les cycles.

Distribution de départ :

| Adversaire | Avant disponibilité des exploiters | Après disponibilité des exploiters |
| --- | ---: | ---: |
| `random-legal` | 10 % | 10 % |
| bots scriptés | 20 % | 20 % |
| historique du learner | 50 % | 30 % |
| adversaires figés les plus difficiles du benchmark | 20 % | 20 % |
| exploiters validés | 0 % | 20 % |

Les poids sont configurables mais enregistrés dans chaque manifeste. Pour cette première version, le tirage est uniforme à l'intérieur d'une catégorie. Un futur échantillonnage prioritaire selon le taux de victoire est hors périmètre.

### 8.2 Comparaison

Depuis un même checkpoint parent, lancer au même budget :

- une branche contre le pool commun figé ;
- une branche self-play.

Comparer les deux branches uniquement contre le benchmark figé et par cross-play. Le rendement façonné d'entraînement n'est pas une métrique de victoire.

## 9. Exploiters

Un exploiter cherche une faiblesse d'une politique cible immuable. Il ne constitue pas une nouvelle référence absolue.

Pour chaque checkpoint finaliste à tester :

1. figer la cible et vérifier son hash ;
2. initialiser trois exploiters indépendants avec des seeds communes à toutes les cibles ;
3. employer une architecture d'exploiter unique et identique pour toutes les cibles, `mlp-compact-v2` par défaut ;
4. entraîner chaque exploiter 50 unités contre la cible, avec le programme de récompense de référence déjà validé ;
5. prolonger jusqu'à 100 unités seulement si la courbe progresse encore, et pour toutes les cibles comparées ;
6. évaluer l'exploiter et sa cible en matchs appariés officiels ;
7. conserver le meilleur score d'exploiter et le pire score de la cible.

Le succès est toujours mesuré par l'issue officielle de la partie, jamais par la somme de récompenses façonnées.

Un exploiter utilisé ensuite comme adversaire d'entraînement est considéré comme « vu ». Après cette adaptation, entraîner de nouveaux exploiters depuis zéro pour obtenir une nouvelle mesure d'exploitabilité. Ne jamais réutiliser le même exploiter à la fois comme exercice et comme examen final.

## 10. Règle de décision

Le gate historique contre `random-legal` et les bots scriptés reste un critère d'éligibilité, pas le critère de classement final.

Ordre de décision :

1. **veto de stabilité** : corruption, incompatibilité, NaN ou effondrement manifeste d'une seed ;
2. **robustesse** : pire seed, quartile inférieur et borne basse de l'intervalle contre le benchmark ;
3. **cross-play** : score global et absence de faiblesse systématique contre une famille ;
4. **exploitabilité** : score de la cible contre son meilleur exploiter ;
5. **efficacité d'apprentissage** : transitions jusqu'au gate et aire sous la courbe ;
6. **coût serveur** : temps mur, CPU, mémoire et inférence.

Si plusieurs architectures restent à moins de deux points de pourcentage du meilleur score robuste et qu'aucune n'est nettement plus exploitable, retenir la plus petite et la moins coûteuse.

Le self-play est rapporté séparément. Il ne départage les architectures que si elles ont suivi un protocole de ligue strictement identique ; sinon, il sert à valider le pipeline futur.

Aucune décision automatique n'est prise si les intervalles se chevauchent fortement, si les critères désignent des gagnants différents ou si une seed présente un comportement contradictoire. Le rapport doit alors demander une décision humaine.

## 11. Interfaces attendues

Intégrer ces fonctions au CLI existant. Les noms peuvent suivre les conventions déjà présentes, mais les capacités suivantes doivent être disponibles sans script ad hoc :

```bash
python -m dinorl_engine.rl s5b freeze-pool --config configs/rl/s5b.yaml
python -m dinorl_engine.rl s5b cross-evaluate --pool <pool-manifest> --resume
python -m dinorl_engine.rl s5b continue --checkpoint <id> --units 50 --opponents fixed
python -m dinorl_engine.rl s5b self-play --checkpoint <id> --units 50
python -m dinorl_engine.rl s5b train-exploiters --target <checkpoint-id> --units 50
python -m dinorl_engine.rl s5b report --run <run-id>
```

Chaque commande longue doit être reprenable, idempotente sur les tâches déjà terminées et produire un manifeste indiquant le commit Git, la configuration, les hashes, les seeds, la progression et les erreurs.

Arborescence logique attendue :

```text
artifacts/rl/s5b/<run_id>/
├── manifest.json
├── pools/
├── crossplay/
├── continuation/
├── selfplay/
├── exploiters/
└── report/
```

## 12. Développement test-driven

Écrire les tests avant le code pour vérifier au minimum :

- immutabilité et hash du pool ;
- refus d'une politique incompatible ou altérée ;
- couverture exacte de toutes les paires de la matrice ;
- inversion du premier joueur avec conservation de la graine ;
- absence de mutation pendant l'évaluation ;
- reprise exacte après interruption ;
- reproductibilité du tirage d'adversaires ;
- impossibilité de jouer contre des poids vivants en self-play ;
- ajout d'un snapshot historique uniquement à la frontière prévue ;
- séparation stricte benchmark / entraînement / ligue ;
- cible d'exploiter immuable ;
- exclusion d'un exploiter déjà vu de l'examen final ;
- agrégats conservant les résultats par seed ;
- reprise d'une matrice ou d'un entraînement partiellement terminé.

Les tests locaux utilisent de petits faux modèles, deux ou trois seeds de match et au plus deux unités d'entraînement. Ils valident le comportement, jamais la vitesse ni la qualité d'apprentissage.

## 13. Checkpoints serveur obligatoires

### RL-S5b-A — Cross-play

Après tests locaux et smoke complet, arrêter le développement avec :

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S5b-A
```

Le serveur gèle le pool, exécute la matrice croisée et produit une archive importable localement. Après import, Codex analyse les résultats et confirme ou modifie la liste des finalistes dans la configuration, sans modifier rétroactivement le pool.

### RL-S5b-B — Continuation contrôlée

Exécuter sur le serveur les 50 unités supplémentaires des finalistes, le contrôle `random-only` et les évaluations tous les 10 unités. Arrêter ensuite avec :

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S5b-B
```

La décision de prolonger jusqu'à 147 unités supplémentaires est prise uniquement après import des courbes.

### RL-S5b-C — Exploiters et self-play

Exécuter sur le serveur les exploiters et l'expérience self-play au budget retenu, puis arrêter avec :

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S5b-C
```

L'archive finale contient les résultats légers et le rapport, mais pas les poids. Après import, produire `architecture-decision-s5b.md` et ne lancer `RL-S6` qu'après validation humaine de cette décision.

## 14. Critères de fin

`RL-S5b` est terminé lorsque :

- une exécution interrompue peut reprendre exactement ;
- le benchmark possède un manifeste immuable et vérifiable ;
- la matrice croisée complète est reproductible ;
- continuation fixe et self-play sont comparables sans contamination du benchmark ;
- les exploiters sont entraînables et réévaluables proprement ;
- tous les résultats serveur ont été importés ;
- le rapport expose scores globaux, pires seeds, cycles, exploitabilité, courbes et coûts ;
- une décision d'architecture justifiée est validée avant `RL-S6`.

