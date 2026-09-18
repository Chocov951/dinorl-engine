# RL-S5c — Bots spécialisés de départ et de référence

## 1. Objet

`RL-S5c` produit trois familles de politiques apprises, compétentes mais volontairement incomplètes :

- `scavenger` : privilégie la victoire aux points de carcasse ;
- `predator` : privilégie les dégâts et la victoire par KO ;
- `controller` : privilégie la bousculade, la boue et le contrôle des ressources.

Chaque famille fournit :

- un starter public clonable par les joueurs ;
- une référence publique immuable ;
- au moins deux variantes cachées servant à vérifier la généralisation ;
- ses métriques de compétence, de style et de contre-stratégie.

Ces bots remplacent les bots déterministes comme principaux adversaires de niveau intermédiaire, mais pas comme outils de test. Les bots déterministes restent conservés pour les tests de règles, les diagnostics et le gate de compétence historique.

`RL-S5c` intervient après la sélection de l'architecture officielle par `RL-S5b` et avant `RL-S6`.

## 2. Principe central : apprentissage depuis zéro

Tous les spécialistes sont entraînés depuis une initialisation aléatoire neuve.

Il est interdit de charger :

- les poids d'un généraliste ;
- les poids d'un autre spécialiste ;
- un optimiseur déjà entraîné ;
- un état de ligue ou de self-play antérieur.

Les mêmes seeds d'initialisation sont utilisées entre archétypes afin de rendre les profils comparables. Par exemple, la seed `S1` de `scavenger`, `predator` et `controller` démarre avec les mêmes poids initiaux, mais avec un programme de récompense spécialisé différent.

Chaque run s'arrête au premier franchissement stable du gate. Il ne continue pas à s'entraîner pour rechercher le meilleur score possible.

## 3. Prérequis figés

Les décisions suivantes issues de `RL-S5b` sont désormais figées pour cette campagne :

- architecture officielle V1 : `mlp-compact-v2` ;
- limite réglementaire : `30` manches ;
- une fin par `ROUND_LIMIT` est un nul réglementaire avec `terminated=True`, `truncated=False` et récompense terminale nulle ;
- toute évaluation officielle reste appariée en deux manches avec inversion du premier joueur.

Avant le lancement de la campagne de production, figer également :

- les versions du moteur, des règles, de l'observation et des actions ;
- les hyperparamètres PPO ;
- les trois programmes de récompense spécialisés ;
- les seeds d'entraînement et d'évaluation ;
- le pool d'évaluation ;
- les seuils de compétence et de style.

Les trois archétypes utilisent `mlp-compact-v2` et les mêmes hyperparamètres PPO. L'identité doit provenir du programme de récompense, pas d'une différence de capacité réseau.

Les rôles premier/second et les côtés A/B sont tirés de façon équilibrée et leurs comptes réellement obtenus sont enregistrés par cycle et pour le run complet. Les évaluations appariées réinitialisent des flux RNG reproductibles par `game_id` ; modifier une limite ou une option de rapport ne doit pas modifier les actions tirées avant la divergence effective des règles.

Le self-play est hors périmètre pour leur création. Il pourra être employé ensuite par les joueurs pour faire évoluer leur starter.

## 4. Adversaire d'entraînement

En V1, les trois spécialistes sont entraînés exclusivement contre `random-legal`.

Cette contrainte :

- reproduit les conditions ayant permis au généraliste actuel d'apprendre les règles ;
- rend les trois profils directement comparables ;
- évite d'entraîner les spécialistes sur les adversaires utilisés par leur gate ;
- limite leur généralisation et conserve des faiblesses exploitables.

Les bots déterministes, les checkpoints `RL-S5b`, les autres spécialistes et les variantes cachées sont réservés à l'évaluation. Ils ne doivent jamais apparaître dans le pool d'entraînement de cette campagne.

## 5. Programmes de récompense

### 5.1 Partie commune obligatoire

Chaque profil conserve :

- victoire : `+1` ;
- défaite : `-1` ;
- nul : `0` ;
- la sémantique déjà validée des récompenses de dégâts infligés et reçus ;
- aucune récompense pour une action illégale, impossible ou sans effet.

La victoire officielle reste la priorité. Le rapport mesure séparément le retour terminal et le retour auxiliaire, avec leur distribution par épisode. Il avertit lorsqu'un profil obtient durablement davantage de retour en perdant tout en satisfaisant son heuristique qu'en recherchant la victoire. Aucun plafond global arbitraire n'est appliqué à la somme des récompenses auxiliaires ; les coefficients et plafonds sont définis événement par événement dans le Reward DSL.

Une récompense événementielle n'est accordée qu'au changement d'état correspondant. Il est interdit de récompenser à chaque observation le maintien d'un même état.

### 5.2 Profil `scavenger`

Objectif : rechercher les trois points de carcasse tout en sachant se défendre.

Configuration initiale proposée :

- bonus lors de la validation effective d'un point de carcasse ;
- petit bonus pour commencer à se nourrir à une distance sûre, supérieure à trois cases ;
- petit bonus pour interrompre une consommation adverse lorsque cela protège une carcasse contestée ;
- récompenses de dégâts communes inchangées.

Ne jamais récompenser une simple présence sur une carcasse ni une tentative de nutrition interrompue.

### 5.3 Profil `predator`

Objectif : créer le contact, infliger des dégâts et conclure par KO.

Configuration initiale proposée :

- poids supérieur sur les dégâts infligés ;
- poids légèrement inférieur sur les dégâts reçus afin de tolérer les échanges offensifs ;
- aucun bonus propre aux carcasses ;
- aucun bonus pour se rapprocher sans conséquence tactique.

Ne jamais récompenser directement la distance courte : cela produirait un agent collé à l'adversaire sans savoir gagner.

### 5.4 Profil `controller`

Objectif : gagner de la position par la bousculade et le terrain.

Une bousculade n'est récompensée que si elle produit au moins un effet utile :

- collision murale ;
- déplacement de l'adversaire dans la boue ;
- interruption d'une nutrition ;
- éloignement de l'adversaire d'une carcasse active ;
- création immédiate d'un accès favorable à une carcasse.

Une bousculade répétée sans changement tactique ne rapporte rien. Le bonus total d'une même bousculade est plafonné, même si plusieurs prédicats sont vrais.

### 5.5 Calibration

Les coefficients exacts sont des paramètres du Reward DSL, pas des constantes dans le code. Une courte campagne de calibration peut modifier ces coefficients.

Après validation des coefficients, créer un hash de chaque programme. La campagne de production ne permet plus de les modifier. Toute modification produit une nouvelle version complète, par exemple `scavenger-reward-v2`.

## 6. Gate de compétence et arrêt anticipé

### 6.1 Gate existant

Réutiliser le gate déterministe déjà validé :

- score contre `random-legal` supérieur ou égal à `90 %` ;
- moyenne contre les bots déterministes strictement supérieure à `55 %` ;
- score individuel strictement supérieur à `40 %` contre chacun ;
- deux évaluations consécutives réussies.

L'évaluation reste déclenchée tous les cinq unités.

### 6.2 Arrêt obligatoire

Pour chaque seed :

1. entraîner depuis zéro ;
2. évaluer tous les cinq unités ;
3. dès la deuxième réussite consécutive, enregistrer le checkpoint comme `first_stable_gate` ;
4. arrêter immédiatement cette seed ;
5. ne jamais reprendre cette seed pour améliorer son score.

Le checkpoint retenu est donc celui de la deuxième réussite consécutive, pas le meilleur checkpoint observé a posteriori.

Le budget maximal initial est de `147` unités. Une seed qui n'atteint pas le gate dans ce budget est un échec et n'est pas prolongée silencieusement.

### 6.3 Validation du style au gate

Après l'arrêt, exécuter la suite comportementale. Si le bot franchit le gate de compétence mais ne présente pas son style, le run est rejeté.

Il est interdit de continuer son entraînement pour essayer de faire apparaître le style. Il faut corriger le programme de récompense lors d'une campagne de calibration distincte, puis recommencer depuis zéro.

## 7. Signatures comportementales

Les seuils ci-dessous sont provisoires et doivent être figés après la campagne de calibration.

### 7.1 `scavenger`

Mesurer :

- proportion des victoires obtenues aux trois points ;
- points de carcasse moyens par manche ;
- consommations commencées, réussies et interrompues ;
- distance à l'adversaire au début d'une nutrition ;
- part des carcasses centrales dans les points gagnés.

Critère initial : au moins `60 %` de ses victoires sont obtenues aux points et ce taux est supérieur à celui de `predator`.

### 7.2 `predator`

Mesurer :

- proportion des victoires obtenues par KO ;
- dégâts infligés et reçus par manche ;
- morsures réussies et manquées ;
- tours nécessaires au premier dégât ;
- points de carcasse accessibles mais ignorés.

Critère initial : au moins `60 %` de ses victoires sont obtenues par KO et ses dégâts moyens dépassent ceux de `scavenger`.

### 7.3 `controller`

Mesurer :

- bousculades tentées et réussies ;
- proportion de bousculades utiles ;
- collisions murales provoquées ;
- poussées terminées dans la boue ;
- interruptions de nutrition par poussée ;
- éloignements d'une carcasse active.

Critère initial : au moins `50 %` de ses bousculades réussies sont utiles et son nombre de bousculades utiles dépasse d'au moins `30 %` celui des deux autres profils.

### 7.4 Règle commune

Les métriques de style sont rapportées avec et sans conditionnement sur la victoire. Un bot ne valide jamais son style en accumulant une action récompensée tout en perdant systématiquement.

## 8. Niveau de force recherché

Un spécialiste de production doit :

- franchir le gate de compétence ;
- rester clairement inférieur au meilleur généraliste issu de `RL-S5b` ;
- représenter un défi crédible pour un joueur débutant ;
- conserver au moins une faiblesse exploitable ;
- ne pas dominer simultanément les deux autres archétypes.

Bande initiale proposée contre le pool fort figé de `RL-S5b` :

- score robuste minimal : `25 %` ;
- score robuste maximal : `50 %` ;
- cible souhaitée : `35–45 %`.

Un score inférieur à `25 %` produit un bot trop faible. Un score supérieur à `50 %` signale un starter probablement trop avancé. Ces seuils sont évalués par famille et par seed, pas uniquement en moyenne globale.

## 9. Contre-stratégies et exploitabilité volontaire

Chaque archétype doit disposer d'un contre identifiable :

- `scavenger` : pression rapide et interruption des nutritions ;
- `predator` : esquive, gestion d'endurance et punition des poursuites ;
- `controller` : trajectoires évitant murs, boue et axes de poussée.

La contre-stratégie est validée de deux façons :

1. au moins une famille adverse ou une politique de test obtient `60 %` ou plus contre le spécialiste ;
2. une population d'exploiters issue d'une base compétente commune améliore son score contre la cible figée et atteint `65 %` ou plus lors d'une évaluation appariée indépendante.

### 9.1 Base commune des exploiters

Les exploiters constituent une exception explicite au principe d'entraînement depuis zéro des starters. L'expérience `RL-S5b` a montré qu'un exploiter Compact entraîné 50 unités depuis zéro apprend principalement à fuir jusqu'au nul, sans produire aucune victoire.

Avant la campagne de production, figer un unique checkpoint `exploiter-base-v1` qui :

- utilise `mlp-compact-v2` ;
- franchit le gate de compétence ;
- n'a jamais été entraîné contre les spécialistes `RL-S5c` ;
- est identique pour toutes les cibles et toutes les comparaisons ;
- possède un hash et une provenance enregistrés.

Pour chaque cible, créer trois exploiters par warm-start des mêmes poids, avec :

- un optimiseur neuf ;
- des seeds communes entre cibles ;
- la cible gelée et vérifiée par hash ;
- des évaluations aux unités `0`, `10`, `20`, `30`, `40` et `50` ;
- un budget initial de `20` unités, extensible à `50` seulement si la courbe progresse encore ;
- le même programme de récompense et les mêmes hyperparamètres pour toutes les cibles.

### 9.2 Validation

Rapporter séparément taux de victoire, taux de nul, taux de défaite et cause terminale. Une politique qui augmente son score uniquement en atteignant `ROUND_LIMIT` est classée `temporisatrice`, pas comme contre-stratégie gagnante.

Pour valider une faiblesse exploitable, demander :

- un score officiel d'au moins `65 %` pour au moins un exploiter ;
- au moins une victoire réelle, et pas uniquement des nuls ;
- une amélioration mesurable par rapport à `exploiter-base-v1` avant fine-tuning ;
- une faiblesse stratégique identifiable dans les replays.

L'exploiter emploie la même architecture pour toutes les cibles et n'entre jamais dans leur entraînement. Un exploiter utilisé ensuite pour adapter une cible devient « vu » et doit être remplacé par un nouvel exploiter pour toute nouvelle mesure.

Si aucun exploiter n'atteint le seuil mais qu'une famille adverse démontre clairement le contre attendu, le résultat est déclaré ambigu et soumis à validation humaine ; le spécialiste n'est pas automatiquement rejeté comme trop robuste.

L'objectif n'est pas d'introduire une erreur triviale ou un comportement cassé. La faiblesse doit être stratégique et observable dans les replays.

## 10. Campagne et sélection des snapshots

### 10.1 Calibration

Pour chaque archétype :

- trois seeds communes ;
- entraînement depuis zéro ;
- arrêt au premier gate stable ;
- validation du style et de la bande de force ;
- une seule révision coordonnée des coefficients autorisée par itération de calibration.

Les résultats de calibration ne sont pas publiés comme starters.

### 10.2 Production

Après gel des trois programmes :

- cinq seeds communes par archétype ;
- au moins quatre seeds sur cinq doivent franchir le gate ;
- au moins trois snapshots par archétype doivent satisfaire compétence, style et bande de force ;
- aucune configuration ne change après consultation partielle des résultats.

### 10.3 Affectation des rôles

Parmi les snapshots éligibles d'un archétype :

- choisir comme snapshot public celui dont le score robuste est le plus proche de la médiane ;
- utiliser exactement ses poids pour la référence publique immuable ;
- utiliser ces mêmes poids comme template clonable du joueur ;
- sélectionner deux autres seeds éligibles comme variantes cachées ;
- ne pas choisir automatiquement le meilleur snapshot.

Cette règle évite de publier une seed exceptionnellement forte ou faible.

## 11. Starter joueur et référence

La référence publique et le starter partagent initialement les mêmes poids, mais ont des rôles différents :

- `reference-<archetype>-v1` reste immuable ;
- `starter-<archetype>-v1` est un template servant à créer une branche du joueur.

Lors de la création d'une branche joueur :

- copier uniquement les poids et le manifeste d'inférence ;
- créer un optimiseur neuf ;
- réinitialiser les compteurs d'entraînement de la branche ;
- conserver la provenance du starter ;
- qualifier l'opération de `warm-start`, jamais de reprise exacte ;
- permettre au joueur de modifier son programme de récompense dans les limites du produit.

La progression du joueur peut ensuite être mesurée contre la référence publique et contre les variantes cachées de la même famille.

## 12. Pool de benchmark V2

Après validation, créer un nouveau pool immuable `benchmark-pool-v2` contenant :

- le pool `RL-S5b` déjà figé ;
- les trois références publiques ;
- les six variantes cachées ;
- les métadonnées de style et de difficulté, sans exposer les poids cachés aux utilisateurs.

Ne jamais modifier rétroactivement le pool utilisé pour décider `RL-S5b`.

Les variantes cachées sont accessibles au serveur d'évaluation, mais absentes des téléchargements, de l'API publique et des logs détaillant leurs probabilités d'action.

## 13. Évaluations finales

Exécuter :

- une matrice cross-play complète entre références et variantes ;
- les matchs appariés contre `random-legal` et les bots déterministes ;
- les matchs contre le pool fort `RL-S5b` ;
- les tests de contre-stratégie ;
- les exploiters warm-start décrits à la section 9 ;
- une évaluation déterministe diagnostique ;
- une évaluation stochastique principale.

Chaque match officiel comporte deux manches avec la même graine et inversion du premier joueur. Rapporter :

- score, victoires, nuls et défaites ;
- intervalle de confiance à `95 %` ;
- résultats premier et second joueur ;
- pire seed et quartile inférieur ;
- mode de victoire ;
- métriques comportementales ;
- durée moyenne des parties ;
- cause terminale et taux de `ROUND_LIMIT` ;
- score selon le rôle premier/second et selon le côté A/B ;
- score contre la meilleure contre-stratégie et le meilleur exploiter.

Le rapport doit signaler les cycles entre archétypes. Une relation non transitive est souhaitable ; une domination universelle ne l'est pas.

La limite reste fixée à 30 manches. Un taux de `ROUND_LIMIT` supérieur à `10 %` pour un spécialiste déclenche un avertissement et une inspection de replays. Un style reposant principalement sur la fuite jusqu'au nul ne peut pas être validé comme identité stratégique.

## 14. Interfaces attendues

Intégrer les capacités au CLI existant, sans scripts ponctuels obligatoires :

```bash
python -m dinorl_engine.rl s5c calibrate --archetype scavenger --config configs/rl/s5c.yaml
python -m dinorl_engine.rl s5c train-production --config configs/rl/s5c.yaml --resume
python -m dinorl_engine.rl s5c evaluate-style --run <run-id>
python -m dinorl_engine.rl s5c train-counter --base <exploiter-base-id> --target <snapshot-id> --units 20 --max-units 50
python -m dinorl_engine.rl s5c build-family --archetype <id> --run <run-id>
python -m dinorl_engine.rl s5c build-benchmark-v2 --run <run-id>
python -m dinorl_engine.rl s5c report --run <run-id>
```

Chaque commande longue est reprenable et idempotente. Chaque sortie enregistre commit Git, configuration, hashes, seeds, progression, point d'arrêt et erreurs.

Arborescence logique :

```text
artifacts/rl/s5c/<run_id>/
├── manifest.json
├── rewards/
├── calibration/
├── production/
├── families/
├── counters/
├── benchmark-v2/
└── report/
```

## 15. Développement test-driven

Écrire les tests avant le code pour vérifier au minimum :

- absence de checkpoint parent entraîné ;
- initialisations communes et reproductibles entre archétypes ;
- entraînement exclusivement contre `random-legal` ;
- architecture `mlp-compact-v2` et limite réglementaire de 30 manches ;
- comptes premier/second et A/B effectivement enregistrés par cycle ;
- équilibre statistique des rôles sur le run complet ;
- évaluation sans mutation des poids ni des RNG de la politique ;
- flux RNG réinitialisés par `game_id` et préfixes identiques entre évaluations contrefactuelles ;
- `ROUND_LIMIT` renvoyé comme nul réglementaire, `terminated=True` et `truncated=False` ;
- arrêt exact à la seconde réussite consécutive ;
- impossibilité de reprendre un run après `first_stable_gate` ;
- rejet d'une seed compétente sans signature de style ;
- récompense uniquement sur événement et absence de boucles de récompense ;
- plafonnement d'une bousculade cumulant plusieurs effets ;
- séparation calibration / production ;
- gel des Reward DSL et de leurs hashes ;
- sélection du snapshot médian plutôt que du meilleur ;
- reset complet de l'optimiseur lors du clonage joueur ;
- immutabilité des références ;
- confidentialité des variantes cachées ;
- warm-start des exploiters depuis une base commune avec optimiseur neuf ;
- comparaison exploiter avant/après fine-tuning ;
- classification d'un exploiter sans victoire et ne produisant que des limites comme `temporisatrice` ;
- invalidation d'un exploiter déjà vu pour une nouvelle mesure ;
- séparation entre pool `RL-S5b` et `benchmark-pool-v2` ;
- calcul exact des signatures comportementales et des agrégats par seed ;
- reprise d'une campagne interrompue avant le gate.

Les tests locaux emploient de petits modèles factices et des budgets réduits. Toute mesure de vitesse, d'apprentissage ou de force est effectuée sur le serveur.

## 16. Checkpoints serveur

### RL-S5c-A — Calibration des récompenses

Après implémentation locale et smoke tests :

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S5c-A
```

Le serveur entraîne trois seeds par archétype, s'arrête aux gates et produit les courbes de compétence et de style. Après import local, les coefficients peuvent être révisés, puis la configuration de production est figée.

### RL-S5c-B — Production des familles

Après gel des profils :

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S5c-B
```

Le serveur entraîne cinq seeds par archétype. Après import, sélectionner mécaniquement les références médianes et les variantes cachées. Aucun réglage n'est autorisé à partir des résultats partiels.

### RL-S5c-C — Cross-play et contre-stratégies

Après construction des familles :

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S5c-C
```

Le serveur exécute matrice croisée, pool fort, tests de style et exploiters warm-start. Après import, générer `starter-families-decision.md` et demander une validation humaine avant la création de `benchmark-pool-v2`.

## 17. Critères de fin

`RL-S5c` est terminé lorsque :

- les trois profils sont entraînés depuis zéro avec l'architecture officielle ;
- chaque run de production s'arrête au premier gate stable ;
- au moins trois snapshots valides existent par archétype ;
- chaque famille possède une identité comportementale mesurée ;
- les bots se situent dans la bande de force intermédiaire ;
- chacun possède une contre-stratégie démontrée ;
- aucun spécialiste ne repose principalement sur `ROUND_LIMIT` pour obtenir ses résultats ;
- les exploiters sont évalués depuis une base compétente commune et les politiques seulement temporisatrices sont identifiées ;
- les rôles premier/second sont équilibrés, enregistrés et rapportés ;
- aucun archétype ne domine universellement les autres ;
- les starters clonent les poids avec un optimiseur neuf ;
- les références sont immuables et les variantes cachées protégées ;
- les résultats des trois checkpoints serveur ont été importés ;
- la décision humaine est enregistrée ;
- `benchmark-pool-v2` est créé sans modifier les preuves de `RL-S5b`.
