# RL-S5c-A2 — préflight

## Décision

**READY pour le checkpoint serveur RL-S5c-A2.** Cette décision valide uniquement le protocole,
la reproductibilité et les tests fonctionnels. Elle ne constitue aucune mesure locale de vitesse,
de compétence, de style ou de force. RL-S5c-B reste explicitement hors périmètre.

## Anomalies techniques corrigées

- La campagne A2 utilise le run `rl-s5c-calibration-v2` et ne lit ni ne modifie les artefacts A1.
- Les évaluations exécutent réellement les seeds 101, 102 et 103 à chaque checkpoint de cinq
  unités et conservent les comptes premier/second et A/B effectivement observés.
- Chaque partie possède un `game_id` persistant. Les flux moteur, learner et adversaire sont
  dérivés séparément par `derive_game_stream_seed()`. Une paire conserve la même graine de carte
  et n'inverse que le premier joueur.
- Le manifeste contient le commit, l'état Git, le manifeste de contenu pertinent, les hashes des
  composants, des DSL, de la configuration, des hyperparamètres PPO et du pool RL-S5b, ainsi que
  les versions Python et dépendances. Un worktree sale est refusé par défaut.
- Chaque branche archive avant l'unité 1 ses poids et son optimiseur neufs, sa seed et
  `parent_checkpoint: null`. Le préflight compare les poids des trois archétypes pour chaque seed.
- Les états terminaux distinguent `failed_budget`, `failed_style`, `failed_integrity` et
  `first_stable_gate`. Une reprise d'entraînement est limitée à `training` ou `interrupted`; une
  reprise après gate ne peut que terminer l'évaluation forte manquante.
- Les sorties par partie, agrégats conditionnés, décomposition de récompense, replays JSON/texte,
  diagnostic anti-reward-hacking et évaluation appariée RL-S5b sont archivés.
- Les programmes Reward DSL sont compilés une seule fois par processus. Une régression A2 qui les
  recompilait trois fois à chaque action a été supprimée ; les évaluations affichent désormais leur
  progression, leur débit et leur ETA partie par partie.
- Les commandes `preflight`, `calibrate`, `status` et `report` sont exécutables. Le lancement sans
  `--quiet` affiche la progression sur stderr et `--resume` ignore les branches déjà terminales.

## Diagnostic `controller`

Les contrôles structurels et six scénarios déterministes confirment que `Bousculer` :

- possède le même index entre `Action`, moteur, masque et PPO ;
- est légale et non masquée avec un adversaire adjacent ;
- débite correctement mouvement et endurance ;
- émet et archive le déplacement, la distance, le mur, la boue, l'interruption de nutrition,
  l'éloignement d'une carcasse et l'accès favorable ;
- est sélectionnable par une politique fraîche lors du smoke test de préflight.

Aucun défaut de chaîne action/masque/événement n'a été trouvé. Les données A1 montrent zéro
bousculade dans les 1 236 parties finales, mais ne permettent pas de distinguer exploration puis
abandon d'une absence totale d'exploration. La révision minimale candidate est donc retenue, avec
un arrêt d'intégrité aux unités 5 et 10 si les trois seeds restent à zéro tentative malgré des
occasions légales.

DSL avant : bousculade tactiquement utile `+0,08`. DSL après : même prédicat événementiel
(déplacement et mur, entrée dans la boue ou nutrition interrompue), plafond `+0,12`. Une poussée
illégale, immobile ou sans effet tactique ne reçoit aucun bonus.

## Diagnostic `predator`

A1 établit 100 % contre `aggressive-v1`, 50 % contre `opportunist-v1`, 0 % contre `prudent-v1`,
aucun gate sur 3 seeds et des ROUND_LIMIT fréquents. A1 ne contient toutefois ni replays ni
télémétrie jointe suffisante pour décider si la cause est la poursuite, l'endurance, la nutrition,
le placement ou la temporisation.

A2 archive désormais contre `prudent-v1` la distance moyenne, l'évolution de l'endurance, les
morsures, dégâts, repos, poursuites sans attaque, occasions de nutrition ignorées, actions sans
événement tactique, premier tour de dégâts et raison terminale, avec replays ciblés. En l'absence
de preuve diagnostique préalable, **le DSL predator reste inchangé** : dégâts infligés `+0,10` et
dégâts reçus `-0,03`. La candidate `+0,12/-0,05` n'est pas appliquée aveuglément.

## `scavenger`

Le DSL est inchangé : dégâts `+0,10/-0,05`, points de carcasse `+0,15/-0,10` et nutrition sûre
`+0,03`. Le modèle est néanmoins réentraîné depuis zéro comme les deux autres archétypes.

## Hashes gelés

- Commit observé : `2e316342774db1df2207c0a3ac9f33945826236c`
- Configuration résolue : `673923433be7d5164fb6700fc5faf9e202c1a9f32e0f02bffc229764a2996e9b`
- Reward `scavenger` : `ea73ee7c6680aa7eebad3c0e69a9f8f81cb8b1317154f667765710600e7c2e9d`
- Reward `predator` : `e3218fb2071d45211671c46f77cf1d526b864be378cf243893f62e236f95e818`
- Reward `controller` : `b52c405f2b9bed31c5c5be5a7075f39e3e0b902451ee5f93c59c96c9d2572c55`
- Pool RL-S5b : `f60caef5f68b68cd06c29a2794cdd5054ae353039100e75dbe73e662aa1113db`
- Contenu influençant A2 : `732fb4bc28de4fe7c7b73091244ce24da45573d2dbe7fecee0b44325732bc9fc`
- Hyperparamètres PPO : `a10d65aaf5f3db56a8788fd5cc70615e65a6b1228fa98c59a3bd9699bdc304cd`
- Moteur : `9a7d2992501acea3531280a60919dc2496d2cab6cff326b06c254552607222ec`
- Règles : `ad441a84b2ba1ee8f048e71ac6ddc7271b09bfb0515a398903076a7018a3d0ed`
- Observation : `529721e65e1d10311b0d6eeb2658854463ebbb907587a9795d652a4c466397b1`
- Actions et masques : `e932c69c0812aaeca8b880f0000a79bf6cd89d71e07e0822f686d1165dcd851f`
- Implémentation PPO : `3666c09394db997cdfa60996e7f6d8bb2acf6b176a4f9f082e39f48caa1b19ae`

Le manifeste complet et immuable est enregistré dans
`artifacts/rl/s5c/rl-s5c-calibration-v2/manifest.json`. Le smoke test frais a sélectionné
32 bousculades sur 256 tirages légaux.

## Validation locale

- Tests ciblés A2 : 25 réussis.
- Suite complète : 613 réussis, 2 ignorés.
- `ruff check src tests` : réussi.
- `ruff format --check src tests` : réussi, 190 fichiers conformes.
- `mypy --strict src/dinorl_engine/core` : réussi, 8 fichiers sans erreur.
- Couverture du noyau : 99 % (643 instructions, 5 non couvertes).
- Préflight réel : `READY`; pool vérifié; 9 initialisations comparées; smoke test réussi.
- Aucun entraînement ni benchmark de performance n'a été lancé localement.

## Risques à surveiller sur le serveur

- Un `ROUND_LIMIT` supérieur à 10 % déclenche une inspection automatique des replays ; une identité
  fondée sur la temporisation invalide le style.
- L'arrêt `failed_integrity` de `controller` est volontairement strict et s'évalue sur les trois
  seeds aux unités 5 et 10.
- La bande RL-S5b de 25–50 %, cible 35–45 %, ne sera connue qu'après exécution serveur.
- Les conclusions sur `predator` ne doivent être prises qu'après lecture des diagnostics contre
  `prudent-v1` produits par A2.
