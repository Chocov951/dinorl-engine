# RL-S5 — rétrospective et décisions de clôture

Statut : `RL_S5_CLOSED_READY_FOR_RL_L8`  
Périmètre : `RL-S5`, `RL-S5b` et `RL-S5c`  
Décision spécialiste : `NO_ELIGIBLE_VARIANT`

Ce document est la source humaine unique des résultats et décisions issus de RL-S5. Les
mesures détaillées restent dans leurs rapports JSON et Markdown, indexés par
`artifacts/rl/RL-S5-registry.json`. La clôture n'a entraîné aucun modèle et n'a exécuté
aucun benchmark.

## 1. RL-S5 — apprentissage initial et architecture

Le MLP a appris plus vite que le petit CNN sans perte compensatoire observée. Le CNN est
donc abandonné pour la V1, tout en restant archivé comme expérience et adversaire de
diagnostic.

Les trois variantes MLP V2 évaluées étaient :

| Architecture | Structure | Paramètres encodeur approximatifs |
| --- | --- | ---: |
| `mlp-compact-v2` | `663 → 64 → 64` | 46 656 |
| `mlp-balanced-v2` | `663 → 96 → 64` | 69 952 |
| `mlp-deep-v2` | `663 → 128 → 128 → 64` | 109 760 |

Un agent entraîné contre `random-legal` apprend les règles et bat rapidement les bots
déterministes. Le jalon de 147 unités correspond à 301 056 transitions apprenant et
environ 4 704 pas d'optimisation par seed avec la configuration employée. Il a montré que
la policy, l'observation et PPO fonctionnent, mais une réussite contre `random-legal` et
les bots scripts ne mesure pas à elle seule un niveau avancé.

`mlp-compact-v2` est l'architecture officielle V1 en raison de sa vitesse, de sa stabilité
inter-seed et du meilleur compromis initial observé. Les joueurs de la première bêta ne
choisissent pas leur architecture.

## 2. RL-S5b — robustesse, continuation, self-play et exploitabilité

Le pool fort immuable RL-S5b est :

```text
f60caef5f68b68cd06c29a2794cdd5054ae353039100e75dbe73e662aa1113db
```

Cross-play à l'unité 147 :

| Architecture | Score moyen | Médiane | Quartile bas |
| --- | ---: | ---: | ---: |
| `mlp-compact-v2` | 0,532 | 0,534 | 0,503 |
| `mlp-v1` | 0,521 | 0,529 | 0,496 |
| `mlp-balanced-v2` | 0,507 | 0,513 | 0,491 |
| `mlp-deep-v2` | 0,440 | 0,449 | 0,117 |

Aucun cycle strict entre architectures n'a été observé. Certaines seeds Deep étaient très
instables. La continuation contrôlée a nettement progressé contre un pool fort. Au budget
testé, le self-play historique n'a pas démontré d'avantage convaincant sur le pool fixe ;
il reste une piste postérieure, mais n'est pas nécessaire au démarrage de la bêta.

Les premiers exploiters partis de zéro ont surtout appris à temporiser jusqu'au nul, sans
démontrer de victoire. Une future étude d'exploitabilité doit partir d'un checkpoint
généraliste compétent commun, recréer un optimiseur pour chaque exploiter, figer chaque
cible et réentraîner des exploiters neufs après toute adaptation. Les résultats doivent
séparer victoires, nuls, défaites et causes terminales. Une politique ne produisant que
des nuls est temporisatrice ; au moins une victoire est nécessaire pour parler de
contre-stratégie gagnante.

## 3. Audit de la limite de 30 manches

La limite réglementaire de 30 manches est conservée. Moins de 1 % des parties ordinaires
des agents unité 197 l'atteignaient ; les taux élevés provenaient principalement des
exploiters temporisateurs. Les parties plafonnées étaient majoritairement inactives en fin
de partie et rien n'indiquait que la limite coupait généralement des stratégies actives
complexes.

Un nul par `ROUND_LIMIT` est `terminated=True`, `truncated=False`, avec une récompense
terminale nulle. Toute comparaison contrefactuelle doit conserver des flux RNG identiques
par `game_id`. Les rôles premier/second et les côtés réellement tirés doivent rester
archivés, et les matchs officiels sont joués en aller-retour en raison du biais de rôle.

## 4. RL-S5c — starters spécialisés

- **A1** : protocole incomplet, provenance et télémétrie insuffisantes.
- **A2** : protocole corrigé ; `scavenger` spécialisé mais trop faible, `predator` et
  `controller` hors gate.
- **A3-CONTROL** : `CONTROL_PASSED`. Le contrôle généraliste a validé la policy,
  l'observation, PPO et `mlp-compact-v2` ; ils ne sont pas la cause principale des échecs
  de spécialisation.
- **A3-PILOT et R2** : les curricula augmentent rapidement la force, mais aucune variante
  ne conserve simultanément style, bande de force, stabilité stochastique et absence de
  reward hacking.

Résultats informatifs A3-R2 :

| Variante | Checkpoint | Score held-out | Conclusion |
| --- | ---: | ---: | --- |
| `scavenger-a2-control` | u150 | 0,4203 | style points insuffisant et reward hacking |
| `scavenger-a3-curriculum` | u95 | 0,4156 | seulement 7,97 % des victoires aux points |
| `scavenger-a3-soft` | u130 | 0,3159 | style insuffisant et chute stochastique |
| `predator-a2-control` | u195 | 0,4904 | dégâts dominant le retour auxiliaire |
| `predator-a3-balanced` | u165 | 0,3516 | reward hacking, chute stochastique et 12,5 % de limites |

Décision finale : `NO_ELIGIBLE_VARIANT`. `controller` est `deferred_post_v1`.

Enseignements :

- une récompense spécialisée peut produire rapidement une stratégie monolithique ;
- même minoritaire, un curriculum peut dépasser rapidement la bande de force visée ;
- un style visible contre des bots faibles peut disparaître contre le held-out ;
- le style doit donc être mesuré sur des adversaires tenus à l'écart ;
- un avertissement de répétabilité doit distinguer un événement non borné d'un événement
  explicitement plafonné ;
- un score global ne suffit pas : routes de victoire, retours par règle, stabilité, rôles,
  côtés et `ROUND_LIMIT` doivent être conservés ;
- les règles de sélection d'un checkpoint doivent être prédéfinies, sans sélection
  opportuniste a posteriori.

Les spécialistes entraînés sont conservés comme adversaires diagnostiques. Ils ne sont ni
des références équilibrées ni des starters officiels.

## 5. Décision produit pour la bêta

Aucun starter spécialisé entraîné n'est publié. Tous les joueurs commencent depuis
`starter-zero-v1`, snapshot initial non entraîné de l'architecture `mlp-compact-v2`, avec
exactement les mêmes poids. Une branche joueur copie seulement les poids et le manifeste
d'architecture, puis reçoit un optimiseur neuf, des RNG propres et des compteurs nuls.

Les politiques historiques utiles sont classées dans `beta-validation-pool-v1`. Une
politique instable, temporisatrice ou affectée par du reward hacking peut servir au
diagnostic, mais jamais être présentée comme adversaire équilibré ni comme seuil unique de
publication. Cette version du pool est immuable.

La bêta doit maintenant observer comment plusieurs joueurs font évoluer cette origine
commune en modifiant récompenses et hyperparamètres. Ces données réelles précéderont toute
nouvelle décision sur les starters spécialisés, le gameplay, les curricula, le self-play,
la limite de manches, les budgets et l'aide aux débutants.

## 6. Reprise du plan

RL-S5 est clos. L'étape suivante est `RL-L8 — Publication et validation finale`.
`RL-S6` n'est pas exécuté et ne doit pas être déclaré réussi. Sa configuration canonique
sur cinq seeds n'étant pas figée sans ambiguïté, elle requiert une décision séparée avant
le checkpoint serveur RL-S6 ; elle ne doit pas être déduite des campagnes spécialisées.

RL-S5 CLÔTURÉ — REPRISE AUTORISÉE À RL-L8
