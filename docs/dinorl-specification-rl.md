# DinoRL — Spécification RL et plan d’implémentation

> **Statut : prêt à implémenter**  
> **Périmètre : environnement RL, récompenses programmables, entraînement, évaluation, artefacts et orchestration**  
> **Méthode : développement test-driven, mesures exclusivement sur le serveur cible**  
> **Prérequis : moteur de jeu livré jusqu’à L7, outils de débogage local disponibles**

## 1. Objet du document

Ce document définit le sous-système d’apprentissage par renforcement de **DinoRL**. Il regroupe :

1. les spécifications fonctionnelles de l’apprentissage ;
2. les contrats techniques de l’environnement, des observations et des récompenses ;
3. l’algorithme, les réseaux et les protocoles d’évaluation ;
4. la gestion des expériences, checkpoints et snapshots ;
5. l’orchestration sur le serveur de calcul ;
6. la méthode TDD et les checkpoints de mesure à exécuter sur le serveur ;
7. les tickets ordonnés destinés aux agents d’implémentation.

Le système est conçu pour être utilisable avant l’intégration à MyBlog. La bibliothèque Python et la CLI sont livrées en premier. L’API HTTP et l’interface MyBlog réutiliseront ultérieurement les mêmes services applicatifs sans contenir de logique RL propre.

## 2. Sources de vérité

| Source | Autorité |
| --- | --- |
| `duel-de-raptors-regles-du-jeu.md` | Règles fonctionnelles du combat |
| `duel-de-raptors-plan-implementation.md` | Architecture et contrats du moteur existant |
| Présent document | Contrats et comportement du sous-système RL |
| Schémas JSON versionnés | Contrats machine des configurations et artefacts |
| Tests de conformité | Traduction exécutable des contrats |
| `SERVER_BASELINE.md` | Référence de performance observée sur le serveur cible |
| Résultats importés des gates `RL-S0` à `RL-S7` | Décisions empiriques de performance et de calibration |

En cas de désaccord, les règles du jeu prévalent sur ce document pour la résolution d’un combat. Le wrapper RL ne modifie jamais une règle du moteur.

Toute modification cassant une observation, une action, une récompense compilée, un checkpoint ou un snapshot impose une nouvelle version explicite. Aucune conversion silencieuse n’est autorisée.

## 3. Statut des décisions

### 3.1 Décisions figées pour la V1

- un seul agent généraliste par politique, entraîné contre un mélange des trois bots principaux ;
- environnement mono-agent Gymnasium et action masquée ;
- observation canonique composée d’une grille de huit canaux et de quinze scalaires ;
- récompenses programmables au moyen d’un langage dédié borné ;
- aucun compteur temporel ni historique dans la récompense V1 ;
- `MaskablePPO`, PyTorch, CPU et quatre epochs d’optimisation par unité ;
- une unité égale à 2 048 transitions apprenant plus une mise à jour PPO complète ;
- checkpoints visibles toutes les cinq unités ;
- politiques officielles stochastiques à température 1 ;
- snapshots publiés en `safetensors`, sans code exécutable ;
- aucun import de modèle externe ;
- conservation des vingt checkpoints complets les plus récents par branche, en plus des checkpoints protégés ;
- entraînement localement développé, mais toute mesure décisionnelle exécutée sur le serveur cible.

### 3.2 Valeurs provisoires à recalibrer par mesure

- budget quotidien initial : 100 unités, soit 204 800 transitions apprenant ;
- hyperparamètres PPO initiaux ;
- seuils anti-effondrement stochastique ;
- mode de concurrence entre parties interactives et entraînement ;
- architecture finale MLP ou CNN ;
- nombre d’unités nécessaire pour rendre un progrès visible, atteindre le gate et approcher le plateau.

Une valeur provisoire ne peut être modifiée qu’après un résultat serveur versionné et importé.

# Partie I — Spécifications fonctionnelles

## 4. Objectif d’apprentissage V1

La V1 doit démontrer qu’une politique généraliste apprend à jouer au Duel de raptors contre plusieurs styles déterministes, et mesurer la vitesse de cet apprentissage.

L’agent n’est pas spécialisé par adversaire. À chaque épisode d’entraînement, son adversaire est tiré uniformément parmi :

- `aggressive-v1` ;
- `prudent-v1` ;
- `opportunist-v1`.

L’identité du bot n’est pas fournie dans l’observation. L’adversaire reste fixe pendant tout l’épisode.

Le contrôleur `random-legal-v1` est ajouté comme baseline facile et diagnostic. Il ne fait pas partie du mélange d’entraînement principal.

La V1 n’utilise ni curriculum, ni self-play, ni mémoire récurrente, ni adaptation en ligne pendant une partie.

## 5. Critères de réussite

Le score d’un ensemble de manches est calculé comme aux échecs :

\[
\operatorname{score}=\frac{\text{victoires}+0{,}5\times\text{nuls}}{\text{nombre de manches}}.
\]

Une seed d’entraînement franchit le jalon technique lorsque son évaluation déterministe satisfait simultanément :

- score d’au moins 90 % contre `random-legal-v1` ;
- moyenne strictement supérieure à 55 % contre les trois bots principaux ;
- score strictement supérieur à 40 % contre chacun des trois bots principaux ;
- respect de ces seuils lors de deux évaluations consécutives.

Le développement utilise trois seeds indépendantes. La validation finale en utilise cinq ; au moins quatre seeds sur cinq doivent franchir le jalon.

Le rapport doit mesurer la vitesse d’apprentissage selon plusieurs axes :

- unités PPO ;
- transitions apprenant ;
- actions réellement exécutées par le moteur ;
- matchs et manches ;
- epochs d’optimisation et pas optimiseur ;
- temps mur et temps CPU.

Le résultat ne doit jamais être résumé à « l’agent gagne ». Il doit indiquer à quel coût d’entraînement les progrès deviennent visibles, quand le gate est atteint et quand la courbe approche son plateau.

## 6. Environnement mono-agent

### 6.1 Contrat général

`DinoRLSingleAgentEnv` adapte le moteur existant sans en modifier l’état ni les règles.

- Le learner contrôle exactement un raptor.
- L’adversaire est un contrôleur interne à l’environnement.
- Le learner choisit une action élémentaire à chaque `step()`.
- Après chaque action du learner, il observe immédiatement le nouvel état si son tour continue.
- Lorsqu’il termine son tour, l’environnement fait jouer automatiquement toutes les actions du tour adverse.
- L’appel `step()` ne rend la main qu’au prochain état de décision du learner ou à l’état terminal.
- Les récompenses produites pendant le tour adverse sont accumulées et renvoyées avec cette transition du learner.

Une **transition apprenant** correspond à un appel valide à `step()` effectué par la politique entraînée. Les actions automatiques de l’adversaire sont comptées séparément comme **actions moteur**.

### 6.2 Initialisation d’un épisode

Pour chaque épisode :

1. tirer uniformément `learner_actor` dans `{A, B}` ;
2. tirer indépendamment `first_actor` dans `{A, B}` ;
3. tirer uniformément un bot parmi les trois adversaires principaux ;
4. initialiser `arena_mvp_v1` avec les règles officielles ;
5. si l’adversaire commence, exécuter son premier tour avant de produire la première observation du learner.

Le siège et l’initiative sont donc randomisés indépendamment. Leur seed est dérivée de manière stable de la seed de l’exécution, de l’environnement vectorisé et de l’index d’épisode.

### 6.3 Terminaison

Un épisode prend fin uniquement lorsque le moteur indique :

- un K.-O. ;
- une victoire à trois points de carcasse ;
- le match nul après 30 manches.

La limite de 30 manches est une terminaison réelle des règles, pas une troncature artificielle de Gymnasium. La valeur ne doit jamais être bootstrapée au-delà de cette limite.

### 6.4 Actions

L’espace discret possède neuf actions, dans l’ordre versionné du moteur :

| Index | Action |
| ---: | --- |
| 0 | `MOVE_NORTH` |
| 1 | `MOVE_EAST` |
| 2 | `MOVE_SOUTH` |
| 3 | `MOVE_WEST` |
| 4 | `BITE` |
| 5 | `SHOVE` |
| 6 | `FEED` |
| 7 | `REST` |
| 8 | `END_TURN` |

Le wrapper expose `action_masks()` sous la forme d’un vecteur booléen de longueur 9. Le masque n’est pas ajouté à l’entrée du réseau.

Une action illégale ayant échappé au masque constitue une erreur technique : l’épisode et le cycle PPO sont abandonnés. Elle n’est ni transformée en `END_TURN`, ni pénalisée comme un choix de jeu.

## 7. Observation du réseau

### 7.1 Perspective canonique

L’observation est toujours exprimée du point de vue du learner.

- Si le learner contrôle A, les coordonnées restent inchangées.
- S’il contrôle B, la carte est tournée de 180° et les rôles A/B deviennent `self`/`opponent`.
- Les actions directionnelles sont converties dans les deux sens : nord ↔ sud et est ↔ ouest après rotation.
- La transformation est bijective et testée sur chaque case et chaque direction.

Cette canonicalisation ne masque pas l’initiative : `self_is_first_actor` indique si le learner est le premier acteur de la manche.

### 7.2 Grille de huit canaux

La grille possède la forme `(8, 9, 9)` et utilise les canaux binaires suivants :

| Canal | Contenu |
| ---: | --- |
| 0 | murs |
| 1 | boue |
| 2 | emplacements des deux carcasses latérales |
| 3 | carcasses latérales encore disponibles |
| 4 | emplacement de la carcasse centrale |
| 5 | carcasse centrale active |
| 6 | position de `self` |
| 7 | position de `opponent` |

Les canaux d’emplacement restent présents même lorsqu’une carcasse est consommée ou inactive. La représentation interne peut être `uint8`, mais l’entrée PyTorch est convertie explicitement en `float32`.

### 7.3 Quinze scalaires

Les scalaires sont normalisés dans `[0,1]` et ordonnés comme suit :

| Index | Variable | Normalisation |
| ---: | --- | --- |
| 0 | PV de `self` | `/ 6` |
| 1 | endurance de `self` | `/ 5` |
| 2 | PM de `self` | `/ 3` |
| 3 | score carcasse de `self` | `/ 3` |
| 4 | action principale disponible pour `self` | booléen |
| 5 | déplacement volontaire déjà effectué par `self` | booléen |
| 6 | PV de `opponent` | `/ 6` |
| 7 | endurance de `opponent` | `/ 5` |
| 8 | score carcasse de `opponent` | `/ 3` |
| 9 | consommation de `opponent` en attente | booléen |
| 10 | repos de `opponent` en attente | booléen |
| 11 | carcasse centrale en attente ou en recharge | booléen |
| 12 | tours individuels avant réactivation centrale | `/ 2` |
| 13 | manche courante | `/ 30` |
| 14 | `self_is_first_actor` | booléen |

Le vecteur final d’un MLP contient `8 × 9 × 9 + 15 = 663` valeurs.

Pour les deux scalaires centraux :

- carcasse active : `central_unavailable=0`, délai `0` ;
- consommation en attente : `central_unavailable=1`, délai `0` ;
- recharge : `central_unavailable=1`, délai égal au nombre de tours individuels restants divisé par 2.

Le délai est borné entre 0 et 2 avant normalisation. L’encodage d’un état terminal éventuel reste soumis aux mêmes bornes.

### 7.4 Informations volontairement absentes

Le réseau ne reçoit pas directement :

- distance de Manhattan ;
- longueur du plus court chemin ;
- coût de déplacement tenant compte de la boue ;
- dernière action ;
- historique de la partie ;
- identité ou style du bot ;
- compteurs de comportement ;
- récompense future ou règle heuristique.

Le réseau doit apprendre les relations spatiales à partir de la carte.

## 8. Adversaires d’entraînement et de diagnostic

### 8.1 Bots principaux

Les identifiants suivants sont immuables :

- `aggressive-v1` ;
- `prudent-v1` ;
- `opportunist-v1`.

Une modification de logique crée un nouvel identifiant. Un ancien bot ne peut jamais changer de comportement sous le même nom.

Les bots :

- ne consultent que l’état public ;
- utilisent le même masque d’actions légales ;
- choisissent une action après chaque action de leur tour ;
- n’accèdent ni aux poids, ni à la récompense, ni aux futures valeurs aléatoires du learner.

### 8.2 Bot aléatoire

`random-legal-v1` tire uniformément parmi toutes les actions légales à chaque décision. Son flux aléatoire est indépendant de ceux du moteur, du learner et des autres environnements.

### 8.3 Distribution et suivi

Le tirage des trois bots principaux est uniforme par épisode, pas par transition. Les métriques enregistrent pour chacun :

- épisodes ;
- transitions apprenant ;
- actions moteur ;
- temps de calcul ;
- résultats et durée des parties.

Si un bot représente plus de 55 % des transitions d’un cycle alors que le tirage d’épisodes est uniforme, un avertissement est enregistré. La distribution n’est pas corrigée automatiquement dans la V1.

## 9. Récompenses

### 9.1 Séparation avec le score officiel

La récompense RL n’a aucune autorité sur le résultat officiel d’un combat. Les victoires, défaites, nuls, Elo et récompenses MyBlog sont calculés exclusivement depuis le résultat du moteur.

Les joueurs peuvent définir des objectifs volontairement inhabituels ou même inefficaces. La plateforme garantit la sécurité et la validité d’exécution, pas la qualité stratégique de la récompense.

### 9.2 Récompense de référence

La configuration fournie par défaut est :

| Événement | Récompense |
| --- | ---: |
| Victoire | `+1.00` |
| Défaite | `-1.00` |
| Match nul | `0.00` |
| Point de carcasse obtenu par le learner | `+0.10` |
| Point de carcasse obtenu par l’adversaire | `-0.10` |
| Dégât infligé | `+0.05` par PV |
| Dégât reçu | `-0.05` par PV |

Toutes ces valeurs, y compris les récompenses terminales, sont configurables par le joueur.

### 9.3 Entrée de la fonction de récompense

La fonction reçoit une transition publique canonique contenant :

- état public avant l’action ;
- action demandée, acteur, direction, coûts et réussite ;
- état public après résolution ;
- événements publics produits par le moteur ;
- éventuel résultat terminal.

Elle peut calculer des distances ou des chemins à partir de ces états, même si ces quantités ne sont pas fournies au réseau. Elle ne peut lire aucune information cachée ni conserver un état entre deux transitions.

Les récompenses des actions adverses sont évaluées depuis la perspective du learner et accumulées jusqu’au prochain retour de `step()`.

### 9.4 Absence de compteurs en V1

La récompense V1 est une fonction pure de la transition courante. Elle ne possède :

- ni compteur utilisateur ;
- ni mémoire modifiable ;
- ni historique implicite ;
- ni accès aux transitions précédentes.

Une V2 pourra proposer un catalogue fermé de compteurs. Dans ce cas, tout compteur utilisable par la récompense devra également faire partie de l’observation du réseau.

## 10. Reward DSL

### 10.1 Objectif

Le joueur définit sa récompense au moyen :

- d’un éditeur visuel pour les compositions simples ;
- d’un langage textuel typé pour les fonctions plus créatives.

Les deux interfaces produisent le même programme source et le même bytecode. Aucun code Python utilisateur n’est exécuté.

### 10.2 Types

Le langage expose uniquement :

`Number`, `Bool`, `Position`, `Actor`, `ActionKind`, `Direction`, `TileKind`, `Outcome` et `Transition`.

Les variables locales sont immuables. Les fonctions peuvent appeler d’autres fonctions à condition que leur graphe d’appels soit acyclique.

Le point d’entrée obligatoire est :

```text
fn reward(t: Transition) -> Number
```

### 10.3 Constructions autorisées

- fonctions typées ;
- constantes et variables locales immuables ;
- `if` / `else` ;
- `return` ;
- arithmétique, comparaisons et booléens ;
- accès en lecture aux champs de `Transition` ;
- appels aux fonctions intégrées du catalogue.

#### 10.3.1 Grammaire minimale

La V1 utilise des identifiants ASCII sensibles à la casse, des commentaires de ligne `//`, des nombres décimaux finis et des points-virgules obligatoires. Elle ne possède ni chaîne de caractères, ni collection définie par l’utilisateur.

```ebnf
program       = { function } ;
function      = "fn", identifier, "(", [ parameters ], ")",
                "->", type, block ;
parameters    = parameter, { ",", parameter } ;
parameter     = identifier, ":", type ;
type          = "Number" | "Bool" | "Position" | "Actor"
              | "ActionKind" | "Direction" | "TileKind"
              | "Outcome" | "Transition" ;
block         = "{", { statement }, "}" ;
statement     = let_statement | if_statement | return_statement ;
let_statement = "let", identifier, ":", type, "=", expression, ";" ;
if_statement  = "if", expression, block, [ "else", block ] ;
return_statement = "return", expression, ";" ;
expression    = literal | constant | field_access | function_call
              | "(", expression, ")"
              | unary_expression | binary_expression ;
function_call = identifier, "(", [ arguments ], ")" ;
arguments     = expression, { ",", expression } ;
field_access  = identifier, { ".", identifier } ;
```

Précédence, de la plus forte à la plus faible : accès/appel, unaire `!` et `-`, multiplication et division, addition et soustraction, comparaisons, égalité, `&&`, puis `||`. Les opérateurs booléens court-circuitent. Aucun cast implicite n’existe ; les littéraux numériques sont des `Number`.

Chaque fonction doit retourner une valeur sur tous ses chemins atteignables. `reward` accepte exactement un paramètre `Transition` et renvoie `Number`. Les surcharges, fonctions anonymes et valeurs de fonction n’existent pas.

`Number` est évalué en IEEE 754 double précision. L’optimiseur n’a pas le droit de réassocier les opérations flottantes. Le résultat vérifié est ensuite converti explicitement dans le dtype attendu par le buffer PPO.

#### 10.3.2 Surface publique de `Transition`

`t.before` et `t.after` sont des vues en lecture seule qui exposent exactement les informations de l’observation canonique, sous forme non normalisée :

- `self` : position, PV, endurance, PM, score, disponibilité de l’action principale et déplacement volontaire ;
- `opponent` : position, PV, endurance, score, consommation en attente et repos en attente ;
- murs, boue, emplacements et disponibilité des carcasses ;
- manche, initiative, indisponibilité centrale et délai de réactivation.

Elles n’ajoutent notamment ni PM adverse, ni distance précalculée, ni identité du bot, ni historique.

Ces vues ne sont pas des valeurs stockables dans une variable et ne constituent donc pas un type supplémentaire du langage. Leurs champs terminaux sont des types du catalogue fermé.

`t.action` expose :

- `actor` dans la perspective `SELF` ou `OPPONENT` ;
- `kind` ;
- `direction`, avec la constante `NO_DIRECTION` lorsque l’action n’est pas directionnelle ;
- coût en PM et en endurance ;
- `success`.

`t.outcome` vaut `ONGOING`, `WIN`, `LOSS` ou `DRAW` depuis la perspective du learner. La route terminale est consultable par les fonctions événementielles `ended_by_ko`, `ended_by_carcass_score` et `ended_by_round_limit`.

Les états `before` et `after`, l’action et les événements appartiennent tous à la même transition moteur. Aucun champ ne peut consulter le choix suivant de l’adversaire.

### 10.4 Constructions interdites

- boucle ;
- récursion directe ou indirecte ;
- mutation ;
- import ;
- entrée-sortie, réseau ou système de fichiers ;
- heure, aléatoire ou environnement système ;
- allocation dynamique ;
- réflexion ou appel de fonction dynamique.

### 10.5 Fonctions intégrées minimales

Fonctions numériques :

- `min`, `max`, `abs`, `clamp`.

Fonctions spatiales :

- `manhattan(position_a, position_b)` ;
- `path_steps(position_a, position_b)` en évitant les murs ;
- `movement_cost(position_a, position_b)` en tenant compte de la boue ;
- `is_adjacent(position_a, position_b)` ;
- `tile_at(position)` ;
- `is_legal(transition, action_kind)`.

Ces fonctions spatiales consultent la carte publique associée à la transition. `path_steps` et `movement_cost` utilisent la géométrie statique des murs et de la boue ; ils n’anticipent aucune action et ne traitent pas la position actuelle de l’autre raptor comme un mur permanent.

Fonctions événementielles, toujours dérivées de la transition :

- dégâts infligés ou reçus ;
- morsure tentée ou réussie ;
- bousculade tentée, réussie ou terminée contre un mur ;
- entrée ou sortie de boue ;
- consommation commencée, interrompue ou validée ;
- repos commencé ou résolu ;
- point de carcasse obtenu ;
- route terminale et `terminal_score`.

Signatures V1 des principales fonctions événementielles :

| Fonction | Retour |
| --- | --- |
| `damage_dealt(t, actor)` | nombre de PV infligés par cet acteur |
| `bite_attempted(t, actor)` | booléen |
| `bite_hit(t, actor)` | booléen |
| `shove_attempted(t, actor)` | booléen |
| `shove_moved_target(t, actor)` | booléen |
| `shove_wall_damage(t, actor)` | dégât de collision infligé |
| `entered_mud(t, actor)` | booléen |
| `exited_mud(t, actor)` | booléen |
| `feed_started(t, actor)` | booléen |
| `feed_interrupted(t, actor)` | booléen |
| `feed_completed(t, actor)` | booléen |
| `rest_started(t, actor)` | booléen |
| `rest_completed(t, actor)` | booléen |
| `carcass_points_gained(t, actor)` | nombre de points obtenus |
| `ended_by_ko(t)` | booléen |
| `ended_by_carcass_score(t)` | booléen |
| `ended_by_round_limit(t)` | booléen |
| `terminal_score(t, win, loss, draw)` | valeur choisie selon l’issue, sinon zéro |

`is_legal(t, action_kind)` interroge le masque de l’acteur de `t.action` dans `t.before`. Les fonctions prenant un `actor` acceptent uniquement `SELF` ou `OPPONENT` dans la V1.

Le catalogue exact est versionné. Une nouvelle fonction intégrée nécessite une nouvelle version de DSL si elle modifie la compatibilité.

### 10.6 Exemple normatif

```text
fn safe_feed(t: Transition) -> Number {
    if feed_started(t, SELF) &&
       manhattan(t.before.self.position, t.before.opponent.position) > 3 {
        return 0.20;
    }
    return 0.00;
}

fn reward(t: Transition) -> Number {
    return terminal_score(t, 1.0, -1.0, 0.0)
         + 0.05 * damage_dealt(t, SELF)
         - 0.05 * damage_dealt(t, OPPONENT)
         + safe_feed(t);
}
```

### 10.7 Compilation

Le pipeline est fixe :

```text
source → lexer → parser → AST typé → validation → optimisation → bytecode → VM bornée
```

Les optimisations autorisées incluent :

- repliement de constantes ;
- suppression des branches mortes ;
- remplacement des noms par des index ;
- analyse des métriques réellement utilisées ;
- calcul paresseux des distances et chemins ;
- court-circuit booléen.

Le programme compilé est mis en cache par SHA-256 de la source canonique, de la version du DSL et de la version du catalogue.

### 10.8 Limites et erreurs

| Limite | Valeur |
| --- | ---: |
| Taille de source | 32 Kio |
| Fonctions | 64 |
| Nœuds AST | 2 048 maximum |
| Profondeur syntaxique | 64 |
| Profondeur d’appels | 32 |
| Instructions VM par transition | 10 000 maximum |

Une division invalide, une valeur non finie ou une récompense dont la valeur absolue dépasse `1e6` arrête le cycle courant. Le cycle n’est pas facturé et un diagnostic est produit. Aucun clamp silencieux n’est appliqué.

Les seuls avertissements non bloquants de la V1 sont :

- fonction inaccessible depuis `reward` ;
- branche statiquement morte ;
- récompense prouvée constamment nulle.

### 10.9 Exactitude et performance

Une implémentation de référence simple évalue l’AST. La VM compilée doit produire exactement le même résultat ou la même erreur sur les tests unitaires, génératifs et les corpus de transitions.

Sur le serveur cible, la récompense de référence compilée ne doit pas augmenter de plus de 10 % le coût mesuré par rapport à son équivalent natif précalculé. Cette exigence est vérifiée par `RL-S2`.

## 11. Algorithme et réseaux

### 11.1 Algorithme V1

- Gymnasium ;
- PyTorch ;
- `MaskablePPO` de `sb3-contrib` ;
- calcul CPU initial ;
- masque appliqué avant sélection ou échantillonnage ;
- aucune normalisation des observations ou récompenses ;
- normalisation des avantages activée.

Le dictionnaire d’observation utilise une `MaskableMultiInputActorCriticPolicy` avec un extracteur personnalisé. `normalize_images=False` est imposé, car les canaux sont déjà encodés explicitement dans `[0,1]` et ne doivent pas être divisés une seconde fois.

Avec `SubprocVecEnv`, `action_masks()` est implémentée directement dans l’environnement : aucun wrapper `ActionMasker` local ne doit être nécessaire dans les sous-processus. Les évaluations utilisent les utilitaires masqués de `sb3_contrib.common.maskable`, jamais leurs équivalents génériques ignorant le masque.

### 11.2 Hyperparamètres initiaux

| Paramètre | Valeur |
| --- | ---: |
| Transitions par rollout global | 2 048 |
| Epochs PPO par mise à jour | 4 |
| Batch | 256 |
| Learning rate | `3e-4` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_range` | `0.20` |
| Coefficient d’entropie | `0.01` |
| Coefficient de valeur | `0.50` |
| Norme maximale du gradient | `0.50` |
| Température officielle | `1.0` |
| Threads Torch initiaux | 1 |

Il n’existe ni clipping de récompense, ni `VecNormalize`, ni planification automatique du learning rate, ni `target_kl` dans la V1.

### 11.3 Architectures candidates

Deux architectures sont implémentées uniquement pour le benchmark initial.

**MLP**

- aplatissement des 663 valeurs ;
- couches `663 → 128 → 64` avec ReLU ;
- représentation de 64 valeurs transmise aux têtes politique et valeur.

**Petit CNN**

- grille : `Conv2d(8,16,3,padding=1)`, ReLU, puis une seconde couche identique ;
- projection avec ReLU de la grille aplatie vers 64 valeurs ;
- projection avec ReLU des quinze scalaires vers 16 valeurs ;
- fusion `80 → 64` avec ReLU ;
- même forme de têtes politique et valeur que le MLP.

Dans les deux cas, les têtes communes sont des projections directes `64 → 9` pour la politique et `64 → 1` pour la valeur. Les encodeurs possèdent respectivement environ 93 248 paramètres pour le MLP et 91 936 pour le CNN, avant ces têtes identiques. Leurs dimensions sont gelées pendant `RL-S4` et `RL-S5`. Le rapport consigne le nombre exact de paramètres, les FLOPs approximatifs et le temps d’inférence.

Après `RL-S5`, une seule architecture est retenue : celle qui apprend le plus vite sur le serveur. Les joueurs ne choisissent pas leur architecture dans la V1.

## 12. Paramètres accessibles aux joueurs

### 12.1 Paramètres essentiels

- programme de récompense ;
- learning rate ;
- `gamma` ;
- coefficient d’entropie ;
- seed ;
- budget en unités ;
- checkpoint parent.

### 12.2 Paramètres avancés

- `gae_lambda` ;
- `clip_range` ;
- taille de batch.

### 12.3 Domaines valides

| Paramètre | Domaine V1 |
| --- | --- |
| Learning rate | `0 < lr`, maximum `0.1`, minimum technique `1e-8` |
| `gamma` | `[0,1]` |
| Coefficient d’entropie | `[0,1]` |
| `gae_lambda` | `[0,1]` |
| `clip_range` | `(0,1]` |
| Batch | diviseur de 2 048 choisi parmi `64, 128, 256, 512, 1024, 2048` |
| Seed | entier non signé sur 32 bits |
| Budget | entier strictement positif, borné par les unités disponibles |

L’interface pourra appliquer une plage ergonomique plus étroite, sans rendre les autres valeurs contractuellement invalides. Le serveur contrôle l’architecture, le device, les processus, `n_envs`, le nombre d’epochs et le nombre de transitions par unité.

## 13. Unités d’entraînement

Une unité est indivisible du point de vue de la facturation :

1. collecter exactement 2 048 transitions apprenant sur l’ensemble des environnements ;
2. calculer avantages et retours ;
3. effectuer quatre epochs PPO ;
4. enregistrer métriques et état de récupération ;
5. valider atomiquement l’unité ;
6. débiter une unité.

Avec le batch de référence de 256, une unité réalise huit minibatchs par epoch, soit 32 pas optimiseur.

Le budget initial est de 100 unités par joueur et par jour, soit 204 800 transitions. Il sera recalibré après `RL-S7` afin que :

- un jour permette d’observer une évolution mesurable ;
- deux jours ne suffisent pas systématiquement à atteindre un agent parfaitement stabilisé ;
- le budget ne soit pas si faible qu’aucune différence ne soit visible.

Une évaluation et un checkpoint visible sont déclenchés au démarrage, puis toutes les cinq unités, soit toutes les 10 240 transitions.

Les évaluations ne consomment pas d’unité d’entraînement. À une échéance de cinq unités, l’état d’entraînement est d’abord validé et débité, puis le checkpoint visible est créé et évalué. Une panne de l’évaluation ne supprime ni ne rembourse les unités déjà correctement entraînées ; elle laisse le checkpoint dans l’état `evaluation_pending` jusqu’à réussite ou échec explicite de l’évaluation.

## 14. Évaluation

### 14.1 Évaluation déterministe

La politique utilise l’action légale de probabilité maximale. Pour chaque bot principal, elle joue les quatre configurations exactes :

- learner en A, learner premier ;
- learner en A, learner second ;
- learner en B, learner premier ;
- learner en B, learner second.

Ces quatre manches correspondent à deux confrontations aller-retour. Les résultats servent au jalon technique défini en section 5.

`random-legal-v1` est évalué sur 200 confrontations appariées, soit 400 manches, avec des seeds versionnées.

### 14.2 Évaluation stochastique de publication

Pour chaque bot principal :

- 200 confrontations appariées ;
- 400 manches ;
- politique échantillonnée à température 1 ;
- mêmes seeds pour les deux initiatives d’une confrontation.

Un checkpoint déterministement éligible n’est publiable que si :

- sa moyenne stochastique contre les trois bots principaux ne baisse pas de plus de 10 points de pourcentage par rapport à sa moyenne déterministe ;
- aucun bot principal ne baisse de plus de 15 points ;
- son score stochastique contre `random-legal-v1` reste au moins égal à 85 %.

Si une valeur se trouve à deux points de pourcentage ou moins d’une frontière de décision, 200 confrontations supplémentaires sont exécutées et le résultat élargi remplace le premier.

Ces seuils anti-effondrement sont provisoires et seront révisés uniquement depuis les courbes serveur.

### 14.3 Seeds de développement et de validation

- comparaison et développement : trois seeds ;
- validation finale : cinq seeds ;
- succès final : au moins quatre seeds sur cinq passent le gate ;
- toutes les courbes individuelles sont conservées ;
- médiane et intervalle interquartile sont les agrégats principaux, la moyenne reste complémentaire.

## 15. Inférence et confrontations officielles

### 15.1 Politique officielle

L’inférence officielle est stochastique :

- logits masqués avant tirage ;
- température fixe `1.0` ;
- aucun choix utilisateur de température ;
- flux RNG indépendant pour chaque participant, dérivé de la seed de confrontation et de son identifiant de participant stable.

Le replay enregistre la seed et les actions effectivement choisies. Un replay rejoue les actions ; il ne relance jamais le réseau pour reconstruire le passé.

### 15.2 PvP, PvBot et duel

Une confrontation contient deux manches aller-retour :

- même carte ;
- même seed de base ;
- mêmes positions ;
- seul le premier joueur change ;
- victoire d’une manche : 1 point ;
- nul : 0,5 point ;
- défaite : 0 point.

Le score total d’une confrontation est compris entre 0 et 2.

### 15.3 Arène

Une tentative d’arène contient deux confrontations, donc quatre manches. Chaque confrontation utilise une seed indépendante et son propre aller-retour.

- Le challenger remplace le champion avec au moins `2,5 / 4`.
- À `2 / 4`, le champion reste en place.
- Toute égalité finale profite au défenseur.

## 16. Expériences, branches et artefacts

### 16.1 Objets métier du serveur RL

| Objet | Description | Mutabilité |
| --- | --- | ---: |
| Expérience | Configuration et filiation scientifique | immuable |
| Exécution | Demande d’entraînement avec budget | statut évolutif |
| État de récupération | Dernier état exact validé après une unité | remplacé atomiquement |
| Checkpoint | Point visible et branchable | immuable |
| Snapshot publié | Politique d’inférence officielle | immuable |

Modifier un paramètre ou une récompense crée une nouvelle expérience enfant. Aucun historique n’est réécrit.

### 16.2 Reprise selon la modification

| Modification | Poids conservés | Critique conservé | Optimiseur conservé |
| --- | ---: | ---: | ---: |
| Reprise strictement identique | oui | oui | oui |
| Récompense ou `gamma` | extracteur + politique | non, réinitialisé | non |
| Autre hyperparamètre PPO | oui | oui | non |
| Architecture | non pris en charge dans le jeu V1 | — | — |

Les règles exactes de transfert sont enregistrées dans le manifeste de la branche.

### 16.3 Sauvegardes

- état de récupération interne après chaque unité validée ;
- checkpoint visible toutes les cinq unités et à la fin d’une exécution ;
- pointeur `best` modifiable vers un checkpoint immuable ;
- publication toujours déclenchée explicitement par le joueur ;
- un checkpoint non publiable reste utilisable pour la reprise et le bac à sable.

### 16.4 Conservation

Sont conservés intégralement :

- checkpoints épinglés ;
- checkpoint final d’une exécution ;
- checkpoint source d’un snapshot publié ;
- vingt checkpoints complets les plus récents de chaque branche.

Pour les checkpoints plus anciens non protégés, le serveur conserve poids d’inférence, métriques et manifeste, mais peut supprimer optimiseur et RNG. Le checkpoint reste consultable et publiable, mais ne permet plus de reprise exacte. Cette perte de capacité est signalée explicitement.

## 17. Checkpoints et snapshots

### 17.1 Checkpoint d’entraînement

Le checkpoint complet est un artefact interne et de confiance. Il contient :

- modèle et critique ;
- état de l’optimiseur ;
- générateurs aléatoires Python, NumPy, PyTorch, Gymnasium et environnements ;
- état exact de chaque partie vectorisée, adversaire courant et statistiques d’épisode partielles ;
- dernière observation, dernier masque, marqueurs de début d’épisode et compteurs internes nécessaires à PPO ;
- configuration complète ;
- source, version et hash de la récompense ;
- compteurs d’unités, transitions, actions, matchs, epochs et pas optimiseur ;
- versions du code, du moteur, des règles, de la carte, des actions et observations ;
- métriques nécessaires à la reprise et à l’audit.

Le format natif SB3 ou PyTorch est autorisé uniquement dans ce stockage interne. Il n’est jamais chargé depuis un fichier téléversé par un utilisateur.

### 17.2 Snapshot publié

```text
snapshot/
├── weights.safetensors
├── manifest.json
├── evaluation.json
└── sha256sums.txt
```

Le manifeste contient au minimum :

- identifiant opaque et propriétaire ;
- checkpoint source et parent éventuel ;
- architecture et dimensions exactes ;
- versions et hashes du moteur, des règles, de la carte, des actions et observations ;
- version de PyTorch et de l’exporteur ;
- date de création ;
- statut de publication ;
- hash de chaque fichier ;
- identifiant du rapport d’évaluation ayant autorisé la publication.

Le runtime reconstruit l’architecture depuis une liste fermée, puis charge uniquement les tenseurs. Le snapshot ne contient ni optimiseur, ni bytecode de récompense, ni objet Python sérialisé, ni code exécutable.

### 17.3 Compatibilité et import

- aucun modèle externe n’est importable en V1 ;
- un snapshot incompatible est refusé avec un code précis ;
- aucune conversion automatique silencieuse ;
- une migration future sera une opération séparée, versionnée et testée ;
- un snapshot publié est immuable ; publier à nouveau crée un autre identifiant.

## 18. Métriques et rapports

### 18.1 Axes communs

Toutes les courbes peuvent être exprimées selon :

- unités ;
- transitions apprenant ;
- actions moteur ;
- épisodes, matchs et manches ;
- epochs PPO ;
- pas optimiseur ;
- temps mur ;
- temps CPU.

### 18.2 Résultats de jeu

- score déterministe et stochastique par bot ;
- victoires, nuls et défaites ;
- fin par K.-O., points ou limite ;
- durée en manches, tours et actions ;
- siège et initiative ;
- score moyen, médian et dispersion par seed.

### 18.3 Catalogue comportemental complet

- distribution des neuf actions ;
- morsures tentées, légales et ayant touché ;
- dégâts infligés et reçus ;
- bousculades tentées, réussies, interrompues par la boue et collisions murales ;
- consommations tentées, validées et interrompues ;
- points latéraux et centraux ;
- repos commencés et résolus ;
- PM et endurance moyens, minima et finaux ;
- entrées, sorties et temps passé dans la boue ;
- route de victoire ;
- part des transitions jouées contre chaque bot.

Les agrégats sont conservés pour l’ensemble des entraînements. Un détail par transition n’est généré que pour les replays diagnostiques d’évaluation.

### 18.4 Diagnostic de récompense

Pour chaque terme nommé ou fonction composante :

- somme positive ;
- somme négative ;
- somme absolue ;
- fréquence de déclenchement ;
- contribution moyenne par épisode ;
- part de la récompense terminale.

La VM expose les composantes sans réévaluer le programme une seconde fois.

### 18.5 Diagnostic PPO

- policy loss, value loss et entropy ;
- KL approximative et clip fraction ;
- explained variance ;
- norme des gradients ;
- learning rate effectif ;
- distribution des actions ;
- nombre moyen d’actions légales ;
- distributions des retours et avantages ;
- valeurs non finies et interruptions éventuelles.

### 18.6 Diagnostic système

- transitions apprenant/s et actions moteur/s ;
- temps de collecte, optimisation, évaluation et sauvegarde ;
- CPU, RSS et charge par processus ;
- taille des checkpoints et snapshots ;
- profondeur et attente de la file ;
- temps de chargement et d’inférence.

### 18.7 Artefacts d’une exécution

```text
run/<run_id>/
├── manifest.json
├── cycles.jsonl
├── eval.jsonl
├── checkpoints/
├── diagnostic_replays/
└── final_report.json
```

Les courbes de toutes les seeds sont conservées. Le rapport agrégé utilise médiane et intervalle interquartile, avec moyenne complémentaire.

## 19. Progression visible

### 19.1 État exposé

Pour un job en attente :

- position dans la rotation ;
- unités estimées devant lui ;
- heure estimée de démarrage et de fin.

Pour un job actif :

- phase `collecting`, `optimizing`, `evaluating` ou `saving` ;
- unité courante et budget total ;
- transitions sur 2 048 pendant la collecte ;
- epoch sur 4 pendant l’optimisation ;
- dernier et prochain checkpoint ;
- temps écoulé et ETA.

Le heartbeat est émis toutes les 30 secondes. Une fin d’unité, un checkpoint, une erreur ou un changement de phase produit immédiatement un événement persistant.

L’ETA utilise les dernières unités du job, les mesures récentes du serveur, le coût des évaluations et la file actuelle. Avant calibration, l’interface affiche explicitement que l’estimation est provisoire.

### 19.2 Disponibilité pour jouer

- le dernier snapshot publié reste jouable pendant tout l’entraînement ;
- un checkpoint visible devient testable en privé dès sa validation ;
- seuls les snapshots publiés et éligibles sont utilisables dans les modes officiels ;
- le joueur n’a jamais besoin d’attendre la fin d’un entraînement pour jouer avec son ancienne version.

## 20. File, priorités et unités

### 20.1 Réservation et facturation

À la soumission d’un job :

1. réserver tout le budget demandé dans l’autorisation de calcul associée ;
2. débiter une unité seulement après sa validation atomique ;
3. ne pas débiter l’unité abandonnée ;
4. libérer le solde lors d’une annulation ou d’un échec définitif ;
5. empêcher toute double réservation au moyen d’une clé d’idempotence.

### 20.2 Rotation des entraînements

- un seul entraînement actif initialement ;
- rotation entre joueurs par tranches maximales de cinq unités ;
- vérification de la file prioritaire après chaque unité ;
- un joueur peut préparer plusieurs expériences, mais une seule de ses exécutions participe activement à la rotation ;
- s’il n’existe aucun autre job, l’exécution continue sans pause artificielle.

### 20.3 Priorités

| Priorité | Tâches |
| ---: | --- |
| P0 | PvP, arène et partie interactive |
| P1 | test privé et évaluation de publication demandée |
| P2 | évaluation périodique automatique |
| P3 | collecte et optimisation PPO |

Les matchs et tests passent avant le prochain cycle d’entraînement. `RL-S3` choisit entre :

- un worker interactif séparé fonctionnant en concurrence ;
- le traitement des tâches prioritaires aux frontières d’unités.

Le choix doit minimiser d’abord la latence des parties, puis la perte de débit d’apprentissage.

Un joueur ne peut avoir qu’un test privé actif ou en attente. Les évaluations identiques, définies par snapshot, suite, seeds et versions, sont mises en cache.

### 20.4 Annulation et reprise automatique

- annulation normale à la frontière de l’unité ;
- unité courante terminée et débitée si elle est validée ;
- arrêt administrateur d’urgence : unité courante abandonnée et non débitée ;
- erreur fonctionnelle définitive : aucun retry ;
- erreur technique temporaire : reprise depuis le dernier état validé, trois tentatives maximum ;
- après trois échecs, statut `failed`, conservation des artefacts valides et restitution du budget restant.

# Partie II — Spécifications techniques

## 21. Architecture logicielle

Le code RL appartient au dépôt séparé `dinorl-engine` et dépend localement du moteur déjà implémenté.

```text
src/dinorl_engine/
├── core/                         moteur existant, non dépendant du RL
├── controllers/                  bots existants et random-legal-v1
├── rl/
│   ├── env/
│   │   ├── single_agent.py
│   │   ├── canonical.py
│   │   ├── observation.py
│   │   └── vectorization.py
│   ├── rewards/
│   │   ├── tokens.py
│   │   ├── lexer.py
│   │   ├── parser.py
│   │   ├── ast.py
│   │   ├── types.py
│   │   ├── validate.py
│   │   ├── optimize.py
│   │   ├── bytecode.py
│   │   ├── vm.py
│   │   ├── reference.py
│   │   └── catalog.py
│   ├── policies/
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   ├── factory.py
│   │   └── export.py
│   ├── training/
│   │   ├── config.py
│   │   ├── unit.py
│   │   ├── runner.py
│   │   ├── resume.py
│   │   └── seeds.py
│   ├── evaluation/
│   │   ├── deterministic.py
│   │   ├── stochastic.py
│   │   ├── gates.py
│   │   └── metrics.py
│   ├── artifacts/
│   │   ├── manifests.py
│   │   ├── checkpoints.py
│   │   ├── snapshots.py
│   │   └── retention.py
│   ├── orchestration/
│   │   ├── database.py
│   │   ├── queue.py
│   │   ├── worker.py
│   │   ├── progress.py
│   │   └── credits.py
│   └── __main__.py
├── server_checks/
│   ├── cli.py
│   ├── manifests.py
│   ├── runner.py
│   ├── importer.py
│   ├── suites/
│   └── __main__.py
└── cli.py

schemas/rl/
├── experiment-v1.schema.json
├── run-v1.schema.json
├── progress-v1.schema.json
├── evaluation-v1.schema.json
├── checkpoint-manifest-v1.schema.json
└── snapshot-manifest-v1.schema.json

tests/rl/
├── unit/
├── properties/
├── integration/
├── contracts/
├── reproducibility/
└── performance/
```

Règle de dépendance : `core` ne peut jamais importer `rl`. Les modules `rl` peuvent importer l’API publique du moteur, mais pas ses détails privés.

## 22. Dépendances et versions

### 22.1 Identifiants initiaux

| Contrat | Identifiant initial |
| --- | --- |
| Observation RL | `rl-observation-v1` |
| Configuration d’expérience | `rl-experiment-v1` |
| Reward DSL | `reward-dsl-v1` |
| Catalogue de récompense | `reward-catalog-v1` |
| Checkpoint | `training-checkpoint-v1` |
| Snapshot publié | `policy-snapshot-v1` |
| Architecture MLP candidate | `mlp-v1` |
| Architecture CNN candidate | `small-cnn-v1` |

Ces identifiants sont inclus dans les hashes et manifestes. L’architecture gagnante conserve son identifiant après `RL-S5` ; « architecture officielle » est un pointeur de configuration et non un renommage de l’artefact.

### 22.2 Bibliothèques

Le paquet cible CPython `>=3.12,<3.13`. Le serveur de référence utilise actuellement CPython 3.12.8 ; son micro-numéro exact est enregistré dans chaque résultat et conditionne les affirmations de reprise bit-à-bit.

Les dépendances directes minimales sont :

- Gymnasium ;
- NumPy ;
- PyTorch CPU ;
- Stable-Baselines3 ;
- sb3-contrib ;
- safetensors ;
- jsonschema ;
- outils de tests et de propriétés déjà retenus par le dépôt.

Les versions exactes et transitives sont figées dans les lockfiles du dépôt. `RL-S0` vérifie leur installation réelle sur le serveur avant que les tickets suivants ne s’appuient sur elles.

Une dépendance de parsing ou de VM supplémentaire n’est acceptée que si elle réduit réellement le risque d’implémentation sans permettre l’exécution de code généraliste.

## 23. Contrat Gymnasium

### 23.1 Espaces

Conceptuellement :

```python
observation_space = gym.spaces.Dict({
    "grid": gym.spaces.Box(0.0, 1.0, shape=(8, 9, 9), dtype=np.float32),
    "features": gym.spaces.Box(0.0, 1.0, shape=(15,), dtype=np.float32),
})
action_space = gym.spaces.Discrete(9)
```

Le MLP concatène la grille aplatie et les scalaires dans son extracteur. Le CNN conserve les deux branches.

### 23.2 Algorithme de `reset()`

```text
valider options et seed
choisir learner_actor, first_actor et opponent_id
réinitialiser le moteur
si l’adversaire est actif : jouer son tour complet
si terminal : produire l’état terminal selon Gymnasium
sinon : produire observation canonique et info
```

### 23.3 Algorithme de `step(action)`

```text
vérifier que le learner est actif et que action_masks[action] est vrai
capturer l’état public canonique avant
appliquer l’action du learner
calculer et accumuler la récompense de la transition
tant que la partie continue et que l’adversaire est actif :
    obtenir son masque légal
    demander son action
    appliquer son action
    calculer et accumuler la récompense depuis la perspective du learner
produire observation suivante, récompense accumulée, terminated, truncated=False et info
```

`info` contient uniquement les diagnostics nécessaires : résultat terminal, compteurs de transition, bot de l’épisode et causes d’erreur. Les informations absentes de l’observation ne doivent jamais être consommées par la politique.

### 23.4 Invariants

- toute observation appartient à `observation_space` ;
- le masque possède au moins une action légale dans un état non terminal ;
- `END_TURN` garantit cette propriété ;
- canonicaliser deux fois par la transformation inverse restitue l’état initial ;
- chaque action canonique appliquée produit la même transition que son action moteur correspondante ;
- la somme `learner_transitions` n’inclut jamais les actions du bot ;
- `engine_actions` les inclut toutes ;
- aucun replay complet n’est construit pendant la collecte standard.

Le vérificateur générique de Gymnasium échantillonne des actions sans connaître le masque. Il est complété par un checker DinoRL qui échantillonne uniquement les actions légales ; une action illégale injectée volontairement est testée séparément comme erreur de contrat.

## 24. Configuration canonique

Toutes les commandes reçoivent un JSON strict, versionné et validé par JSON Schema.

Exemple conceptuel :

```json
{
  "schema_version": "1.0.0",
  "experiment_id": "opaque-id",
  "parent_checkpoint_id": null,
  "seed": 12345,
  "budget_units": 20,
  "reward_source": "fn reward(t: Transition) -> Number { return terminal_score(t, 1.0, -1.0, 0.0); }",
  "ppo": {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "entropy_coef": 0.01,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "batch_size": 256
  }
}
```

Règles :

- propriétés supplémentaires interdites ;
- validation avant réservation d’unités ;
- sérialisation JSON canonique pour les hashes ;
- identifiants et chemins d’artefacts générés par le serveur ;
- aucun chemin arbitraire fourni par l’utilisateur ;
- paramètres contrôlés par le serveur ajoutés au manifeste résolu, pas à la configuration utilisateur.

## 25. Vectorisation

Le rollout global contient toujours 2 048 transitions apprenant. Pour les configurations candidates :

| `n_envs` | `n_steps` par environnement |
| ---: | ---: |
| 2 | 1 024 |
| 4 | 512 |
| 8 | 256 |

`RL-S1` compare chacune avec `DummyVecEnv` et `SubprocVecEnv`. La sélection utilise le débit **end-to-end** en transitions apprenant par seconde, le temps par unité, la mémoire, la stabilité et le coût d’inférence. Le débit brut du moteur ne suffit pas.

La référence initiale est quatre environnements et un thread Torch. Le nombre d’environnements et de processus reste un paramètre serveur, jamais un choix du joueur.

## 26. Cycle atomique et reprise exacte

### 26.1 Écriture atomique

Chaque unité écrit dans un dossier temporaire du même système de fichiers :

1. sérialiser l’état complet ;
2. `fsync` des fichiers nécessaires ;
3. calculer et vérifier les hashes ;
4. écrire le manifeste de validation ;
5. renommer atomiquement vers l’état de récupération officiel ;
6. valider la transaction SQLite et le débit de l’unité.

Une panne avant l’étape 5 ne remplace pas le dernier état valide. Une panne entre stockage et transaction est réconciliée par identifiant d’unité et clé d’idempotence au redémarrage.

### 26.2 Reproductibilité

Sur le même commit, les mêmes dépendances, le même type de matériel et le même nombre de processus, une reprise doit produire exactement les mêmes poids, compteurs et métriques que l’exécution continue.

Cette exigence impose de sauvegarder et restaurer tous les RNG, y compris ceux des sous-environnements et adversaires.

Les sous-seeds sont dérivées par une fonction cryptographique stable à partir d’un domaine textuel et des identifiants concernés, par exemple `run/env/episode/opponent`. Le `hash()` natif de Python, l’ordre de parcours d’un dictionnaire non canonique et l’identifiant du processus ne sont jamais utilisés comme source de seed.

Elle impose également de restaurer les parties éventuellement inachevées à la fin du rollout. Le checkpoint comprend donc, pour chaque environnement vectorisé : état du moteur, bot courant, état de son contrôleur, accumulateurs d’épisode et flux RNG. Le runner restaure aussi les dernières observations et marqueurs internes attendus par PPO avant de reprendre la collecte. Réinitialiser simplement les environnements à la reprise est interdit, car cela modifierait la trajectoire suivante.

Entre matériels ou versions différentes, seule une reproductibilité statistique est exigée. Le manifeste doit empêcher de présenter cette reprise comme bit-à-bit exacte.

## 27. Persistance et orchestration

### 27.1 SQLite

La V1 utilise une base SQLite dédiée en mode WAL. Elle ne partage aucun fichier avec MyBlog.

Tables logiques minimales :

- `experiments` : configuration immuable, parenté et hashes ;
- `runs` : budget, progression et résultat ;
- `jobs` : état de file, priorité, tentatives et heartbeat ;
- `job_events` : événements persistants ;
- `unit_ledger` : réservation, validation, débit et restitution ;
- `artifacts` : type, chemin interne, hash, taille et protection ;
- `evaluations` : protocole, seeds, métriques et décision ;
- `evaluation_cache` : clé déterministe et résultat réutilisable.

Les blobs lourds, poids et replays diagnostiques restent dans le système de fichiers.

Le `unit_ledger` est un journal technique propre aux jobs : il garantit qu’une même unité n’est ni exécutée ni débitée deux fois. Il ne crée pas les allocations quotidiennes des joueurs. Avant l’intégration MyBlog, la CLI utilise une autorité locale de développement. Après intégration, MyBlog émet une autorisation de budget opaque et idempotente ; le serveur RL ne peut jamais consommer plus que son montant et renvoie le consommé et le solde libéré pour réconciliation.

### 27.2 États des jobs

```text
queued → compiling → running → evaluating → completed
                              ↘ failed
queued/running/evaluating → cancelled
```

Chaque transition d’état est validée par une fonction métier unique et enregistrée. Un heartbeat ancien ne suffit pas à marquer un job en échec : le superviseur vérifie d’abord le processus et le dernier artefact atomique.

### 27.3 Ordonnancement

L’orchestrateur est un processus unique. Un worker d’entraînement actif est utilisé au départ. Aucun Celery, Redis ou ordonnanceur distribué n’est introduit sans besoin mesuré.

Les priorités, la rotation par joueur et les limites de tests privés sont appliquées dans l’orchestrateur, pas dans MyBlog.

## 28. Interface en ligne de commande

La CLI constitue l’interface officielle avant l’API HTTP.

```bash
python -m dinorl_engine.rl train --config experiment.json
python -m dinorl_engine.rl resume --checkpoint <checkpoint_id> --units 20
python -m dinorl_engine.rl evaluate --checkpoint <checkpoint_id> --suite deterministic
python -m dinorl_engine.rl evaluate --checkpoint <checkpoint_id> --suite publication
python -m dinorl_engine.rl publish --checkpoint <checkpoint_id>
python -m dinorl_engine.rl inspect --run <run_id>
python -m dinorl_engine.rl status --job <job_id>
python -m dinorl_engine.rl status --job <job_id> --follow
python -m dinorl_engine.rl cancel --job <job_id>
```

Exigences :

- codes de sortie documentés ;
- sortie humaine sur `stderr`, résultat JSON optionnel sur `stdout` ;
- option `--json` pour l’automatisation ;
- aucune saisie interactive dans les commandes serveur ;
- même couche de services que la future API ;
- publication impossible sans rapport d’éligibilité valide.

## 29. Rapports de progression

Le modèle `ProgressV1` contient au minimum :

- `job_id`, `run_id`, statut et phase ;
- unités demandées, réservées, validées et restantes ;
- index de l’unité en cours ;
- transitions collectées sur 2 048 ;
- epoch courante sur 4 ;
- dernier heartbeat ;
- temps écoulé ;
- ETA et indicateur de confiance ;
- dernier checkpoint et prochaine échéance ;
- position estimée dans la file ;
- dernière erreur publique éventuelle.

Le worker limite les écritures de heartbeat à une toutes les 30 secondes. La CLI `--follow` peut rafraîchir plus souvent en lisant le dernier état sans forcer de nouvelles écritures.

## 30. Protocoles de mesure serveur

### 30.1 Règle absolue

Le développement et les tests de correction s’exécutent localement. Tout test qui produit une mesure utilisée pour choisir une architecture, un parallélisme, un budget, un seuil ou une optimisation s’exécute sur le serveur cible.

Une mesure locale peut diagnostiquer un bug, mais ne peut jamais justifier une décision de capacité.

### 30.2 Procédure obligatoire

À chaque checkpoint `RL-Sx`, l’agent d’implémentation :

1. termine les tests locaux de correction ;
2. prépare la suite, ses configurations, seeds et schémas de résultat ;
3. vérifie que les résultats de mesure ne sont pas déjà inventés ou simulés ;
4. s’arrête avec le message exact :

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-Sx
```

L’utilisateur commit et déploie le même état du dépôt sur le serveur, puis exécute :

```bash
python -m dinorl_engine.server_checks run --suite RL-Sx
```

La commande :

- refuse un dépôt contenant des modifications suivies non commitées ;
- capture commit Git, hashes des dépendances, versions, CPU, OS et configuration ;
- effectue un échauffement séparé ;
- effectue les répétitions prévues ;
- permet la reprise des suites longues ;
- conserve résultats bruts et résumé ;
- génère :

```text
server-results-RL-Sx-<run_id>.tar.gz
```

Après transfert de cette archive dans l’environnement local :

```bash
python -m dinorl_engine.server_checks import server-results-RL-Sx-<run_id>.tar.gz
```

L’importeur vérifie :

- intégrité et SHA-256 de l’archive ;
- schéma des résultats ;
- commit Git ;
- versions et hashes des dépendances ;
- identité de la suite et des configurations ;
- complétude des répétitions.

Il installe les résultats légers sous :

```text
benchmarks/server/RL-Sx/<run_id>/
```

et régénère un résumé Markdown. Les JSON bruts légers, manifestes, résumés et décisions sont versionnés dans Git. Les poids et gros checkpoints ne le sont jamais.

L’agent ne poursuit pas le ticket suivant tant que l’archive valide n’a pas été importée.

### 30.3 Checkpoints obligatoires

| Gate | Mesure serveur | Décision ou preuve attendue |
| --- | --- | --- |
| `RL-S0` | installation, imports, wrapper, masque, une unité PPO | compatibilité réelle de la pile et exécution complète |
| `RL-S1` | 2/4/8 envs, Dummy/Subproc | vectorisation retenue par débit end-to-end |
| `RL-S2` | AST de référence, VM et récompense native | égalité exacte et surcharge VM ≤ 10 % |
| `RL-S3` | unité PPO, sauvegarde, reprise, priorité interactive | temps par unité, reprise exacte, choix concurrence/frontière |
| `RL-S4` | MLP et CNN, 10 unités chacun, une seed | smoke sans NaN et premiers signaux d’apprentissage |
| `RL-S5` | MLP et CNN, 147 unités × 3 seeds | architecture finale selon vitesse, temps et score |
| `RL-S6` | candidat figé, 5 seeds | au moins 4/5 franchissent les gates et absence d’effondrement stochastique |
| `RL-S7` | courbes prolongées | `T_visible`, `T_gate`, `T_plateau` et budget quotidien recalibré |

Les valeurs exactes de 10 et 147 unités correspondent respectivement à 20 480 et 301 056 transitions apprenant.

## 31. Protocole des gates serveur

### 31.1 `RL-S0` — Compatibilité minimale

- installer depuis les lockfiles ;
- importer moteur, Gymnasium, PyTorch, SB3 et sb3-contrib ;
- exécuter les tests contractuels du wrapper ;
- entraîner une unité complète avec la récompense de référence ;
- sauvegarder et recharger l’état ;
- produire un rapport de durée et mémoire sans en faire encore un seuil de CI.

### 31.2 `RL-S1` — Vectorisation

- matrice `n_envs ∈ {2,4,8}` × `{DummyVecEnv, SubprocVecEnv}` ;
- même modèle, mêmes seeds et même nombre de transitions ;
- un échauffement exclu et au moins cinq répétitions ;
- débit end-to-end, temps par phase, CPU et RSS ;
- décision écrite et justifiée dans `decision.md`.

### 31.3 `RL-S2` — Reward VM

- corpus identique de transitions publiques ;
- comparaison exacte AST/bytecode sur toutes les sorties et erreurs ;
- récompense de référence native contre bytecode équivalent ;
- échauffement et au moins cinq répétitions ;
- médiane de surcharge ≤ 10 % ;
- en cas d’échec, profilage avant toute extension native.

### 31.4 `RL-S3` — Unité et reprise

- unité continue contre reprise depuis l’unité précédente ;
- égalité bit-à-bit sur le même serveur et le même build ;
- mesure collecte/optimisation/sauvegarde ;
- test d’un crash avant et après renommage atomique ;
- test de non-double-débit ;
- comparaison du worker interactif concurrent avec la priorité aux frontières d’unités ;
- décision selon latence des parties puis débit d’entraînement.

### 31.5 `RL-S4` — Smoke architectures

- 10 unités par architecture ;
- une seed commune ;
- mêmes adversaires et suites d’évaluation ;
- absence de NaN, divergence ou action illégale ;
- courbes complètes, sans sélection définitive.

### 31.6 `RL-S5` — Comparaison architectures

- 147 unités par architecture et par seed ;
- trois seeds communes ;
- rapport des transitions nécessaires aux seuils, temps mur, score final, stabilité et coût d’inférence ;
- si les métriques désignent des gagnants différents, arrêt pour décision collective ;
- l’architecture retenue est inscrite dans un fichier de décision versionné et l’autre n’est plus proposée aux joueurs.

### 31.7 `RL-S6` — Validation finale

- configuration candidate figée avant lancement ;
- cinq seeds ;
- évaluations déterministes et stochastiques complètes ;
- au moins quatre seeds sur cinq passent le gate déterministe ;
- chaque snapshot publiable satisfait le gate anti-effondrement ;
- aucune modification de configuration après consultation partielle des résultats.

### 31.8 `RL-S7` — Calibration produit

Pour au moins trois seeds représentatives, prolonger les courbes afin d’estimer :

- `T_visible` : première amélioration robuste perceptible ;
- `T_gate` : coût typique du franchissement du jalon ;
- `T_plateau` : début d’une zone de gains marginaux faibles.

Choisir ensuite le budget quotidien `D` de sorte que, dans la mesure du possible :

```text
T_visible ≤ D < T_plateau / 2
```

Le budget de 100 unités reste en vigueur jusqu’à cette décision.

## 32. Test-driven development

### 32.1 Discipline générale

Pour chaque comportement :

1. écrire le test ou la propriété ;
2. constater son échec ;
3. implémenter le minimum ;
4. faire passer le test ;
5. ajouter cas limites et invariants ;
6. refactoriser ;
7. mesurer sur le serveur uniquement si le comportement est critique pour les performances.

### 32.2 Pull request locale

La suite de validation ordinaire contient :

- tests unitaires et de contrat ;
- tests de propriétés ;
- intégrations courtes ;
- une unité PPO sur une configuration minimale ;
- aucun seuil de performance stochastique ;
- aucun benchmark serveur simulé.

Le smoke manuel ou nocturne exécute 10 unités, soit 20 480 transitions.

Commandes locales minimales depuis la racine du dépôt :

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy --strict src/dinorl_engine/rl src/dinorl_engine/server_checks
.venv/bin/pytest tests/rl
```

Les tests marqués `server_measurement` sont exclus de cette validation et ne sont exécutés que par `dinorl_engine.server_checks` sur le serveur cible.

### 32.3 Tests obligatoires par domaine

**Environnement**

- espaces Gymnasium ;
- neuf actions et masque ;
- auto-jeu de l’adversaire ;
- accumulation des récompenses ;
- terminaison au tour adverse ;
- compteur transitions/actions ;
- rotation et mapping directionnel exhaustifs.

**Récompenses**

- lexer, parser, types et erreurs localisées ;
- graphe d’appels acyclique ;
- limites de ressources ;
- court-circuit ;
- non-finis et dépassement ;
- égalité référence/VM par tests génératifs ;
- pureté et absence d’état entre transitions.

**PPO**

- masque appliqué à l’entraînement et à l’inférence ;
- 2 048 transitions exactes ;
- quatre epochs ;
- batchs complets ;
- métriques finies ;
- déterminisme de reprise.

**Artefacts**

- hashes ;
- écriture atomique ;
- corruption détectée ;
- compatibilité stricte ;
- export `safetensors` sans pickle ;
- snapshot non modifiable ;
- rétention et épinglage.

**Orchestration**

- réservation et débit idempotents ;
- annulation ;
- trois retries techniques ;
- aucun retry fonctionnel ;
- priorité P0 à P3 ;
- rotation par joueur ;
- cache d’évaluation ;
- heartbeat 30 secondes et événements immédiats.

## 33. Référence de performance existante

Le serveur cible mesuré utilise CPython 3.12.8, Linux, quatre CPU logiques et le démarrage multiprocessing `fork`.

La baseline du moteur sans réseau atteint environ :

- 207,5 combats/s avec un processus ;
- 255,7 combats/s avec deux processus ;
- 316,2 combats/s avec quatre processus ;
- 376,0 combats/s avec huit processus.

Ces valeurs montrent des rendements décroissants et une latence croissante. Elles ne préjugent pas du meilleur nombre d’environnements PPO.

Le replay complet réduit le débit observé d’environ 78,7 %. Il reste donc désactivé pendant l’entraînement et n’est activé que pour les diagnostics ciblés.

Aucun chiffre de cette section ne devient un seuil absolu de CI. Toute comparaison doit reprendre la charge, l’échauffement, les répétitions, les versions et le matériel du protocole serveur.

## 34. Sécurité

- aucune exécution de Python utilisateur ;
- Reward DSL borné et sans accès système ;
- aucune récursion, boucle ou allocation dynamique ;
- validation stricte avant réservation ;
- aucune URL ou chemin d’artefact arbitraire ;
- checkpoints internes jamais téléversables ;
- snapshots `safetensors` et manifestes vérifiés ;
- hashes contrôlés avant chaque chargement ;
- identifiants opaques ;
- journaux sans secret, poids ou source privée de récompense lorsque la visibilité ne l’autorise pas ;
- base d’orchestration inaccessible directement depuis MyBlog ;
- futur appel interservice authentifié ;
- limites de taille sur sources, configurations et artefacts.

## 35. Taxonomie minimale des erreurs

| Famille | Exemples | Retry |
| --- | --- | ---: |
| `CONFIG_*` | schéma, domaine, paramètre inconnu | non |
| `REWARD_PARSE_*` | lexing ou syntaxe | non |
| `REWARD_TYPE_*` | type ou appel invalide | non |
| `REWARD_RUNTIME_*` | non-fini, dépassement, limite VM | non |
| `ENV_CONTRACT_*` | observation ou masque invalide | non |
| `ILLEGAL_ACTION_ESCAPE` | action hors masque | non |
| `CHECKPOINT_INCOMPATIBLE` | version ou architecture | non |
| `ARTIFACT_CORRUPT` | hash ou fichier absent | non |
| `WORKER_CRASH` | arrêt du processus | oui, maximum 3 |
| `TEMPORARY_STORAGE` | indisponibilité transitoire | oui, maximum 3 |
| `SERVER_GATE_MISMATCH` | commit ou dépendances différents | non, corriger puis relancer |

Les messages publics sont stables et non sensibles. Les diagnostics internes peuvent inclure une trace, jamais exposée directement à l’utilisateur.

## 36. Frontière avec MyBlog

L’intégration MyBlog est hors du présent lot, mais le contrat est préparé :

- la bibliothèque et la CLI fonctionnent sans Django ;
- la future API appelle les mêmes services d’application ;
- MyBlog ne lance jamais de sous-processus PPO ;
- MyBlog ne lit ni n’écrit la base SQLite du serveur RL ;
- MyBlog reste autorité des comptes, allocations quotidiennes, Elo, arène et économie ;
- le serveur RL reste autorité des expériences, jobs, poids, checkpoints, évaluations et snapshots ;
- le ledger RL garantit l’exécution et la consommation à l’intérieur d’une autorisation MyBlog, sans frapper de nouvelles unités ;
- les pas d’environnement ne transitent jamais par HTTP ;
- les parties interactives et les tests utilisent des tâches prioritaires courtes.

# Partie III — Plan d’implémentation pour agents LLM

## 37. Règles d’exécution des tickets

Chaque ticket doit préciser avant modification :

- fichiers autorisés ;
- tests à écrire en premier ;
- schémas ou interfaces affectés ;
- commandes locales de validation ;
- artefacts attendus ;
- dépendances ;
- condition d’arrêt.

Un agent :

- ne modifie pas les règles du moteur pour faciliter le RL ;
- ne change pas un schéma sans mettre à jour fixtures et tests ;
- ne choisit pas une optimisation à partir d’une mesure locale ;
- ne poursuit jamais après un gate serveur requis non importé ;
- n’invente jamais un résultat de benchmark ;
- ne committe pas les poids, checkpoints ou archives volumineuses ;
- conserve les changements utilisateur non liés.

## 38. Vue d’ensemble des lots

| Lot | Résultat | Dépendances | Arrêt serveur |
| --- | --- | --- | --- |
| RL-L0 | contrats, dépendances, schémas, CLI vide | moteur L7 | — |
| RL-L1 | wrapper, observation, masque, random bot | RL-L0 | — |
| RL-L2 | première unité PPO reproductible | RL-L1 | `RL-S0` |
| RL-L3 | vectorisation instrumentée | RL-L2 | `RL-S1` |
| RL-L4 | Reward DSL et VM | RL-L1 | `RL-S2` |
| RL-L5 | cycle atomique, reprise, priorités courtes | RL-L3, RL-L4 | `RL-S3` |
| RL-L6 | MLP et CNN comparables | RL-L5 | `RL-S4` |
| RL-L7 | métriques et comparaison complète | RL-L6 | `RL-S5` |
| RL-L8 | évaluation, snapshots et publication | RL-L7 | `RL-S6` |
| RL-L9 | file, crédits, progression et rétention | RL-L5, RL-L8 | — |
| RL-L10 | calibration produit et clôture | RL-L9 | `RL-S7` |

## 39. Tickets détaillés

### RL-L0 — Contrats et squelette

**Livrer**

- arborescence `rl/`, `server_checks/`, schémas et tests ;
- lockfiles avec dépendances RL ;
- versions initiales des observations, Reward DSL, checkpoint et snapshot ;
- dataclasses ou modèles stricts de configuration ;
- commandes CLI déclarées, retournant explicitement `not_implemented` ;
- vérification automatique que `core` n’importe pas `rl`.

**Tests avant code**

- JSON valides et invalides ;
- rejet des propriétés inconnues ;
- domaines d’hyperparamètres ;
- sérialisation canonique et hashes ;
- règle de dépendance.

**Acceptation**

- installation locale reproductible ;
- schémas documentés ;
- aucune modification comportementale du moteur.

### RL-L1 — Environnement, observation et adversaires

**Livrer**

- `DinoRLSingleAgentEnv` ;
- canonicalisation A/B ;
- grille 8 × 9 × 9 et quinze scalaires ;
- masque de neuf actions ;
- auto-jeu du tour adverse ;
- `random-legal-v1` ;
- métriques `learner_transitions` et `engine_actions`.

**Tests avant code**

- correspondance des 81 cases sous rotation ;
- mapping exhaustif des directions ;
- toutes les bornes des scalaires ;
- terminaison pendant le tour adverse ;
- répartition indépendante siège/initiative ;
- bot random uniforme sur des masques synthétiques ;
- action illégale sans mutation du moteur.

**Acceptation**

- `gymnasium.utils.env_checker` adapté au masque passe ;
- les fixtures miroir produisent des observations miroir identiques ;
- aucun replay alloué dans la boucle.

### RL-L2 — Unité PPO minimale et gate de compatibilité

**Livrer**

- intégration `MaskablePPO` ;
- MLP provisoire ;
- récompense de référence native ;
- collecte globale de 2 048 transitions ;
- quatre epochs ;
- sauvegarde et rechargement minimal ;
- suite `RL-S0`.

**Tests avant code**

- masque respecté pendant entraînement et prédiction ;
- compte exact de transitions et epochs ;
- toutes les métriques finies ;
- chargement du modèle sauvegardé.

**Condition d’arrêt**

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S0
```

Ne commencer RL-L3 qu’après import d’un `RL-S0` valide.

### RL-L3 — Vectorisation

**Livrer**

- factories d’environnements seedées ;
- `DummyVecEnv` et `SubprocVecEnv` ;
- configurations 2/4/8 ;
- instrumentation end-to-end ;
- suite `RL-S1` et gabarit de décision.

**Tests avant code**

- unicité et reproductibilité des flux RNG ;
- total global toujours égal à 2 048 ;
- mêmes distributions attendues entre backends ;
- fermeture sans processus orphelin.

**Condition d’arrêt**

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S1
```

Le backend et `n_envs` retenus deviennent la référence serveur après import.

### RL-L4 — Reward DSL

**Livrer**

- lexer, parser, AST typé et diagnostics ;
- détection des cycles d’appels ;
- interpréteur de référence ;
- optimiseur, bytecode et VM ;
- catalogue complet de fonctions ;
- cache par hash ;
- décomposition des métriques par terme ;
- suite `RL-S2`.

**Tests avant code**

- grammaire complète ;
- erreurs avec position source ;
- limites de ressources ;
- pureté ;
- tests génératifs référence/VM ;
- programme de référence et exemples stratégiques ;
- calcul paresseux des métriques spatiales.

**Condition d’arrêt**

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S2
```

Une surcharge supérieure à 10 % bloque la suite et déclenche un profilage.

### RL-L5 — Atomicité, reprise et tâches prioritaires

**Livrer**

- runner d’une unité atomique ;
- état de récupération complet ;
- restauration de tous les RNG ;
- transactions de débit idempotentes minimales ;
- simulation de crash ;
- prototypes des deux modes de priorité interactive ;
- suite `RL-S3`.

**Tests avant code**

- panne à chaque étape d’écriture ;
- récupération du dernier état valide ;
- aucune double facturation ;
- équivalence continue/reprise ;
- restauration d’un rollout terminé au milieu de plusieurs parties distinctes ;
- partie prioritaire correctement servie ;
- nettoyage des processus.

**Condition d’arrêt**

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S3
```

La décision concurrence/frontière d’unité est versionnée avant RL-L6.

### RL-L6 — Architectures et smoke

**Livrer**

- MLP et petit CNN ;
- comptage de paramètres ;
- mêmes têtes politique et valeur ;
- configurations strictement comparables ;
- suite `RL-S4` de 10 unités.

**Tests avant code**

- formes de tenseurs ;
- gradients finis ;
- masque identique ;
- export/rechargement ;
- budget et seeds identiques.

**Condition d’arrêt**

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S4
```

Le smoke n’autorise pas encore la suppression d’une architecture.

### RL-L7 — Métriques, évaluations et comparaison

**Livrer**

- catalogue complet de métriques ;
- évaluations déterministes et stochastiques ;
- replays diagnostiques ciblés ;
- rapports JSONL et final ;
- agrégation seeds médiane/IQR ;
- suite `RL-S5` de 147 unités × 3 seeds × 2 architectures.

**Tests avant code**

- score échecs ;
- quatre configurations déterministes ;
- appariement des seeds ;
- agrégats sur fixtures ;
- aucune écriture par transition en entraînement ;
- rapports conformes aux schémas.

**Condition d’arrêt**

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S5
```

Après import, produire `architecture-decision.md`. Si vitesse, temps mur et score final ne convergent pas, demander explicitement la décision de l’utilisateur.

### RL-L8 — Publication et validation finale

**Livrer**

- gate déterministe sur deux évaluations ;
- gate stochastique et extension près des frontières ;
- export `safetensors` ;
- manifeste, résultats et hashes ;
- registre de snapshots immuables ;
- refus des incompatibilités et imports externes ;
- suite `RL-S6`.

**Tests avant code**

- chaque seuil et frontière ;
- cas 4/5 et 3/5 seeds ;
- corruption et mauvais hash ;
- tentative de pickle ou architecture inconnue ;
- immutabilité et double publication ;
- replay par actions.

**Condition d’arrêt**

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S6
```

La V1 n’est pas déclarée techniquement apprenante avant import d’un rapport montrant au moins 4/5 seeds conformes.

### RL-L9 — Orchestration complète et progression

**Livrer**

- SQLite WAL et migrations internes ;
- expériences, runs, jobs, ledger et artefacts ;
- rotation par joueur et priorités P0 à P3 ;
- trois retries techniques ;
- annulation aux frontières ;
- cache d’évaluation ;
- heartbeat 30 secondes ;
- ETA ;
- CLI complète de soumission, suivi et annulation ;
- rétention et épinglage.

**Tests avant code**

- concurrence SQLite contrôlée ;
- réservation simultanée ;
- rotation équitable ;
- priorité des parties ;
- absence de famine sous limites normales ;
- récupération après redémarrage ;
- allègement des checkpoints anciens ;
- progression et ETA sur horloge injectée.

**Acceptation**

- le système fonctionne sans MyBlog ;
- un utilisateur peut soumettre, suivre, annuler, reprendre, tester et publier par CLI ;
- un ancien snapshot reste disponible pendant l’entraînement.

### RL-L10 — Calibration et clôture

**Livrer**

- suite longue `RL-S7` reprenable ;
- extraction de `T_visible`, `T_gate`, `T_plateau` ;
- proposition de budget quotidien ;
- rapport final de capacité ;
- documentation d’exploitation et de reprise.

**Condition d’arrêt**

```text
ACTION UTILISATEUR REQUISE — CHECKPOINT SERVEUR RL-S7
```

Après import, remplacer dans un fichier de décision les valeurs provisoires devenues mesurées. Ne pas réécrire les anciens rapports.

## 40. Définition de terminé

Le sous-système RL V1 est terminé lorsque :

- les tickets RL-L0 à RL-L10 sont acceptés ;
- tous les tests de correction passent localement et en CI ;
- les huit archives serveur ont été importées et validées ;
- une architecture unique est figée ;
- le Reward DSL satisfait exactitude, sécurité et surcharge maximale ;
- la reprise exacte est démontrée sur le serveur ;
- au moins quatre seeds finales sur cinq satisfont le jalon ;
- l’évaluation stochastique n’indique pas d’effondrement manifeste ;
- le budget quotidien est recalibré depuis les courbes réelles ;
- entraînement, suivi, test privé et publication sont utilisables par CLI ;
- les artefacts publiés ne contiennent aucun code exécutable ;
- les décisions de performance sont traçables vers des résultats serveur versionnés ;
- aucun composant ne dépend encore de MyBlog pour fonctionner.

## 41. Hors périmètre V1

- intégration visuelle et API MyBlog ;
- self-play ou ligue de snapshots ;
- curriculum automatique ;
- politiques récurrentes ou attentionnelles ;
- choix d’architecture par le joueur ;
- GPU ;
- entraînements simultanés multiples ;
- import de modèles externes ;
- exécution de Python utilisateur ;
- compteurs ou mémoire de récompense ;
- langage de programmation des bots du futur mode coding game ;
- compilation Cython, C++ ou Rust sans profilage et besoin serveur démontré.

Ces extensions doivent conserver les frontières établies : moteur pur, observations versionnées, artefacts immuables et décisions de performance fondées sur le serveur.

## 42. Références techniques

- [Documentation officielle de MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) — politiques multi-entrées, masques d’action, évaluation masquée et contrainte `SubprocVecEnv`.
- [Gymnasium — Handling Time Limits](https://gymnasium.farama.org/v1.2.3/tutorials/gymnasium_basics/handling_time_limits/) — distinction entre terminaison intrinsèque et troncature externe.
- [Safetensors — API PyTorch](https://huggingface.co/docs/safetensors/api/torch) — sauvegarde et chargement de tenseurs sans objet Python exécutable.
