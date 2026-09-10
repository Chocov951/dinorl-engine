# DinoRL — Instructions d’implémentation des outils de débogage local

> **Statut : prêt à implémenter après le Gate L5**  
> **Périmètre : affichage texte, contrôle manuel et séquences d’actions prédéfinies**  
> **Méthode : TDD, tickets exécutés dans l’ordre**

## 1. Objectif

Ajouter au dépôt `dinorl-engine` trois outils locaux permettant de tester et de reproduire un combat sans MyBlog ni RL :

1. afficher dans un terminal la carte et l’état d’un replay ;
2. contrôler manuellement un raptor ;
3. piloter un raptor avec une séquence d’actions écrite à l’avance.

Ces outils sont des adaptateurs de débogage. Ils utilisent le moteur existant sans modifier les règles du jeu.

## 2. Moment d’implémentation

L’implémentation commence après le **Gate L5**, lorsque les éléments suivants existent et sont stables :

- moteur de règles complet ;
- `PublicState` sérialisable ;
- événements et replay v1 ;
- protocole `Controller` ;
- `MatchRunner` ;
- bots déterministes ;
- exécutable CLI initial.

Cette extension est compatible avec L6 et L7. Elle ne doit pas modifier les résultats des benchmarks sans replay et ne doit jamais être importée par `dinorl_engine.core`.

## 3. Arborescence

```text
src/dinorl_engine/
├── debug/
│   ├── __init__.py
│   ├── manual.py
│   └── text_renderer.py
├── controllers/
│   └── sequence.py
└── cli.py

fixtures/debug/
├── bite_then_retreat.json
└── interrupted_feed.json

tests/
├── debug/
│   ├── test_manual_controller.py
│   └── test_text_renderer.py
├── controllers/
│   └── test_sequence_controller.py
└── integration/
    └── test_debug_cli.py
```

## 4. Affichage texte

### 4.1 Interface

Créer une fonction pure :

```python
def render_state(
    state: PublicState,
    *,
    event: ReplayEvent | None = None,
) -> str:
    ...
```

Elle :

- reçoit uniquement un état public et éventuellement l’événement courant ;
- renvoie une chaîne complète ;
- n’écrit pas directement dans `stdout` ;
- ne recalcule aucune règle ;
- n’accède pas à l’état mutable interne du moteur ;
- produit toujours le même texte pour la même entrée.

### 4.2 Symboles

| Symbole | Élément |
| --- | --- |
| `A` | raptor A |
| `B` | raptor B |
| `/` | mur |
| `~` | boue |
| `l` | carcasse latérale disponible |
| `C` | carcasse centrale active |
| `c` | carcasse centrale en attente ou en recharge |
| `.` | case libre |

Un raptor masque visuellement le symbole de la case qu’il occupe. L’état de cette case et celui des carcasses restent indiqués dans le panneau situé sous la grille.

### 4.3 Format minimal

```text
Round 4 — Turn 7 — Actor A
A: HP=6 END=3 PM=2 SCORE=1 MAIN=yes
B: HP=4 END=1 PM=0 SCORE=0 MAIN=no

. . . . . . . . B
. . . . . / . . .
. . / . . . . ~ ~
l . ~ . ~ / . . .
/ / . . C . . / /
. . . / ~ . ~ . l
~ ~ . . . . / . .
. . . / . . . . .
A . . . . . . . .

Central carcass: active
Pending feed: none
Pending rest: B
Last action: B SHOVE A
```

Le format exact est gelé par des tests snapshot. Les lignes se terminent par `\n`, sans codes ANSI, afin que les sorties restent comparables dans la CI. Les lignes doivent intégrer un espace entre chaque symbole de la grille. L'espace ne représente rien et n'est pas un symbole de case, il est uniquement destiné à améliorer la lisibilité. 

### 4.4 Lecture d’un replay

Ajouter :

```bash
python -m dinorl_engine.cli inspect-map
python -m dinorl_engine.cli replay path/to/replay.json
python -m dinorl_engine.cli replay path/to/replay.json --step
python -m dinorl_engine.cli replay path/to/replay.json --speed 2
python -m dinorl_engine.cli replay path/to/replay.json --event 42
```

Règles :

- `inspect-map` affiche l’état initial de `arena_mvp_v1` ;
- sans option, `replay` affiche les événements successivement à vitesse 1× ;
- `--step` attend Entrée entre deux événements ;
- `--speed` accepte `0.25`, `0.5`, `1`, `2` ou `4` ;
- `--event N` affiche uniquement l’événement demandé ;
- un replay invalide est refusé avant le premier affichage ;
- les temporisations sont injectables afin que les tests ne dorment jamais.

## 5. Contrôle manuel

### 5.1 Interface

Créer `ManualController`, conforme au protocole `Controller` existant.

Il reçoit par injection :

- une fonction de lecture ;
- une fonction d’écriture ;
- le renderer texte.

Il ne doit pas appeler directement `input()` ou `print()` dans sa logique testable.

À chaque décision, il :

1. affiche l’état courant ;
2. affiche uniquement les actions légales ;
3. accepte le numéro ou le nom exact d’une action ;
4. redemande une saisie après une valeur inconnue ;
5. retourne l’action choisie au `MatchRunner`.

Exemple :

```text
Legal actions:
1. MOVE_NORTH
2. MOVE_EAST
3. BITE
4. END_TURN

Action >
```

### 5.2 Arrêt

- `EOF` ou `Ctrl+C` annule proprement le match avec un code de sortie non nul ;
- l’annulation ne doit pas être convertie silencieusement en `END_TURN` ;
- une saisie invalide ne modifie jamais le moteur ;
- le contrôleur manuel n’est disponible que dans la CLI locale.

### 5.3 Commandes

```bash
python -m dinorl_engine.cli play \
  --controller-a manual \
  --controller-b opportunist-v1

python -m dinorl_engine.cli play \
  --controller-a aggressive-v1 \
  --controller-b manual
```

Deux contrôleurs manuels peuvent être utilisés dans le même terminal.

## 6. Séquences prédéfinies

### 6.1 Format

Une séquence est un fichier JSON organisé par tours propres au contrôleur :

```json
{
  "schema_version": "1.0.0",
  "controller_id": "interrupted-feed-a",
  "turns": [
    {"actions": ["MOVE_NORTH", "MOVE_NORTH", "END_TURN"]},
    {"actions": ["MOVE_EAST", "BITE", "END_TURN"]},
    {"actions": ["MOVE_SOUTH", "FEED"]}
  ]
}
```

Chaque entrée de `turns` correspond à un tour du raptor concerné, indépendamment du nombre de tours joués par l’adversaire.

### 6.2 Validation statique

Avant le match, refuser :

- une version inconnue ;
- une action inconnue ;
- un tableau de tours vide ;
- un tour sans action ;
- une action placée après `FEED`, `REST` ou `END_TURN` ;
- un tour ne se terminant pas par `FEED`, `REST` ou `END_TURN` ;
- des propriétés JSON supplémentaires non prévues.

La validation statique ne cherche pas à prédire la légalité dépendant du combat.

### 6.3 Exécution stricte

Lorsqu’il est interrogé, `SequenceController` retourne l’action suivante du tour courant.

- Si l’action est légale, elle est envoyée au moteur.
- Si elle est illégale dans l’état réellement obtenu, le match est arrêté.
- Si la séquence est épuisée avant la fin du match, le match est arrêté.
- Si le match se termine avant la fin du fichier, les actions restantes sont simplement ignorées.
- Aucune action de secours n’est choisie automatiquement.

Erreurs typées :

```text
SequenceSchemaError
SequenceIllegalActionError
SequenceExhaustedError
```

Le diagnostic contient :

- contrôleur et acteur ;
- manche et tour ;
- index du tour scripté ;
- index de l’action ;
- action demandée ;
- actions légales ;
- raison fournie par le moteur ;
- rendu texte de l’état.

### 6.4 Commandes

```bash
python -m dinorl_engine.cli play \
  --controller-a sequence:fixtures/debug/interrupted_feed.json \
  --controller-b aggressive-v1

python -m dinorl_engine.cli play \
  --controller-a sequence:fixtures/debug/plan_a.json \
  --controller-b sequence:fixtures/debug/plan_b.json \
  --replay-out replay.json
```

Le préfixe `sequence:` n’est interprété que par la CLI locale. Le service HTTP :

- ne reconnaît pas le type de contrôleur `sequence` ;
- n’accepte jamais un chemin de fichier ;
- ne charge jamais un module Python fourni par l’appelant.

## 7. Compatibilité future avec le coding game

Ces outils doivent réutiliser le protocole générique `Controller`, sans ajouter de dépendance du noyau vers les contrôleurs.

Ils préparent un futur mode de programmation, mais n’implémentent pas :

- l’éditeur MyBlog ;
- le stockage de programmes utilisateur ;
- un langage de règles ;
- l’exécution de Python utilisateur ;
- une sandbox ;
- le classement no-RL.

Le futur mode public devra privilégier un langage de règles borné et interprété. Il ne devra jamais employer `eval`, `exec` ou charger directement du Python envoyé par un utilisateur sur MyBlog ou sur le service moteur.

## 8. Tickets TDD

### DBG-001 — Renderer texte

- **Dépendance :** Gate L5.
- **Test rouge :** l’état initial ne possède aucun rendu texte stable.
- **Implémentation :** `render_state`, symboles, panneaux et tests snapshot.
- **Acceptation :** carte 9 × 9 correcte ; états de carcasse visibles ; aucune règle recalculée ; sortie déterministe sans ANSI.

### DBG-002 — Lecture terminal d’un replay

- **Dépendance :** DBG-001.
- **Test rouge :** la CLI ne sait ni lire ni sélectionner un événement.
- **Implémentation :** `inspect-map`, `replay`, `--step`, `--speed`, `--event`.
- **Acceptation :** validation préalable ; horloge injectée ; aucune attente réelle dans les tests ; codes de sortie documentés.

### DBG-003 — Contrôleur manuel

- **Dépendance :** DBG-001 et protocole `Controller` de L5.
- **Test rouge :** aucune action ne peut provenir d’une source de saisie injectée.
- **Implémentation :** parsing, boucle de saisie, annulation propre et branchement CLI.
- **Acceptation :** uniquement les actions légales sont proposées ; saisie invalide sans mutation ; manuel contre bot et manuel contre manuel fonctionnels.

### DBG-004 — Contrôleur de séquence

- **Dépendance :** protocole `Controller` et `MatchRunner` de L5.
- **Test rouge :** une fixture d’actions ne peut pas piloter un combat.
- **Implémentation :** schéma JSON, parseur, contrôleur strict et erreurs typées.
- **Acceptation :** validation statique complète ; action illégale et épuisement détectés ; deux séquences peuvent s’affronter ; aucune action de secours.

### DBG-005 — Tests d’intégration et non-exposition HTTP

- **Dépendance :** DBG-002 à DBG-004 ; peut précéder L7 et être complété après L7.
- **Test rouge :** les commandes complètes et le rejet HTTP ne sont pas vérifiés.
- **Implémentation :** tests subprocess/CLI, fixtures canoniques et test de rejet du type `sequence` par le service.
- **Acceptation :** replays reproductibles ; diagnostics lisibles ; les contrôleurs debug sont inaccessibles par HTTP.

## 9. Commandes de validation

```bash
.venv/bin/pytest tests/debug tests/controllers/test_sequence_controller.py
.venv/bin/pytest tests/integration/test_debug_cli.py
.venv/bin/ruff check src/dinorl_engine/debug src/dinorl_engine/controllers/sequence.py
.venv/bin/ruff format --check src/dinorl_engine/debug src/dinorl_engine/controllers/sequence.py
.venv/bin/mypy --strict src/dinorl_engine/debug src/dinorl_engine/controllers/sequence.py
```

Après DBG-005, exécuter également toute la suite moteur afin de démontrer l’absence de régression :

```bash
.venv/bin/pytest
```

## 10. Définition de fini

L’extension est terminée lorsque :

- la carte et chaque événement d’un replay peuvent être affichés dans le terminal ;
- un humain peut jouer contre chacun des trois bots ;
- deux humains peuvent jouer dans le même terminal ;
- une ou deux séquences JSON peuvent piloter un combat complet ;
- une action scriptée devenue illégale arrête le match avec l’état et les actions légales ;
- les sorties texte sont déterministes et testées ;
- les tests n’attendent jamais une saisie ou une temporisation réelle ;
- les outils debug ne sont pas importés par le noyau ;
- le chemin rapide `replay=False` ne subit pas de régression mesurable ;
- aucun contrôleur manuel, séquentiel ou chemin local n’est accessible par le service HTTP ;
- la suite complète du moteur reste verte.

## 11. Consigne à donner à Codex

```text
Implémente les tickets DBG-001 à DBG-005 dans l’ordre et en TDD.
Commence uniquement par DBG-001. Ne crée pas de branche, commit ou pull request.
Ne modifie ni les règles, ni l’API du noyau, ni les contrats de replay sans blocage explicite.
À la fin de chaque ticket, fournis les fichiers modifiés, le test rouge observé,
les commandes exécutées, leurs résultats et les risques restants. Attends ma validation
avant de commencer le ticket suivant.
```
