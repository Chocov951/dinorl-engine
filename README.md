# DinoRL Engine

Moteur déterministe du **Duel de raptors**. Ce dépôt hébergera le noyau de règles,
les contrôleurs scriptés, l'exécution de matchs, les replays et le service HTTP de
DinoRL.

Le noyau du jeu est livré jusqu'au lot moteur L7. Les lots **RL-L0** à
**RL-L2** ajoutent les contrats versionnés, les schémas JSON stricts, le
wrapper Gymnasium mono-agent avec masque d'actions, la récompense de référence
et une unité CPU `MaskablePPO` de 2 048 transitions. La suite serveur `RL-S0`
se lance avec `python -m dinorl_engine.server_checks run --suite RL-S0` sur un
commit propre ; l'archive résultante s'importe avec
`python -m dinorl_engine.server_checks import <archive>`. La CLI d'entraînement
générale reste déclarative jusqu'aux lots suivants.

## Contrats RL

Les schémas machine v1 se trouvent dans [`schemas/rl/`](schemas/rl/) :
configuration d'expérience, exécution, progression, évaluation, manifeste de
checkpoint de confiance et manifeste de snapshot publié. Les exemples valides
associés sont dans [`fixtures/rl/contracts/`](fixtures/rl/contracts/). Tous les
objets interdisent les propriétés inconnues et sont hashés à partir d'un JSON
UTF-8 compact, trié et sans valeurs non finies.

## Prérequis

- CPython 3.12 ;
- `pip`.

## Installation de développement

Sous Linux ou macOS :

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pip install --no-deps -e .
```

Sous Windows PowerShell :

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.lock
.venv\Scripts\python -m pip install --no-deps -e .
```

## Vérifications

```bash
ruff check src tests
ruff format --check src tests
mypy --strict src/dinorl_engine/core src/dinorl_engine/rl src/dinorl_engine/server_checks
pytest
```

Les règles fonctionnelles et le plan d'implémentation se trouvent dans [`docs/`](docs/).
