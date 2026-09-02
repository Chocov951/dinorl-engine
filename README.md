# DinoRL Engine

Moteur déterministe du **Duel de raptors**. Ce dépôt hébergera le noyau de règles,
les contrôleurs scriptés, l'exécution de matchs, les replays et le service HTTP de
DinoRL.

Le projet est actuellement à l'étape **ENG-001** : seul l'échafaudage du package et
des outils de développement est présent. Aucune règle de jeu n'est encore
implémentée.

## Prérequis

- Python 3.12 ou une version ultérieure ;
- `pip`.

## Installation de développement

Sous Linux ou macOS :

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Sous Windows PowerShell :

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Vérifications

```bash
ruff check src tests
ruff format --check src tests
mypy --strict src/dinorl_engine/core
pytest
```

Les règles fonctionnelles et le plan d'implémentation se trouvent dans [`docs/`](docs/).

