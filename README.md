# DinoRL Engine

Moteur déterministe du **Duel de raptors**. Ce dépôt hébergera le noyau de règles,
les contrôleurs scriptés, l'exécution de matchs, les replays et le service HTTP de
DinoRL.

Le noyau du jeu est livré jusqu'au lot moteur L7. Les lots **RL-L0** à
**RL-L3** ajoutent les contrats versionnés, les schémas JSON stricts, le
wrapper Gymnasium mono-agent avec masque d'actions, la récompense de référence
et une unité CPU `MaskablePPO` de 2 048 transitions. RL-L3 ajoute la matrice
vectorisée `DummyVecEnv`/`SubprocVecEnv` pour 2, 4 et 8 environnements. Les
suites serveur `RL-S0`, `RL-S1` et `RL-S2` se lancent sur un commit propre ; leurs
archives résultantes s'importent avec
`python -m dinorl_engine.server_checks import <archive>`. La CLI d'entraînement
générale reste déclarative jusqu'aux lots suivants.

Sur PythonAnywhere, le profil versionné `pythonanywhere` réutilise le build CPU
Torch fourni par la plateforme dans un virtualenv créé avec
`--system-site-packages`. Il s'exécute avec
`python -m dinorl_engine.server_checks run --suite RL-Sx --profile pythonanywhere`.

## Diagnostic local RL-S5

La comparaison complète RL-S5 peut être lancée localement pour diagnostiquer
les architectures, avec reprise et progression visuelle :

```powershell
.venv\Scripts\python.exe -m dinorl_engine.server_checks local-rl-s5 `
  --output-dir "$env:USERPROFILE\Downloads\dinorl-rl-s5-local" `
  --progress
```

Le rapport `rl-s5-local-diagnostic.json` est volontairement non importable :
seule une mesure sur le serveur cible peut sélectionner l'architecture selon
la spécification.

La variante MLP V2 locale exclut le CNN et compare trois nouveaux MLP au
`mlp-v1` dans un tournoi sans self-play :

```powershell
.venv\Scripts\python.exe -m dinorl_engine.server_checks local-rl-s5-v2 `
  --output-dir "benchmarks\server\RL-S5" `
  --progress `
  --json
```

Elle réutilise les checkpoints finaux de `.rl-s5-local-work`, reprend ses
propres entraînements dans `.rl-s5-v2-local-work` et écrit
`rl-s5-v2-local-diagnostic.json`.

## RL-S5b — checkpoint serveur A

Après que les douze checkpoints RL-S5 sont disponibles, le protocole RL-S5b
fige leur hash dans un pool, puis exécute la matrice stochastique appariée. La
configuration versionnée est [configs/rl/s5b.yaml](configs/rl/s5b.yaml) ; elle
est écrite en JSON, qui est un sous-ensemble valide de YAML, pour ne pas
ajouter de dépendance au serveur.

Sur le serveur, depuis la racine du dépôt :

```bash
python -m dinorl_engine.rl s5b freeze-pool --config configs/rl/s5b.yaml --json
# Reprendre avec le chemin `pool` retourné :
python -m dinorl_engine.rl s5b cross-evaluate \
  --pool artifacts/rl/s5b/rl-s5b-default/pools/<pool_sha256>.json --json
python -m dinorl_engine.rl s5b archive \
  --run artifacts/rl/s5b/rl-s5b-default \
  --output-dir . --json
```

`cross-evaluate` reprend chaque paire/seed terminée sans la rejouer. Il produit
`crossplay.json`, `crossplay.csv`, `crossplay-report.md` et le manifeste de
campagne, sans modifier les poids. L’archive `server-results-RL-S5b-A-*.tar.gz`
ne contient aucun poids et s’importe localement avec :

```powershell
.venv\Scripts\python.exe -m dinorl_engine.server_checks import "<archive>.tar.gz"
```

Ne pas lancer `s5b continue`, `s5b self-play` ou `s5b train-exploiters` avant
l’import et l’analyse du checkpoint serveur A.

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
