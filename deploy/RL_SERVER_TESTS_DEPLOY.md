# Réinstallation serveur — `dinorl-engine-0.1.0`

À exécuter dans une console Bash du serveur (PythonAnywhere), après le `pull`. Remplacer `DinoRL` et `<branche>` si nécessaire.

```bash
APP_ROOT="/home/DinoRL/releases/dinorl-engine-0.1.0"
VENV="/home/DinoRL/.virtualenvs/dinorl-engine-0.1.0"

cd "$APP_ROOT"

git branch --show-current
git pull --ff-only origin <branche>
git rev-parse HEAD
```

Créer un virtualenv propre, en conservant l’ancien comme sauvegarde :

```bash
if [ -d "$VENV" ]; then
  mv "$VENV" "${VENV}.backup-$(date +%Y%m%d-%H%M%S)"
fi

python3.12 --version
python3.12 -m venv "$VENV"

"$VENV/bin/python" -m pip install -r requirements-dev.lock
"$VENV/bin/python" -m pip install --no-deps -e .
"$VENV/bin/python" -m pip check
```

Vérifier les imports de la pile RL :

```bash
"$VENV/bin/python" -c "
import gymnasium, torch, stable_baselines3, sb3_contrib
from dinorl_engine.core.engine import DinoRLEnv
print('Imports RL OK')
"
```

Lancer les validations de correction :

```bash
"$VENV/bin/ruff" check src tests
"$VENV/bin/ruff" format --check src tests
"$VENV/bin/mypy" --strict src/dinorl_engine/core src/dinorl_engine/rl src/dinorl_engine/server_checks
"$VENV/bin/pytest" tests/rl
```

Avant RL-S0, le dépôt ne doit avoir aucune modification suivie :

```bash
git diff --quiet
git diff --cached --quiet
```

Puis lancer le gate serveur avec le Python du virtualenv :

```bash
"$VENV/bin/python" -m dinorl_engine.server_checks run --suite RL-S0
```

La commande crée une archive dans le dépôt :

```bash
ls -lh server-results-RL-S0-*.tar.gz
```

Télécharger ensuite cette archive vers l’environnement local, puis, sur le poste local au même commit :

```bash
.venv/bin/python -m dinorl_engine.server_checks import server-results-RL-S0-<run_id>.tar.gz
```

Si le service Flask est actif, vérifier dans l’onglet **Web** que le virtualenv pointe vers :

```text
/home/DinoRL/.virtualenvs/dinorl-engine-0.1.0
```

Puis recharger l’application et contrôler :

```bash
curl -fsS https://<votre-domaine>/health
```

Ne supprimer le virtualenv sauvegardé qu’après validation de RL-S0 et des smoke tests. La procédure WSGI et les tests authentifiés complets sont dans [deploy/README.md](C:/Users/louis.remy/OneDrive%20-%20Acelys/Bureau/Projets/Chat_acelys/dinorl-engine/deploy/README.md:1).