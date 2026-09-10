# Déployer DinoRL Engine sur PythonAnywhere

Cette procédure déploie le service Flask en WSGI avec Python 3.12, sans Docker
et sans dépendance à MyBlog. Les commandes sont exécutées dans une console Bash
PythonAnywhere. Remplacer `<user>` et `<domain>` par les valeurs du compte.

## Configuration requise

| Variable | Obligatoire | Usage |
| --- | --- | --- |
| `DINORL_ENGINE_TOKEN` | oui | jeton Bearer courant, au moins 32 octets |
| `DINORL_ENGINE_PREVIOUS_TOKEN` | non | ancien jeton accepté temporairement pendant une rotation |

Les jetons ne doivent être placés ni dans le dépôt, ni dans les logs. Le service
refuse les simulations si aucun jeton valide n'est configuré ; `/health` reste
public. HTTPS doit être activé pour le domaine de production.

## Installation reproductible

Transférer la révision validée dans un répertoire versionné, par exemple
`/home/<user>/releases/dinorl-engine-0.1.0`, puis vérifier l'interpréteur :

```bash
cd /home/<user>/releases/dinorl-engine-0.1.0
python3.12 --version
python3.12 -m venv /home/<user>/.virtualenvs/dinorl-engine-0.1.0
/home/<user>/.virtualenvs/dinorl-engine-0.1.0/bin/python -m pip install \
  -r requirements.lock
/home/<user>/.virtualenvs/dinorl-engine-0.1.0/bin/python -m pip install --no-deps -e .
```

L'installation éditable est volontaire : `schemas/*.json` reste l'unique source
des contrats et le service les charge depuis le dépôt transféré. Elle n'autorise
aucune modification du répertoire de release après activation.

Créer les secrets hors du dépôt avec des permissions privées :

```bash
umask 077
mkdir -p /home/<user>/.config
python3.12 -c "import secrets; print(secrets.token_urlsafe(48))" \
  > /home/<user>/.config/dinorl-engine-token
```

Le fichier facultatif `dinorl-engine-previous-token` utilise les mêmes
permissions et n'existe que pendant une rotation.

## Configuration WSGI

Dans l'onglet **Web**, créer une application en configuration manuelle Python
3.12, sélectionner le virtualenv ci-dessus, configurer le répertoire de travail
sur le répertoire de release et forcer HTTPS. Remplacer le contenu du fichier
WSGI géré par PythonAnywhere par :

```python
import os
import sys
from pathlib import Path

project_home = Path("/home/<user>/releases/dinorl-engine-0.1.0")
sys.path.insert(0, str(project_home))

secret_home = Path("/home/<user>/.config")
os.environ["DINORL_ENGINE_TOKEN"] = (
    (secret_home / "dinorl-engine-token").read_text(encoding="utf-8").strip()
)
previous_token = secret_home / "dinorl-engine-previous-token"
if previous_token.exists():
    os.environ["DINORL_ENGINE_PREVIOUS_TOKEN"] = previous_token.read_text(encoding="utf-8").strip()
else:
    os.environ.pop("DINORL_ENGINE_PREVIOUS_TOKEN", None)

from deploy.pythonanywhere_wsgi import application
```

Le serveur PythonAnywhere importe la variable WSGI `application`. Ne pas lancer
`app.run()` et ne pas démarrer de serveur ou de processus d'entraînement dans la
route. Recharger ensuite l'application depuis l'onglet **Web**.

## Smoke tests

Charger le jeton dans la console sans l'afficher :

```bash
export DINORL_ENGINE_TOKEN="$(cat /home/<user>/.config/dinorl-engine-token)"
export DINORL_ENGINE_URL="https://<domain>"
```

Vérifier d'abord `GET /health` et l'échec fermé sans authentification :

```bash
python3.12 - <<'PY'
import json
import os
import urllib.error
import urllib.request

base_url = os.environ["DINORL_ENGINE_URL"]
with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
    assert response.status == 200
    assert json.load(response)["status"] == "ok"

request = urllib.request.Request(
    f"{base_url}/v1/matches/simulate",
    data=b"{}",
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    urllib.request.urlopen(request, timeout=5)
except urllib.error.HTTPError as error:
    assert error.code == 401
else:
    raise AssertionError("une simulation sans jeton a été acceptée")
PY
```

Puis exécuter une simulation authentifiée et quatre demandes parallèles. Le test
échoue si l'ensemble parallèle dépasse cinq secondes, si une réponse dépasse
512 Kio ou si un hash de replay diverge :

```bash
python3.12 - <<'PY'
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
import uuid

url = os.environ["DINORL_ENGINE_URL"] + "/v1/matches/simulate"
token = os.environ["DINORL_ENGINE_TOKEN"]

def simulate(seed):
    payload = {
        "protocol_version": "1.0.0",
        "request_id": str(uuid.uuid4()),
        "rules_version": "1.0.0",
        "map_id": "arena_mvp_v1",
        "seed": seed,
        "controllers": {
            "A": {"kind": "scripted", "id": "aggressive-v1"},
            "B": {"kind": "scripted", "id": "opportunist-v1"},
        },
        "include_replay": True,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "DinoRL-deployment-smoke/1.0.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        body = response.read(524289)
        assert response.status == 200 and len(body) <= 524288
    document = json.loads(body)
    replay = json.dumps(
        document["replay"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(replay).hexdigest() == document["replay_sha256"]

simulate(0)
started = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    list(executor.map(simulate, range(1, 5)))
assert time.perf_counter() - started < 5, "quatre simulations dépassent cinq secondes"
PY
```

Après les fumées, vérifier les journaux d'accès et d'erreur : aucun en-tête
`Authorization`, cookie, jeton, contenu de requête ou replay complet ne doit y
apparaître.

## Rotation du jeton

1. Copier temporairement le jeton courant vers
   `/home/<user>/.config/dinorl-engine-previous-token`.
2. Générer un nouveau `dinorl-engine-token` avec la commande sécurisée ci-dessus.
3. Recharger l'application, configurer MyBlog avec le nouveau jeton et exécuter
   les smoke tests.
4. Supprimer le fichier du jeton précédent, recharger et revérifier les routes.

## Retour arrière

Conserver le répertoire et le virtualenv de la version précédente. En cas
d'échec, remettre dans le fichier WSGI son ancien `project_home`, son ancien
virtualenv et, si nécessaire, le jeton précédent ; recharger l'application,
puis vérifier `/health` et une simulation authentifiée. Ne supprimer la release
défaillante qu'après retour complet du service et conservation de ses logs.
