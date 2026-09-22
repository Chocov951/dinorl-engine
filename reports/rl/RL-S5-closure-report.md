# RL-S5 — rapport de clôture

Statut : `RL_S5_CLOSED_READY_FOR_RL_L8`

## Campagnes découvertes

- `RL-S5`
- `RL-S5b`
- `RL-S5c-A1`
- `RL-S5c-A2`
- `RL-S5c-A3-CONTROL`
- `RL-S5c-A3-PILOT`
- `RL-S5c-A3-PILOT-R2`

## Sources et artefacts

- Rétrospective : `docs/rl/RL-S5-retrospective.md`
- Registre : `artifacts/rl/RL-S5-registry.json`
- Artefacts indexés : 95 ; locaux : 94 ; manquants : 1.
- Pool exécutable : 67 politiques ; hash `6cfc4b18669e3c1e830c5d1b65553e2f7e4be6092482de06ee0035fa14951d08`.
- Starter zéro : hash `50018e5c7b87ea369d0e01b4f2d4be9f1a24e3cab554c8170886fd6fb3c065e5`.
- Tous les artefacts locaux du registre possèdent un SHA-256 recalculé ; les absences restent explicitement marquées `missing`.

## Décisions

- Architecture V1 : `mlp-compact-v2`.
- Origine commune des joueurs : `starter-zero-v1`, non entraîné.
- Spécialistes : aucun starter publié ; `NO_ELIGIBLE_VARIANT`.
- `controller` : `deferred_post_v1`.
- Aucun entraînement ni benchmark n'a été exécuté par la clôture.
- RL-S6 reste non exécuté ; sa configuration canonique doit faire l'objet d'une décision séparée avant le checkpoint serveur.

## Spécification et validation

La spécification canonique inscrit `mlp-compact-v2`, `starter-zero-v1`, la séparation bots officiels/validation/diagnostic, le report des spécialistes post-V1 et la clôture de RL-S5.

Commandes exécutées :

```text
python -m dinorl_engine.rl s5 close --repository . --discover-artifacts --build-validation-pool --create-zero-starter --quiet --json
python -m dinorl_engine.rl s5 audit-closure --repository . --json
ruff check src tests
ruff format --check src tests
mypy --strict src/dinorl_engine/core
pytest -q
```

Résultats : 17 tests ciblés réussis ; suite complète `654 passed, 2 skipped` ; Ruff, format et mypy strict réussis.

Dette conservée : la spécialisation et `controller` sont post-V1 ; la configuration RL-S6 doit être décidée et figée avant sa campagne cinq seeds.

## Passage de relais

RL-L8 peut commencer avec le registre, le starter zéro et le pool de validation. RL-S6 ne doit pas être déclaré réussi avant son exécution sur cinq seeds.

RL-S5 CLÔTURÉ — REPRISE AUTORISÉE À RL-L8
