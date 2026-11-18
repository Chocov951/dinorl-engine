# RL-L9 — orchestration locale V1

## Livré

- base SQLite dédiée, WAL et migration interne (`user_version=2`) ;
- expériences, runs, jobs, événements, unités validées, artefacts, évaluations et cache ;
- priorités P0–P3, rotation par propriétaire à chaque frontière d’unité ;
- retries techniques bornés à trois, annulation normale à la frontière d’unité et restitution ;
- heartbeats limités à 30 secondes, projection d’ETA injectée et explicitement provisoire sans mesure ;
- cache d’évaluation idempotent et rétention excluant les artefacts épinglés ;
- CLI de soumission, reprise, évaluation, publication protégée par éligibilité, inspection, état et annulation ;
- worker mono-processus qui ne débite qu’après validation de l’unité.

## Validation

Les contrats TDD couvrent WAL, migration concurrente, réservation simultanée,
priorités, rotation, annulation, retries, cache, rétention, worker, progression et CLI.
Les contrôles statiques du sous-système passent. Les mesures de capacité restent réservées
au serveur conformément à la spécification ; aucune n’est utilisée ici.

## Décisions

Le worker reste mono-processus et les tâches P0–P2 sont sélectionnées avant la prochaine
unité P3. Les données lourdes ne transitent pas par SQLite : seuls leurs chemins, hashes,
tailles et protections y sont conservés.
