# Consignes pour les agents d'implémentation

## Sources de vérité

1. `docs/duel-de-raptors-regles-du-jeu.md` définit les règles fonctionnelles.
2. `docs/duel-de-raptors-plan-implementation.md` définit l'architecture, les contrats
   et l'ordre des tickets.
3. Les schémas JSON versionnés et les fixtures de conformité complèteront ces documents.

En cas de contradiction, arrêter l'implémentation concernée et signaler le désaccord.
Ne pas déduire de règle à partir de l'ancien DinoWars.

## Discipline d'implémentation

- Travailler ticket par ticket, dans l'ordre des dépendances du plan.
- Suivre une boucle TDD : observer le test rouge, écrire le minimum, puis vérifier.
- Ne modifier que les fichiers autorisés par le ticket en cours.
- Ne pas créer de branche, commit ou pull request.
- Ne pas ajouter de dépendance, fonctionnalité ou règle absente du plan.
- Ne jamais ajouter de secret au dépôt.
- Préserver la séparation du noyau : `core` dépend uniquement de la bibliothèque standard.
- Conserver un comportement déterministe et des types stricts dans `core`.

## Validation

Utiliser Python 3.12 ou une version ultérieure, puis exécuter les contrôles adaptés au
ticket :

```bash
ruff check src tests
ruff format --check src tests
mypy --strict src/dinorl_engine/core
pytest
```

Le noyau complet devra atteindre au moins 95 % de couverture lorsque les tickets de
règles seront implémentés.

## Compte rendu de ticket

Chaque ticket se termine avec le format imposé à la section 30.3 du plan : fichiers
modifiés, test rouge observé, commandes et résultats, couverture, décisions, risques
et travail restant.

