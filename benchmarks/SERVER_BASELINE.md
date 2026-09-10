# Référence de performance du serveur cible

Statut : **référence réelle pour la suite du développement**  
Date des mesures : **2026-09-10**  
Moteur : `0.1.0` — règles : `1.0.0` — carte : `arena_mvp_v1`

Ce document consolide les deux rapports exécutés sur le serveur cible. Pour les
décisions de capacité, de parallélisme et de coût du replay, cette référence
serveur prévaut sur les mesures réalisées sur un poste de développement. Elle
reste une baseline observée, pas un SLA ni un seuil automatique de test.

## Provenance

| Rapport fourni | Généré à (UTC) | SHA-256 du fichier source |
| --- | --- | --- |
| `baseline-2026-09-10.json` | `2026-09-10T14:06:53.714301+00:00` | `5dfe4a61d07d1f7f038cca5086b74db403e9ce5c1a97b54f12a854a780aad0b8` |
| `replay-cost-2026-09-10.json` | `2026-09-10T14:07:39.262881+00:00` | `a1986935ab991f0b2b4af814177922ff407552b38b3ab1f391fe84060ee6b7a5` |

Les valeurs ci-dessous sont recopiées sans extrapolation depuis ces rapports.
Les fichiers provenaient de l'exécution réelle transmise par le propriétaire du
serveur ; ils ne doivent pas être confondus avec les rapports Windows locaux de
même date présents dans `benchmarks/results`.

## Environnement mesuré

| Élément | Valeur |
| --- | --- |
| Python | CPython `3.12.8` |
| Système | `Linux-6.8.0-1061-aws-x86_64-with-glibc2.35` |
| Architecture / CPU déclaré | `x86_64` |
| CPU logiques visibles | 4 |
| Démarrage multiprocessing | `fork` |

Charge commune aux deux rapports : les 9 paires ordonnées des trois contrôleurs
scriptés, les graines 0 à 19, soit 180 combats par répétition, une répétition
d'échauffement exclue, puis 5 répétitions mesurées et 900 combats au total par
configuration.

## Baseline sans replay

| Processus | Combats/s | Actions/s | Moyenne/combat | p95/combat | Accélération | Pic mémoire/processus |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 207,528 | 12 872,485 | 4,809 ms | 11,566 ms | 1,000× | 23 457 792 octets (22,37 Mio) |
| 2 | 255,688 | 15 859,748 | 7,549 ms | 16,791 ms | 1,232× | 16 392 192 octets (15,63 Mio) |
| 4 | 316,236 | 19 615,399 | 11,912 ms | 34,243 ms | 1,524× | 17 272 832 octets (16,47 Mio) |
| 8 | 375,976 | 23 320,983 | 19,616 ms | 53,306 ms | 1,812× | 17 272 832 octets (16,47 Mio) |

Lecture : sur cet environnement à 4 CPU logiques, le débit continue d'augmenter
jusqu'à 8 processus, mais avec des rendements décroissants et une latence par
combat croissante. Le chiffre à 8 processus décrit le benchmark par lots ; il ne
constitue pas une promesse de débit HTTP ni une recommandation automatique sur
le nombre de workers WSGI.

## Coût du replay sur un processus

| Mode | Combats/s | Actions/s | Moyenne/combat | p95/combat |
| --- | ---: | ---: | ---: | ---: |
| Sans replay | 218,355 | 13 544,052 | 4,579 ms | 9,030 ms |
| Avec replay | 46,569 | 2 888,554 | 21,473 ms | 34,942 ms |

- ratio de débit avec/sans replay : `0,213271` ;
- ralentissement observé avec replay : `78,673 %` ;
- les deux modes ont exécuté exactement 55 825 actions sur 900 combats.

Le débit sans replay de ce rapport et celui de la matrice précédente proviennent
de deux exécutions distinctes. Chacun sert de référence à son propre benchmark ;
ils ne doivent pas être combinés pour recalculer le coût du replay.

## Tailles sérialisées

| Objet | p50 | p95 | Maximum | Limite contractuelle |
| --- | ---: | ---: | ---: | ---: |
| Replay | 63 960 octets | 135 523 octets | 136 056 octets | 491 520 octets (480 Kio) |
| Réponse complète | 64 367 octets | 135 943 octets | 136 476 octets | 524 288 octets (512 Kio) |

Toutes les mesures respectent les limites v1. Le p95 de la réponse complète est
également inférieur à l'objectif de 256 Kio. Les tailles sont identiques aux
mesures locales pour cette charge déterministe, contrairement aux durées et aux
débits qui dépendent de l'hôte.

## Règles d'utilisation pour les prochains tickets

1. Utiliser les valeurs serveur de ce document pour raisonner sur la capacité,
   le parallélisme, les budgets de temps et le coût du replay.
2. Utiliser les mesures locales uniquement pour diagnostiquer une régression ou
   accélérer une boucle de développement ; ne pas les présenter comme capacité
   du serveur cible.
3. Ne créer aucun seuil de CI absolu à partir du débit ou des latences : ils sont
   sensibles à la charge de l'hôte. Toute suspicion de régression doit être
   confirmée avec la même charge, un échauffement séparé et au moins cinq
   répétitions mesurées.
4. Continuer d'appliquer comme contraintes dures les maxima contractuels de
   480 Kio pour le replay et 512 Kio pour la réponse complète.
5. Ne pas déduire le débit HTTP des combats/s : les mesures n'incluent ni réseau,
   ni proxy, ni ordonnanceur WSGI. Les smoke tests HTTP à quatre demandes et la
   limite de cinq secondes restent des validations séparées.
6. Rejouer et remplacer cette référence après toute modification du moteur, des
   règles, des contrôleurs, du format replay, de Python ou de l'environnement
   serveur susceptible d'affecter les résultats. Conserver alors les rapports
   bruts et leurs SHA-256 avec la nouvelle référence.
7. Aucune extension native ne doit être ouverte sur la seule base de l'écart
   poste/serveur. Une telle décision doit partir d'un besoin produit chiffré et
   d'un profilage réalisé sur la charge serveur représentative.
