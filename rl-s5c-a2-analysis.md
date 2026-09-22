# Analyse RL-S5c-A2 — calibration des bots spécialisés

## Décision

**NO-GO global. Ne pas lancer la production sur cinq seeds.**

La campagne est techniquement complète, mais aucun des neuf candidats ne satisfait la
bande de force imposée face au pool RL-S5b. `scavenger` est le seul archétype à atteindre
un gate de compétence stable contre les bots de calibration. `predator` et `controller`
épuisent tous leurs budgets sans gate stable.

Cette décision ne repose pas sur les durées locales. Elle repose uniquement sur les
résultats fonctionnels archivés par la campagne.

## Résultat de la calibration

| Archétype | Seed | Arrêt | Gate stable | Score final calibration | V/N/D | Limite 30 manches |
|---|---:|---:|:---:|---:|---:|---:|
| scavenger | 19 | unité 65 | oui | 0,9980 | 1231/5/0 | 0,40 % |
| scavenger | 20 | unité 95 | oui | 0,9911 | 1214/22/0 | 1,78 % |
| scavenger | 21 | unité 115 | oui | 0,9887 | 1208/28/0 | 2,27 % |
| predator | 19 | budget 147 | non | 0,9438 à l'unité 145 | 1116/101/19 | 8,17 % |
| predator | 20 | budget 147 | non | 0,8936 à l'unité 145 | 994/221/21 | 17,88 % |
| predator | 21 | budget 147 | non | 0,8847 à l'unité 145 | 977/233/26 | 18,85 % |
| controller | 19 | budget 147 | non | 0,7945 à l'unité 145 | 755/454/27 | 36,73 % |
| controller | 20 | budget 147 | non | 0,8693 à l'unité 145 | 930/289/17 | 23,38 % |
| controller | 21 | budget 147 | non | 0,8750 à l'unité 145 | 947/269/20 | 21,76 % |

Une unité représente 2 048 transitions. Les trois scavengers atteignent donc leur premier
gate stable après respectivement 133 120, 194 560 et 235 520 transitions. Les six autres
candidats consomment chacun 301 056 transitions.

## Validation de la bande de force

La bande autorisée contre le pool RL-S5b est `[0,25 ; 0,50]`, avec une cible souhaitée
de `[0,35 ; 0,45]`. Chaque candidat est évalué sur 1 440 parties en aller-retour.

| Archétype | Seed | Score pool | IC 95 % | V/N/D | Premier | Second | Bande valide |
|---|---:|---:|---:|---:|---:|---:|:---:|
| scavenger | 19 | 0,1663 | [0,1467 ; 0,1859] | 195/89/1156 | 0,1889 | 0,1438 | non |
| scavenger | 20 | 0,2181 | [0,1956 ; 0,2405] | 286/56/1098 | 0,2299 | 0,2062 | non |
| scavenger | 21 | 0,1747 | [0,1571 ; 0,1922] | 182/139/1119 | 0,2042 | 0,1451 | non |
| predator | 19 | 0,2160 | [0,1946 ; 0,2374] | 280/62/1098 | 0,2590 | 0,1729 | non |
| predator | 20 | 0,1753 | [0,1556 ; 0,1951] | 188/129/1123 | 0,1785 | 0,1722 | non |
| predator | 21 | 0,0847 | [0,0714 ; 0,0981] | 82/80/1278 | 0,0875 | 0,0819 | non |
| controller | 19 | 0,1097 | [0,0959 ; 0,1236] | 73/170/1197 | 0,0924 | 0,1271 | non |
| controller | 20 | 0,0562 | [0,0466 ; 0,0659] | 19/124/1297 | 0,0549 | 0,0576 | non |
| controller | 21 | 0,0899 | [0,0767 ; 0,1032] | 55/149/1236 | 0,0757 | 0,1042 | non |

Même les deux meilleurs résultats, `scavenger/20` (0,2181) et `predator/19` (0,2160),
restent sous 0,25. Leurs intervalles de confiance supérieurs restent eux aussi sous la
borne. L'échec de bande est donc net, pas une ambiguïté statistique marginale.

Les écarts premier/second atteignent 8,61 points pour `predator/19` et 5,91 points pour
`scavenger/21`. Les matchs aller-retour restent indispensables. Les compteurs de rôles
d'entraînement sont globalement équilibrés et ne suggèrent pas un défaut grossier de
tirage.

## Analyse par archétype

### Scavenger

L'apprentissage est reproductible sur les trois seeds et produit rapidement une politique
quasi parfaite contre `random-legal` et les trois scripts. Le style attendu est présent :
98,35 % à 98,94 % des victoires sont obtenues aux points. Les parties limitées restent
faibles, entre 0,40 % et 2,27 %.

Cependant, cette compétence ne se transfère pas au pool fort. Le score chute à
0,166–0,218. Les avertissements `repeatable_action_reward_risk` sur les trois seeds sont
cohérents avec une récompense auxiliaire qui favorise de façon répétée les points de
carcasse et l'alimentation sûre. Le candidat apprend très bien le scénario faible qui lui
est présenté, mais pas une politique de niveau intermédiaire robuste.

Conclusion : identité de style réussie, compétence faible réussie, bande de force échouée.

### Predator

Le style KO est extrêmement marqué : le `ko_win_rate` vaut 1,0 aux derniers checkpoints.
Mais aucun seed ne franchit le gate complet. `predator/19` s'en approche à l'unité 125 :
0,915 contre `random-legal`, moyenne 0,75 contre les scripts, mais seulement 0,25 contre
`prudent-v1`, sous le minimum individuel strict de 0,40.

Les avertissements `single_auxiliary_rule_dominates` montrent que le signal de dégâts
domine la récompense auxiliaire. Deux seeds dépassent également 10 % de parties limitées.
La variance inter-seed est forte dans le pool RL-S5b : 0,2160, 0,1753 puis 0,0847.

Conclusion : style réussi mais trop monolithique, compétence non stable et robustesse
inter-seed insuffisante.

### Controller

Les trois seeds échouent le gate et restent très faibles contre le pool RL-S5b
(0,056–0,110). Les taux de limite de manches, de 21,76 % à 36,73 %, indiquent des
politiques fréquemment temporisatrices. Elles gagnent beaucoup contre `random-legal`, mais
leur moyenne contre les trois scripts plafonne à 0,5.

La validation de style comporte en outre une anomalie : `useful_shove_rate` vaut 1,0417
pour le seed 19 et 1,0076 pour le seed 21. Le numérateur compte les shoves classés utiles,
y compris certains impacts sans déplacement, tandis que le dénominateur compte seulement
les shoves ayant déplacé la cible. Cette métrique appelée « rate » peut donc dépasser 1.
Elle ne doit pas servir à valider le style avant correction. De plus, le diagnostic de
style inclut l'éloignement d'une carcasse et l'accès favorable, alors que la récompense
`useful_shove` ne rémunère que mur, boue ou interruption d'alimentation : les deux notions
de « utile » ne sont pas alignées.

Conclusion : échec de compétence, comportement trop passif et validation de style invalide.

## Interprétation générale

Le problème principal n'est pas l'architecture `mlp-compact-v2`. Le scavenger démontre
qu'elle peut apprendre une politique cohérente et reproductible. Le problème est la
combinaison « entraînement depuis zéro uniquement contre `random-legal` + shaping très
spécialisé ». Elle crée un grand écart entre le gate faible et le pool fort : les agents
optimisent leur identité locale sans acquérir la robustesse tactique nécessaire.

Le protocole sépare correctement compétence, style et bande de force, et cette séparation
révèle précisément l'échec : réussir le gate faible ne prédit pas l'entrée dans la bande.
Il ne faut donc ni promouvoir `scavenger/20` par défaut, ni abaisser la borne de 0,25 après
observation des résultats.

## Suite recommandée

1. Corriger d'abord la métrique `controller` afin qu'elle soit bornée dans `[0,1]` et
   aligner exactement la définition diagnostique de `useful_shove` sur celle de la
   récompense.
2. Conserver ces résultats A2 comme baseline immuable et ne pas les écraser.
3. Définir une itération A3 de calibration, toujours sur trois seeds, avant toute
   production cinq-seeds.
4. Garder l'entraînement depuis zéro et `mlp-compact-v2`, mais introduire un curriculum
   explicitement spécifié : `random-legal` pour l'amorçage, puis adversaires scripts ou
   membres gelés du pool. Cette modification exige une nouvelle spécification car elle
   change le protocole d'entraînement.
5. Rééquilibrer les récompenses : plafonner ou réduire les signaux répétables du
   scavenger, diversifier le predator au-delà des seuls dégâts, et renforcer chez le
   controller les conséquences tactiques terminales plutôt que les shoves isolés.
6. Réexécuter les trois validations séparées et conserver la règle d'arrêt au premier gate
   stable. Ne lancer la production que si plusieurs seeds par archétype entrent réellement
   dans la bande `[0,25 ; 0,50]`.

## Verdict opérationnel

- `scavenger` : **prometteur mais non publiable** ; meilleur candidat de travail : seed 20.
- `predator` : **à recalibrer** ; seed 19 est le meilleur point de départ analytique,
  pas un candidat de production.
- `controller` : **à revoir avant réentraînement**, métrique comprise.
- production RL-S5c : **bloquée** jusqu'à une nouvelle calibration concluante.

