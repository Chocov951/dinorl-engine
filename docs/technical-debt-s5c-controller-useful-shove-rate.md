# Dette technique — `controller/useful_shove_rate`

Statut : `deferred_post_v1`.

Dans A2, `useful_shove_rate` peut dépasser 1 car le numérateur accepte notamment une
collision murale utile sans déplacement, tandis que le dénominateur compte uniquement les
bousculades ayant déplacé la cible. La définition diagnostique de « utile » diffère aussi
de celle du Reward DSL.

Avant toute réactivation post-V1 de `controller`, choisir un dénominateur cohérent, borner
la métrique dans `[0,1]`, aligner diagnostic et récompense, puis rejouer les tests et la
calibration. Cette dette ne bloque pas A3 : `controller` n'y participe pas.
