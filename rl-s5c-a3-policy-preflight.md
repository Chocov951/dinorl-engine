# RL-S5c-A3 — preflight fonctionnel de la policy

Statut local : prêt pour validation automatisée par `s5c a3-preflight`.

Le contrôle utilise `mlp-compact-v2`, les hyperparamètres PPO historiques, la récompense
généraliste de référence, le même environnement, les mêmes observations, actions et
masques que les spécialistes, et part de zéro sur la seed 20 pendant 147 unités.

Les tests locaux couvrent les contrats d'observation existants et les scénarios réduits :
carcasse accessible, trois points, morsure adjacente, interruption de nutrition, repos et
endurance. Ils couvrent aussi joueur actif, PV, endurance, PM, action principale,
nutrition, états des carcasses, premier/second joueur et terminaison. Aucun score
d'apprentissage définitif n'est produit localement.

Le contrôle serveur doit montrer un apprentissage net, franchir le gate historique sans
effondrement stochastique et atteindre un niveau cohérent avec les généralistes RL-S5b.
Sinon la décision est `FAILED_POLICY_CONTROL` et le pilot reste verrouillé.
