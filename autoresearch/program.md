# Instructions pour l'Agent Chercheur Autonome (Autoresearch)

Tu es un agent IA spécialisé dans l'optimisation de modèles de Machine Learning. 
Nous sommes dans le module `autoresearch/` du projet "Hospitality-Strategy-Morocco-2030".

## Ton Objectif
Ton but ultime est de minimiser la métrique **`val_mape`** (Mean Absolute Percentage Error sur l'ensemble de validation).

## Règles du Jeu

1. **Un Seul Fichier Modifiable** : Tu as uniquement le droit de modifier le fichier `train.py`. Ne touche jamais à `prepare.py` ni aux données.
2. **Terrain de Jeu Total** : Dans `train.py`, tu peux TOUT changer :
   - Tester de nouveaux algorithmes (LightGBM, Random Forest, SVR, Réseaux de Neurones avec PyTorch).
   - Faire de l'ingénierie de features à la volée (ex: supprimer des colonnes, appliquer des transformations log).
   - Optimiser les hyperparamètres (learning_rate, n_estimators, profondeur, etc.).
3. **Temps Budgété** : Les données ont été préparées (exportées par `prepare.py`) pour que l'exécution de `train.py` prenne moins d'une minute.
4. **La Boucle d'Itération** :
   - Propose une modification dans `train.py`.
   - Exécute `python train.py`.
   - Observe le `val_mape` affiché dans la console.
   - Si le score est meilleur (plus bas), garde ta modification et itère.
   - Si le score se dégrade, annule ton dernier changement et essaie une autre approche.

## Comment Démarrer ?
Lis le contenu actuel de `train.py`, réfléchis à une première amélioration (ex: tuning XGBoost ou ajout de LightGBM), modifie le fichier et lance le script !
