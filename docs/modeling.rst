Modélisation Prédictive
=======================

Protocole d'Évaluation Chronologique
------------------------------------
Pour les séries temporelles, un découpage aléatoire est proscrit car il détruirait la structure d'autocorrélation. Nous appliquons un split temporel strict :

* **Train Set** : Données de janvier 1995 à décembre du (test_year - 1).
* **Test Set** : Données de janvier de test_year à avril 2026 servant uniquement à valider la généralisation des modèles.
* **Prédiction** : Période de prévision pure allant de mai 2026 à décembre 2030 (incluant la Coupe du Monde).

Les hyperparamètres des modèles sont ajustés par recherche ou configurés de manière stable dans leurs modules individuels respectifs du dossier `src/models/`.

Métriques d'Évaluation
----------------------
Les modèles sont comparés sur la base de quatre métriques de régression standards :

* **MAPE (Mean Absolute Percentage Error)** : Mesure l'erreur relative moyenne en pourcentage (cible : < 10%).
* **RMSE (Root Mean Squared Error)** : Pénalise lourdement les grandes erreurs de prédiction.
* **MAE (Mean Absolute Error)** : Écart moyen en valeur absolue.
* **$R^2$ (Coefficient de Détermination)** : Indique la proportion de variance expliquée par le modèle.

Modèles Prédictifs Retenus (Top 3)
----------------------------------

Le pipeline a été optimisé pour n'entraîner et n'évaluer que les 3 meilleurs modèles identifiés pour la prévision de la demande touristique au Maroc :

1. **SARIMA (Modèle Statistique)** :
   - Captures des variations saisonnières stables par différenciation saisonnière d'ordre 12. Idéal comme baseline historique stable.

2. **Régression Ridge (Machine Learning)** :
   - Modèle de régression linéaire avec régularisation L2 (pénalité de Ridge) sur les lags temporels. Très robuste et rapide.

3. **LSTM (Deep Learning)** :
   - Réseau de neurones récurrents (RNN) de type Long Short-Term Memory, particulièrement adapté pour capturer les tendances non linéaires à long terme et les événements exceptionnels (Coupe du Monde 2030).

Bilan Comparatif des Performances
----------------------------------
Le tableau comparatif contenant les résultats de l'évaluation de ces 3 modèles est automatiquement enregistré dans le fichier CSV :

.. code-block:: text

   data/model_performance_metrics.csv

Ce fichier répertorie pour chaque modèle le $R^2$, le RMSE, le MAE et le MAPE, triés par ordre décroissant de performance.

