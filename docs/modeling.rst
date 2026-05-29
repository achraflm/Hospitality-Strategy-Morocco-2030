Modelisation Predictive
=======================

Protocole d'Evaluation Chronologique
--------------------------------------
Pour les series temporelles, un decoupage aleatoire est proscrit car il detruirait la structure d'autocorrelation. Nous appliquons un split temporel strict :

* **Train Set** : Donnees de janvier 1995 a decembre 2022.
* **Test Set** : Donnees de janvier 2023 a avril 2026 servant uniquement a valider la generalisation des modeles.
* **Prediction** : Periode de prevision pure allant de mai 2026 a decembre 2035 (incluant la Coupe du Monde).

**Amélioration Méthodologique : Walk-Forward Training**
Pour les modèles complexes (XGBoost, LSTM), ce découpage statique a été remplacé par une approche **Walk-Forward Validation**. Le modèle s'entraîne sur une fenêtre temporelle, prédit le pas suivant, puis l'intègre pour se ré-entraîner. Cette méthode élimine totalement la fuite de données (data leakage), particulièrement lors du scaling dynamique, et garantit une évaluation robuste, au plus proche des conditions réelles de production.

Les hyperparametres des modeles sont ajustes par recherche ou configures de maniere stable dans leurs modules individuels respectifs du dossier ``src/models/``.

Double Cible de Prediction
----------------------------

Le pipeline predit **deux variables** independamment :

1. **Arrivees touristiques** (``Arrivals``) — entrees de touristes en nombre de voyageurs.
   Features : ``get_feature_list()`` — 36 variables.
   Modeles : entraines dans ``notebooks/03_machine_learning.ipynb``.
   Metriques : ``data/model_performance_metrics_ML.csv``.

2. **Nuitees** (``Nights``) — nombre de nuits passees par les touristes.
   Features : ``get_nights_feature_list()`` — 49 variables incluant les lags Nights et ``nuitees_per_arrival``.
   Modeles : entraines dans ``notebooks/08_nuitees_prediction.ipynb``.
   Metriques : ``data/model_performance_metrics_nuitees.csv``.

Le lien entre les deux cibles est la **Duree Moyenne de Sejour** :

  Nuitees = Arrivees x Duree Moyenne de Sejour

La prediction des Nuitees permet un calcul direct et plus precis du taux d'occupation hotelier :

  Occ(t) = min(0.95, Nuitees_predites(t) / (Chambres x 365))

  RevPAR(t) = Occ(t) x ADR(t)

Metriques d'Evaluation
-----------------------
Les modeles sont compares sur la base de quatre metriques de regression standards :

* **MAPE (Mean Absolute Percentage Error)** : Mesure l'erreur relative moyenne en pourcentage (cible : < 10%).
* **RMSE (Root Mean Squared Error)** : Penalise lourdement les grandes erreurs de prediction.
* **MAE (Mean Absolute Error)** : Ecart moyen en valeur absolue.
* **R2 (Coefficient de Determination)** : Indique la proportion de variance expliquee par le modele.

Évaluation des Modèles (Deep Learning & XGBoost)
-------------------------------------------------

Les algorithmes classiques et avancés ont été entraînés et évalués sur la période post-COVID. 

**Résultats pour la cible "Arrivals" :**

1. **Ridge** (R2 = 0.779, MAPE = 11.6%) — Meilleur modèle linéaire.
2. **Decision Tree** (R2 = 0.693, MAPE = 10.3%)
3. **XGBoost (Walk-Forward)** (R2 = 0.532, MAPE = 11.8%)
4. **LSTM / GRU** (R2 = -0.126, MAPE = 19.4%)

**Résultats pour la cible "Nights" :**

1. **XGBoost (Walk-Forward)** (R2 = 0.489, MAPE = 12.1%)
2. **LSTM / LSTM 2-Layers / GRU** (R2 = 0.352, MAPE = 14.3%)

**Pourquoi le Deep Learning (LSTM, GRU) échoue-t-il sur ces données ?**
Malgré la mise en place d'un entraînement *Walk-Forward* rigoureux pour simuler l'adaptation continue aux chocs, les réseaux récurrents purs comme le LSTM ou le GRU ne parviennent pas à offrir d'excellentes performances. La raison principale réside dans le **manque drastique de volume de données historiques**. 
Les réseaux de neurones profonds nécessitent des dizaines de milliers d'observations pour extraire des *patterns* temporels. Ici (séries annuelles/mensuelles agrégées), le bruit massif lié à la rupture structurelle du COVID-19 écrase le signal. Les modèles plus simples avec forte régularisation (comme **Ridge**) ou les méthodes d'ensembles par arbres (comme **XGBoost**) se montrent beaucoup plus résilients face à la rareté de la donnée.

Top 3 Modèles Globaux
-------------------------

Les 3 meilleurs modèles finaux retenus pour les simulations sont :

1. **Régression Ridge (Arrivées)** : R2 = 0.779
2. **Decision Tree (Arrivées)** : R2 = 0.693
3. **XGBoost Walk-Forward (Nuitées)** : R2 = 0.489

Bilan Comparatif des Performances
------------------------------------
Le tableau comparatif contenant les résultats complets est
automatiquement enregistré dans les fichiers CSV :

.. code-block:: text

   notebooks/results/ml_results.csv (Modèles ML Classiques)
   notebooks/results/dl_wf_results.csv (Modèles Walk-Forward : DL et XGBoost)

Ces fichiers répertorient pour chaque modèle le R2, le RMSE, le MAE et le MAPE.


Courbes des Prévisions vs Données Réelles (Ensemble de Test)
--------------------------------------------------------------

Afin de valider la capacité de généralisation de nos modèles sur des données non vues lors de l'entraînement, nous comparons les prévisions des 3 meilleurs modèles par rapport aux valeurs réelles sur l'ensemble de test (janvier 2023 - avril 2026).

Prévision des Arrivées Touristiques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/05_arrivals_test_comparison_top3.png
   :align: center
   :alt: Courbe de prévision des arrivées (Test Set)
   :width: 100%

   Comparaison des prévisions des Top 3 modèles (Ridge, XGBoost, LSTM) vs Arrivées réelles sur l'ensemble de test (2023-2026). Le modèle Ridge capture fidèlement le profil saisonnier, tandis que le XGBoost en Walk-Forward assure l'extrapolation.

Prévision des Nuitées Hôtelières
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/09_nights_test_comparison_top3.png
   :align: center
   :alt: Courbe de prévision des nuitées (Test Set)
   :width: 100%

   Comparaison des prévisions des Top 3 modèles vs Nuitées réelles sur l'ensemble de test (2023-2026).


