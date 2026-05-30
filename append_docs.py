import os

appendix = """

Resultats Complets des Modeles (Arrivees et Nuitees)
----------------------------------------------------

Afin d'offrir une vision exhaustive des performances, voici les metriques obtenues pour les differents modeles evalues sur le jeu de test.

**Tableau 1 : Performances sur les Arrivees Touristiques**

=========  ========  ======  ======  =======
Modele     R2        RMSE    MAE     MAPE
=========  ========  ======  ======  =======
XGBoost    0.874     136869  104937   7.23%
Ridge      0.867     140843  110072   7.12%
LSTM       0.813     166827  137769   9.60%
SARIMA    -0.821     521753  493111  35.83%
=========  ========  ======  ======  =======

**Tableau 2 : Performances sur les Nuitees Touristiques**

=========  ========  ======  ======  =======
Modele     R2        RMSE    MAE     MAPE
=========  ========  ======  ======  =======
XGBoost    0.486     433902  315852  12.75%
LSTM       0.262     520338  395139  15.38%
GRU       -0.079     629342  517907  21.31%
=========  ========  ======  ======  =======

**Pourquoi le R2 des nuitees est-il plus bas ?**
Les nuitees touristiques presentent un score R2 globalement plus faible (0.48 pour le meilleur modele) que les arrivees. Cela s'explique par la difficulte a modeliser la *duree de sejour*, qui est tres volatile et depend de facteurs subjectifs (budget, meteo locale, evenements non-programmes) non captures par les donnees macroeconomiques globales.

Comparaison Graphique des Modeles Alternatives
----------------------------------------------

Bien que notre modele principal reste tres performant, d'autres approches comme le **LSTM+CNN** (avec Walk-Forward validation) ou la **Regression Ridge** donnent egalement de tres bonnes predictions.

.. figure:: ../notebooks/results/autoresearch_output/LSTM_plus_CNN_Arrivals_WF_Prediction.png
   :width: 100%
   :align: center
   :alt: Prediction des Arrivees par LSTM+CNN (Walk-Forward)

.. figure:: ../notebooks/results/autoresearch_output/Ridge_Arrivals_Prediction.png
   :width: 100%
   :align: center
   :alt: Prediction des Arrivees par Ridge Regression

Impact de l'Inflation et de la Coupe du Monde 2030
--------------------------------------------------

Dans les scenarios de simulation vers 2030, il est primordial de considerer l'impact de la **Coupe du Monde** durant l'ete 2030. Un choc d'inflation (estime a une moyenne de +6.3%) est integre specifiquement pour les **mois de Juin (6) et Juillet (7) 2030**. Cette hyper-demande mondiale va faire grimper les prix de l'hebergement, affectant mecaniquement le rendement (ROI) tout en attirant des flux massifs.
"""

with open('docs/modeling.rst', 'a', encoding='utf-8') as f:
    f.write(appendix)
