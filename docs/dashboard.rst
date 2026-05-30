Application Streamlit de Modélisation (dashboard.py)
======================================================

Le projet fournit une application interactive basée sur **Streamlit** pour permettre aux utilisateurs d'interagir facilement avec le pipeline de données et d'ajuster les modèles prédictifs.

.. contents:: Table des Matières
   :local:
   :depth: 2


Dashboard de Modélisation et Prévision (``dashboard.py``)
---------------------------------------------------------

Le fichier ``dashboard.py`` est une application **Streamlit de modélisation et d'exploration** qui permet aux scientifiques des données et aux analystes d'expérimenter en temps réel avec le pipeline de Machine Learning.

.. note::
   Pour lancer cette application, exécutez depuis la racine du projet :

   .. code-block:: bash

      streamlit run dashboard.py

   L'application sera accessible dans votre navigateur à l'adresse ``http://localhost:8501``.

Objectif et Fonctionnalités
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cette application permet de simuler et configurer les paramètres clés de l'entraînement :

1. **Choix du Split Temporel** : L'utilisateur peut déplacer un slider pour définir l'année de début du split de test. Les données antérieures servent à l'entraînement, tandis que les données postérieures valident la généralisation du modèle.
2. **Méthode d'Évaluation** : Possibilité de tester les modèles via une méthode classique ("Normal") ou par apprentissage progressif ("Walk-Forward").
3. **Sélection des Caractéristiques (Feature Engineering)** : Sélection dynamique des variables explicatives parmi les lags, les moyennes mobiles et les variables événementielles (``cdm_event``, ``is_covid``, etc.).
4. **Sélection et Entraînement des Modèles** : L'utilisateur choisit les modèles à entraîner en temps réel (XGBoost, SARIMA, Ridge, LSTM, LSTM 2 Layers, LSTM + CNN, GRU).
5. **Nombre d'Époques de Deep Learning** : Configuration du nombre d'époques pour les modèles de Deep Learning (LSTM/RNN).
6. **Exploration Interactive des Données (EDA)** : Visualisation des séries historiques et de la cible de prévision (Arrivées ou Nuitées).
7. **Projections 2030** : Outil intégré de simulation de l'avenir projetant les courbes jusqu'en 2030, en incluant l'impact de l'inflation paramétrable et un Boost d'attractivité pour la Coupe du Monde 2030.

.. warning::
   **Important : Pas de calcul de ROI dans ce Dashboard**.
   Cette application est dédiée aux prévisions de séries temporelles (Arrivées ou Nuitées) et à la comparaison algorithmique. Les calculs de rentabilité financière ou NPV/IRR sont exclusifs à la plateforme web React.
