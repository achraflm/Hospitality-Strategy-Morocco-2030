Applications Streamlit Interactives
=============================================================

Le projet fournit deux applications interactives basées sur **Streamlit** pour permettre aux utilisateurs d'interagir facilement avec le pipeline de données et d'ajuster les modèles prédictifs.

.. contents:: Table des Matières
   :local:
   :depth: 2


1. Dashboard de Modélisation et Prévision (``dashboard.py``)
------------------------------------------------------------------

Le fichier ``dashboard.py`` est une application **Streamlit de modélisation et d'exploration** qui permet aux scientifiques des données et aux analystes d'expérimenter en temps réel avec le pipeline de Machine Learning.

.. note::
   Pour lancer cette application, exécutez depuis la racine du projet :

   .. code-block:: bash

      streamlit run dashboard.py

Objectif et Fonctionnalités
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cette application permet de simuler et configurer les paramètres clés de l'entraînement :

1. **Exploration Interactive des Données (EDA)** : Visualisation des séries historiques et de la table source.
2. **Choix du Split Temporel** : L'utilisateur peut définir l'année de début du test split.
3. **Sélection des Caractéristiques (Feature Engineering)** : Sélection dynamique des variables explicatives parmi les lags, les moyennes mobiles et les variables événementielles (``cdm_event``, ``is_covid``).
4. **Sélection et Entraînement des Modèles** : L'utilisateur choisit les modèles à évaluer (comme SARIMA, Ridge, XGBoost, LSTM).
5. **Projections 2030 (Coupe du Monde)** : Simulation des arrivées ou des nuitées jusqu'en 2030 en appliquant des paramètres macro-économiques (Taux d'inflation, Boost d'attractivité Coupe du Monde).


2. Application de Simulation des Modèles et Projections (``simulation.py``)
---------------------------------------------------------------------------

Le fichier ``simulation.py`` est une application **Streamlit autonome** dédiée à l'évaluation et la simulation à la volée de différents modèles sur un horizon de projection défini par l'utilisateur.

.. note::
   Pour lancer cette application, exécutez depuis la racine du projet :

   .. code-block:: bash

      streamlit run simulation.py

Objectif et Positionnement
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contrairement à ce que son nom pourrait laisser penser, **cette application ne contient pas de module de calcul de ROI hôtelier**. Son but est uniquement de simuler différents modèles d'Intelligence Artificielle et de générer une plage de prédictions dynamiques. 

Ce simulateur permet de :

1. **Choisir la cible de prédiction** : Arrivées touristiques OU Nuitées hôtelières.
2. **Paramétrer l'horizon de projection** : Permet de choisir l'année de fin de prévision (par exemple de 2028 à 2040).
3. **Sélectionner les modèles à comparer** : XGBoost, Ridge, LSTM, SARIMA, Prophet, etc.
4. **Intégration AutoResearch** : Le module d'IA autonome évalue les modèles de Deep Learning en mode **Walk-Forward** et génère des insights textuels sur leur performance de généralisation. Le ML classique est évalué de façon standard.
5. **Visualiser les projections** : Les courbes de chaque algorithme sont tracées simultanément jusqu'à l'année cible, offrant un *range* de prédictions clair pour appréhender l'incertitude et la tendance.

.. warning::
   **Important : Pas de calcul de ROI**.
   L'application ``simulation.py`` est uniquement dédiée aux prévisions de séries temporelles et à l'analyse algorithmique. Les notions d'investissement, CapEx, WACC ou Valeur Actuelle Nette (NPV) n'existent pas dans cette application.
