Application Streamlit de Simulation ROI (``simulation.py``)
=============================================================

Le fichier ``simulation.py`` est une application **Streamlit autonome** dédiée à la
simulation financière interactive d'investissements hôteliers sur 10 ans (2026-2035).
Contrairement au pipeline principal (``main.py``), cette application est entièrement
orientée utilisateur et ne nécessite aucune connaissance en programmation pour être utilisée.

.. figure:: _static/_static/streamlit_roi_simulator.png
   :align: center
   :alt: Capture d'écran de l'application Streamlit de simulation ROI
   :width: 100%

   Vue générale de l'application Streamlit ``simulation.py`` après exécution d'une simulation comparée
   pour Marrakech avec les 3 modèles SARIMA, Ridge et LSTM.

.. note::
   Pour lancer l'application, exécutez depuis la racine du projet :

   .. code-block:: bash

      streamlit run simulation.py

   L'application sera accessible dans votre navigateur à ``http://localhost:8501``.


Objectif et Positionnement
---------------------------

Ce simulateur permet aux analystes et décideurs hôteliers de :

1. **Choisir une ville** parmi 6 métropoles marocaines stratégiques.
2. **Paramétrer un investissement hôtelier** (nombre de chambres, capex, ADR, occupation, WACC).
3. **Obtenir automatiquement les projections des Top 3 modèles** d'apprentissage automatique
   identifiés lors de l'entraînement (SARIMA, Ridge, LSTM).
4. **Comparer visuellement** les profils de rentabilité cumulée de chaque modèle sur 10 ans.
5. **Recevoir une recommandation d'investissement synthétique** basée sur les indicateurs NPV,
   IRR et Payback.


Architecture Interne
---------------------

L'application repose sur un enchaînement modulaire :

.. code-block:: text

   simulation.py
   ├── get_clean_tourism_data()           ← Chargement et nettoyage des données (mis en cache)
   ├── generate_projections()             ← Prédictions récursives par modèle
   │   ├── SarimaModel                    ← Prévision statistique directe
   │   ├── RidgeModel                     ← Prévision ML récursive
   │   └── LstmModel                      ← Prévision Deep Learning récursive
   ├── HotelROISimulator.simulate_with_forecast()   ← Cash flows selon arrivées prédites
   └── HotelROISimulator.calculate_metrics_for_gop() ← NPV, IRR, Payback, ROI

Chargement et Préparation des Données
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @st.cache_data
   def get_clean_tourism_data():
       df = loader.load_and_merge_tourism_data()
       df = cleaner.integrate_covid_data(df)
       df = cleaner.reconstruct_historical_arrivals(df)
       df = cleaner.reconstruct_historical_receipts(df)
       return df

Le décorateur ``@st.cache_data`` assure que les données ne sont chargées et nettoyées
**qu'une seule fois** par session, même si l'utilisateur modifie les paramètres de la
barre latérale.

Génération des Projections (``generate_projections``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

La fonction ``generate_projections`` joue le rôle de **dispatcher** : elle sélectionne
la bonne stratégie de prévision selon le nom du modèle passé en paramètre :

.. list-table:: Stratégie de prévision par type de modèle
   :header-rows: 1
   :widths: 20 20 60

   * - Modèle
     - Famille
     - Stratégie
   * - ``SARIMA``
     - Statistique
     - Appel direct à ``model.predict(steps=N)`` — le modèle SARIMA prédit nativement
       une séquence future.
   * - ``Ridge``
     - Machine Learning
     - Appel à ``forecast_recursive_ml()`` — prévision récursive : la prédiction du mois
       ``t`` devient le lag de la prédiction du mois ``t+1``.
   * - ``LSTM``
     - Deep Learning
     - Appel à ``forecast_recursive_dl()`` — identique à la récursion ML mais avec les
       données reformatées en séquences 3D ``(batch, window, features)``.

Identification Automatique du Top 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

L'application lit automatiquement le fichier ``data/model_performance_metrics.csv``
généré lors de l'entraînement via ``main.py`` et sélectionne les 3 modèles les mieux
classés selon le critère **R²** (coefficient de détermination) :

.. code-block:: python

   metrics_df = pd.read_csv(metrics_path)
   valid_df = valid_df.sort_values(by='R2', ascending=False)
   top_3_models = valid_df['Mapped_Model'].unique().tolist()[:3]

En l'absence du fichier de métriques, un fallback est activé sur ``['Ridge', 'CatBoost',
'Random Forest']``.


Paramètres Configurables (Barre Latérale)
------------------------------------------

Paramètres Financiers de Base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Paramètre
     - Unité
     - Défaut (Marrakech)
     - Description
   * - **Ville Cible**
     - —
     - Marrakech
     - Choisie parmi 6 villes : Marrakech, Casablanca, Agadir, Tanger, Rabat, Fès.
       Chaque ville charge des paramètres financiers pré-calibrés.
   * - **Investissement (CapEx)**
     - Millions USD
     - 150 M$
     - Coût total de construction ou d'acquisition de l'hôtel.
   * - **ADR Initial**
     - USD / nuit
     - 250 $
     - Tarif journalier moyen de départ (Average Daily Rate).
   * - **Nombre de Chambres**
     - Unité
     - 200
     - Capacité totale de l'hôtel.
   * - **Taux d'Occupation de Base**
     - %
     - 65%
     - Taux d'occupation annuel hors événements exceptionnels.
   * - **WACC (Taux d'Actualisation)**
     - %
     - 8%
     - Coût Moyen Pondéré du Capital pour le calcul de la VAN.
   * - **Marge OpEx**
     - %
     - 65%
     - Part des coûts opérationnels dans le revenu total ; ``GOP = Revenu × (1 − OpEx)``.
   * - **Taux d'Inflation Annuel**
     - %
     - 2,5%
     - Appliqué annuellement à l'ADR pour simuler l'érosion monétaire.

Paramètres Coupe du Monde FIFA 2030
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Paramètre
     - Défaut
     - Description
   * - **Activer les impacts 2030**
     - Activé
     - Applique un sur-boost de +15% sur l'occupation et un boost configurable sur l'ADR
       pour l'année 2030 uniquement.
   * - **Boost ADR 2030**
     - +40%
     - Augmentation de l'ADR en 2030 due à la demande exceptionnelle de la Coupe du Monde.
       Paramètre issu de l'analyse sectorielle comparative (Qatar 2022, Russie 2018).

Paramètres Modélisation & Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Paramètre
     - Défaut
     - Description
   * - **Époques Deep Learning**
     - 10
     - Nombre de passes d'entraînement pour LSTM. Augmenter pour de meilleures prédictions
       au prix d'un temps de calcul plus long.
   * - **Variables d'entrée (Features)**
     - 14 sélectionnées
     - Sous-ensemble des 36 features disponibles du pipeline. Voir la page
       :doc:`feature_engineering` pour la description complète de chaque feature.


Constantes Pré-calibrées par Ville
------------------------------------

L'application embarque des constantes financières pré-calibrées basées sur des études
sectorielles hôtelières au Maroc :

.. list-table:: Paramètres par défaut par ville
   :header-rows: 1
   :widths: 15 10 12 12 51

   * - Ville
     - CapEx (M$)
     - ADR ($)
     - Part Nuitées
     - Recommandation d'Investissement
   * - **Marrakech**
     - 150
     - 250
     - 35%
     - ✅ Investir (Prioritaire, forte demande touristique internationale)
   * - **Casablanca**
     - 180
     - 230
     - 20%
     - ✅ Investir (Tourisme d'affaires premium)
   * - **Agadir**
     - 130
     - 165
     - 18%
     - ⚠️ À étudier (Saisonnier balnéaire, forte variabilité)
   * - **Tanger**
     - 145
     - 155
     - 10%
     - 🕐 Attendre (En développement rapide — Tanger Med)
   * - **Rabat**
     - 165
     - 175
     - 9%
     - 🕐 Attendre (Administratif haut de gamme)
   * - **Fès**
     - 120
     - 135
     - 8%
     - ❌ Éviter (Besoin d'infrastructures touristiques majeures)


Calcul des Cash Flows et Indicateurs
--------------------------------------

Le simulateur utilise la méthode ``simulate_with_forecast()`` de ``HotelROISimulator``
qui couple les prédictions d'arrivées touristiques au taux d'occupation de l'hôtel :

.. math::

   \text{Occ}_t = \min\left(0.95,\ \text{Occ}_{\text{base}} \times \frac{\hat{A}_t}{A_{2025}}\right)

Où :

- :math:`\hat{A}_t` est le volume d'arrivées prédit par le modèle pour l'année :math:`t`
- :math:`A_{2025}` est le volume d'arrivées réel de référence (2025)
- :math:`\text{Occ}_{\text{base}}` est le taux d'occupation paramétré

La formule de revenus annuels est :

.. math::

   \text{Revenu}_t = \text{Chambres} \times \text{Occ}_t \times 365 \times \text{ADR}_t

Où :math:`\text{ADR}_t = \text{ADR}_0 \times (1 + r_{\text{inflation}})^t` est indexé à l'inflation.

Le **Gross Operating Profit (GOP)** est calculé comme :

.. math::

   \text{GOP}_t = \text{Revenu}_t \times (1 - \text{OpEx\_margin})

Les indicateurs financiers clés calculés sont :

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Indicateur
     - Formule simplifiée
     - Interprétation
   * - **NPV** (VAN)
     - :math:`\sum_{t=0}^{10} \frac{CF_t}{(1+r)^t}`
     - Valeur actualisée nette. Une NPV positive indique un projet rentable au-delà du coût du capital.
   * - **IRR** (TRI)
     - :math:`r` tel que :math:`\text{NPV}(r) = 0`
     - Taux de rentabilité interne. Comparer au WACC : IRR > WACC signifie une création de valeur.
   * - **Payback**
     - 1ère année :math:`t` où :math:`\sum CF_t \geq 0`
     - Délai de récupération de l'investissement initial.
   * - **ROI Cumulé**
     - :math:`\frac{\sum \text{GOP}_t - \text{CapEx}}{\text{CapEx}} \times 100`
     - Retour sur investissement brut sur 10 ans, non actualisé.


Recommandation d'Expert Automatique
-------------------------------------

Après la simulation, l'application génère automatiquement une note d'analyse qui
évalue la décision selon le ROI maximal observé parmi les 3 modèles :

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Seuil ROI
     - Décision
     - Signification
   * - ROI ≥ 80%
     - ✅ **FAVORABLE**
     - L'investissement est fortement recommandé.
   * - 40% ≤ ROI < 80%
     - ⚠️ **À ÉTUDIER**
     - Acceptable, mais une analyse de sensibilité complémentaire est conseillée.
   * - ROI < 40%
     - ❌ **DÉFAVORABLE**
     - Le projet ne justifie pas le niveau de risque dans le contexte actuel.
