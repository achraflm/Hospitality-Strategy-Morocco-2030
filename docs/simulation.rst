Application Streamlit de Simulation ROI (``simulation.py``)
=============================================================

Le fichier ``simulation.py`` est une application **Streamlit autonome** dediee a la
simulation financiere interactive d'investissements hoteliers sur 10 ans (2026-2035).
Contrairement au pipeline principal (``main.py``), cette application est entierement
orientee utilisateur et ne necessite aucune connaissance en programmation pour etre utilisee.

.. note::
   Pour lancer l'application, executez depuis la racine du projet :

   .. code-block:: bash

      streamlit run simulation.py

   L'application sera accessible dans votre navigateur a ``http://localhost:8501``.


Objectif et Positionnement
---------------------------

Ce simulateur permet aux analystes et decideurs hoteliers de :

1. **Choisir la cible de prediction** : Arrivees touristiques (``Arrivals``) OU Nuitees (``Nights``).
2. **Choisir une ville** parmi 6 metropoles marocaines strategiques.
3. **Parametrer un investissement hotelier** (nombre de chambres, capex, ADR, occupation, WACC).
4. **Obtenir automatiquement les projections des Top 3 modeles** par cible, identifies automatiquement
   a partir des fichiers de metriques CSV generes lors de l'entrainement.
5. **Comparer visuellement** les profils de rentabilite cumulee de chaque modele sur 10 ans.
6. **Visualiser le RevPAR annuel** (Revenue Per Available Room) en mode Nuitees.
7. **Recevoir une recommandation d'investissement synthetique** basee sur les indicateurs NPV,
   IRR, Payback et ROI cumule.


Selecteur de Cible de Prediction
----------------------------------

L'application expose un selecteur unique qui change completement le mode de simulation :

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Comportement
   * - **Arrivees (Arrivals)**
     - Lit ``data/model_performance_metrics.csv`` pour identifier le Top 3.
       L'occupation hoteliere est deduite par le ratio de croissance des arrivees par
       rapport a l'annee de reference 2025 :
       ``Occ(t) = min(0.95, Occ_base x Arrivees_predites(t) / Arrivees_2025)``
   * - **Nuitees (Nights)**
     - Lit ``data/model_performance_metrics_nuitees.csv`` (genere par notebook 08).
       L'occupation est calculee directement et de facon plus precise :
       ``Occ(t) = min(0.95, Nuitees_predites(t) / (Chambres x 365))``
       Un graphique **RevPAR annuel** supplementaire est affiche : ``RevPAR = Occ x ADR``.



L'application repose sur un enchaenement modulaire :

.. code-block:: text

   simulation.py
   +-- get_clean_tourism_data()           -- Chargement et nettoyage des donnees (mis en cache)
   +-- generate_projections()             -- Predictions recursives par modele ET par cible
   |   +-- SarimaModel                   -- Prevision statistique directe
   |   +-- RidgeModel / RandomForest...  -- Prevision ML recursive
   |   +-- LstmModel / RnnModel          -- Prevision Deep Learning recursive
   +-- HotelROISimulator
       +-- simulate_with_forecast()       -- Cash flows : arrivees -> ratio croissance -> Occ
       +-- simulate_with_nuitees_forecast() -- Cash flows : nuitees -> Occ directe -> RevPAR
       +-- calculate_metrics_for_gop()   -- NPV, IRR, Payback, ROI


Chargement et Preparation des Donnees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @st.cache_data
   def get_clean_tourism_data():
       df = loader.load_and_merge_tourism_data()
       df = cleaner.integrate_covid_data(df)
       df = cleaner.reconstruct_historical_arrivals(df)
       df = cleaner.reconstruct_historical_receipts(df)
       return df

Le decorateur ``@st.cache_data`` assure que les donnees ne sont chargees et nettoyees
**qu'une seule fois** par session, meme si l'utilisateur modifie les parametres de la
barre laterale.

Identification Automatique du Top 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

L'application lit automatiquement le fichier de metriques correspondant a la cible choisie
et selectionne les 3 modeles les mieux classes selon le critere **R2** :

.. code-block:: python

   # Mode Arrivees
   metrics_path = "data/model_performance_metrics.csv"

   # Mode Nuitees
   metrics_path = "data/model_performance_metrics_nuitees.csv"

   metrics_df = pd.read_csv(metrics_path)
   valid_df = valid_df.sort_values(by='R2', ascending=False)
   top_3_models = valid_df['Mapped_Model'].unique().tolist()[:3]

En l'absence du fichier de metriques Nuitees, un fallback est active sur
``['Ridge', 'Random Forest', 'XGBoost']``.


Paramètres Configurables (Barre Laterale)
------------------------------------------

Selecteur de Cible
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parametre
     - Description
   * - **Variable a predire**
     - Radio : "Arrivees touristiques (Arrivals)" ou "Nuitees (Nights)".
       Ce choix modifie les features disponibles, le fichier de metriques lu,
       la methode de simulation du ROI, et les graphiques affiches.

Parametres Financiers de Base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Parametre
     - Unite
     - Defaut (Marrakech)
     - Description
   * - **Ville Cible**
     - --
     - Marrakech
     - Choisie parmi 6 villes : Marrakech, Casablanca, Agadir, Tanger, Rabat, Fes.
   * - **Investissement (CapEx)**
     - Millions USD
     - 150 M$
     - Cout total de construction ou d'acquisition de l'hotel.
   * - **ADR Initial**
     - USD / nuit
     - 250 $
     - Tarif journalier moyen de depart (Average Daily Rate).
   * - **Nombre de Chambres**
     - Unite
     - 200
     - Capacite totale de l'hotel.
   * - **Taux d'Occupation de Base**
     - %
     - 65%
     - Taux d'occupation annuel hors evenements exceptionnels.
   * - **WACC (Taux d'Actualisation)**
     - %
     - 8%
     - Cout Moyen Pondere du Capital pour le calcul de la VAN.
   * - **Marge OpEx**
     - %
     - 65%
     - Part des couts operationnels dans le revenu total.
   * - **Taux d'Inflation Annuel**
     - %
     - 2.5%
     - Applique annuellement a l'ADR.

Parametres Coupe du Monde FIFA 2030
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parametre
     - Defaut
     - Description
   * - **Activer les impacts 2030**
     - Active
     - Applique un boost de +40% sur l'ADR pour l'annee 2030 uniquement (configurable).
       En mode Arrivees : boost additionnel de +15% sur l'occupation.
       En mode Nuitees : seul le boost ADR est applique (l'occupation est deja modelisee
       par les nuitees predites).
   * - **Boost ADR 2030**
     - +40%
     - Augmentation de l'ADR en 2030. Parametre issu de l'analyse sectorielle
       comparative (Qatar 2022, Russie 2018).

Parametres Modelisation & Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parametre
     - Defaut
     - Description
   * - **Epoques Deep Learning**
     - 10
     - Nombre de passes d'entrainement pour LSTM/RNN.
   * - **Variables d'entree (Features)**
     - 14-16 selectionnees
     - En mode Arrivees : sous-ensemble des 36 features (``get_feature_list()``).
       En mode Nuitees : sous-ensemble des 49 features (``get_nights_feature_list()``),
       incluant les lags Nights et ``nuitees_per_arrival``.


Constantes Pre-calibrees par Ville
------------------------------------

L'application embarque des constantes financieres pre-calibrees basees sur des etudes
sectorielles hotelires au Maroc :

.. list-table:: Parametres par defaut par ville
   :header-rows: 1
   :widths: 15 10 12 12 51

   * - Ville
     - CapEx (M$)
     - ADR ($)
     - Part Nuitees
     - Recommandation d'Investissement
   * - **Marrakech**
     - 150
     - 250
     - 35%
     - Investir (Prioritaire, forte demande touristique internationale)
   * - **Casablanca**
     - 180
     - 230
     - 20%
     - Investir (Tourisme d'affaires premium)
   * - **Agadir**
     - 130
     - 165
     - 18%
     - A etudier (Saisonnier balneaire, forte variabilite)
   * - **Tanger**
     - 145
     - 155
     - 10%
     - Attendre (En developpement rapide -- Tanger Med)
   * - **Rabat**
     - 165
     - 175
     - 9%
     - Attendre (Administratif haut de gamme)
   * - **Fes**
     - 120
     - 135
     - 8%
     - Eviter (Besoin d'infrastructures touristiques majeures)


Calcul des Cash Flows et Indicateurs
--------------------------------------

Mode Arrivees (``simulate_with_forecast``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le taux d'occupation suit la croissance relative des arrivees predites :

  Occ(t) = min(0.95, Occ_base x Arrivees_predites(t) / Arrivees_2025)

Mode Nuitees (``simulate_with_nuitees_forecast``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le taux d'occupation est deduit directement des nuitees predites :

  Occ(t) = min(0.95, Nuitees_predites(t) / (Chambres x 365))

  RevPAR(t) = Occ(t) x ADR(t)

Dans les deux modes, la formule de revenus annuels est :

  Revenu(t) = Chambres x Occ(t) x 365 x ADR(t)

  GOP(t) = Revenu(t) x (1 - OpEx_margin)

Les indicateurs financiers calcules sont :

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Indicateur
     - Formule simplifiee
     - Interpretation
   * - **NPV** (VAN)
     - Sum CF(t) / (1+r)^t
     - Valeur actualisee nette. Une NPV positive indique un projet rentable.
   * - **IRR** (TRI)
     - r tel que NPV(r) = 0
     - Taux de rentabilite interne. Comparer au WACC.
   * - **Payback**
     - 1ere annee t ou Sum CF(t) >= 0
     - Delai de recuperation de l'investissement initial.
   * - **ROI Cumule**
     - (Sum GOP - CapEx) / CapEx x 100
     - Retour sur investissement brut sur 10 ans.
   * - **RevPAR** (mode Nuitees)
     - Occ(t) x ADR(t)
     - Revenue Per Available Room. Indicateur standard de performance hoteliere.


Recommandation d'Expert Automatique
-------------------------------------

Apres la simulation, l'application genere automatiquement une note d'analyse basee
sur le ROI maximal observe parmi les 3 modeles :

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Seuil ROI
     - Decision
     - Signification
   * - ROI >= 80%
     - FAVORABLE
     - L'investissement est fortement recommande.
   * - 40% <= ROI < 80%
     - A ETUDIER
     - Acceptable, mais une analyse de sensibilite complementaire est conseillee.
   * - ROI < 40%
     - DEFAVORABLE
     - Le projet ne justifie pas le niveau de risque dans le contexte actuel.

