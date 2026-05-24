Ingénierie des Caractéristiques (Feature Engineering)
======================================================

Le pipeline de feature engineering (``src/features.py``) construit **49 variables
prédictives** à partir des données brutes nettoyées. Ces variables couvrent la mémoire
temporelle, la saisonnalité cyclique, les chocs exogènes, les événements calendaires,
la détection non supervisée d'anomalies, et les caractéristiques spécifiques aux
**Nuitées** (``Nights``) — la seconde cible de prédiction.

Toutes les features sont construites par ``build_features(df_clean)``. Deux listes
canoniques sont disponibles :

* ``get_feature_list()`` — 36 features pour prédire les **Arrivées** (``Arrivals``).
* ``get_nights_feature_list()`` — 49 features pour prédire les **Nuitées** (``Nights``).


Tableau Récapitulatif Complet des Features
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Feature
     - Type
     - Description
   * - ``lags_1``
     - Numérique
     - Arrivées touristiques du mois précédent (t−1).
   * - ``lags_2``
     - Numérique
     - Arrivées du mois t−2. Capture la dynamique court terme.
   * - ``lags_6``
     - Numérique
     - Arrivées à t−6 mois. Utile pour les cycles semestriels.
   * - ``lags_12``
     - Numérique
     - Arrivées il y a 12 mois exactement (même mois N−1). **Feature la plus importante**
       pour la saisonnalité annuelle.
   * - ``roll_mean_3``
     - Numérique
     - Moyenne glissante sur 3 mois (calculée sur ``shift(1)`` pour éviter le leakage).
   * - ``roll_mean_6``
     - Numérique
     - Moyenne glissante sur 6 mois.
   * - ``roll_mean_12``
     - Numérique
     - Moyenne glissante sur 12 mois — tendance annuelle lissée.
   * - ``roll_std_3``
     - Numérique
     - Écart-type glissant sur 3 mois — volatilité court terme.
   * - ``roll_std_6``
     - Numérique
     - Écart-type glissant sur 6 mois.
   * - ``roll_std_12``
     - Numérique
     - Écart-type glissant sur 12 mois — volatilité long terme.
   * - ``growth_yoy``
     - Numérique
     - Croissance Year-over-Year en % : :math:`\frac{A_{t-1} - A_{t-13}}{A_{t-13}} \times 100`.
   * - ``month_sin``
     - Numérique
     - Encodage cyclique du mois : :math:`\sin(2\pi \cdot \text{mois}/12)`.
   * - ``month_cos``
     - Numérique
     - Encodage cyclique du mois : :math:`\cos(2\pi \cdot \text{mois}/12)`.
   * - ``quarter``
     - Entier (1–4)
     - Trimestre de l'année.
   * - ``year``
     - Entier
     - Année calendaire (1995–2035). Capture la tendance à long terme.
   * - ``saison_1``
     - Binaire
     - Indicateur Printemps (mars, avril, mai). One-Hot Encoding.
   * - ``saison_2``
     - Binaire
     - Indicateur Été (juin, juillet, août). One-Hot Encoding.
   * - ``saison_3``
     - Binaire
     - Indicateur Automne (septembre, octobre, novembre). One-Hot Encoding.
   * - ``is_summer``
     - Binaire
     - 1 si mois ∈ {6, 7, 8} (juin–août), sinon 0.
   * - ``is_high_season``
     - Binaire
     - 1 si mois ∈ {4, 5, 7, 8, 10, 12} — mois de haute fréquentation.
   * - ``is_vacances``
     - Binaire
     - 1 si mois ∈ {7, 8, 12} — vacances scolaires marocaines.
   * - ``jours_feries_count``
     - Entier (0–3)
     - Nombre de jours fériés nationaux marocains dans le mois (août = 3, mai = 1, etc.).
   * - ``is_ramadan``
     - Binaire
     - 1 si le mois coïncide avec le Ramadan (calendrier lunaire précalculé 1995–2026).
   * - ``is_special_event``
     - Binaire
     - 1 pour : COP22 Marrakech (nov 2016) et Séisme Al Haouz (sep 2023).
   * - ``Oil_price_lag1``
     - Numérique
     - Prix du pétrole à t−1. Proxy du coût des transports aériens.
   * - ``Oil_price_lag3``
     - Numérique
     - Prix du pétrole à t−3 (délai de réaction sur les réservations).
   * - ``FDI_lag1``
     - Numérique
     - Investissements Directs Étrangers à t−1.
   * - ``FDI_lag3``
     - Numérique
     - IDE à t−3.
   * - ``Poverty_rate_lag1``
     - Numérique
     - Taux de pauvreté à t−1 (proxy de la conjoncture socio-économique).
   * - ``Poverty_rate_lag3``
     - Numérique
     - Taux de pauvreté à t−3.
   * - ``REER_lag1``
     - Numérique
     - Taux de Change Effectif Réel (REER) à t−1. Un REER élevé rend le Maroc plus cher.
   * - ``REER_lag3``
     - Numérique
     - REER à t−3.
   * - ``is_covid``
     - Binaire
     - 1 pour la période mars 2020 – décembre 2021 (choc pandémique).
   * - ``cdm_event``
     - Binaire
     - 1 pour déc 2025 – jan 2026 (CAN Maroc 2025) et juin–juil 2030 (Coupe du Monde FIFA).
   * - ``anomaly_zscore``
     - Binaire
     - 1 si la variation mensuelle dépasse ±2,2 écarts-types (détection de chocs brusques).
   * - ``anomaly_iforest``
     - Binaire
     - 1 si Isolation Forest classifie le point comme anomalie (contamination = 8%).
   * - ``anomaly_prophet``
     - Binaire
     - 1 si le résidu de la décomposition tendance + saisonnalité dépasse ±2,5 σ.


Mémoire Temporelle et Lags
---------------------------

Les algorithmes d'apprentissage automatique (ML) n'ayant pas de notion naturelle de
l'ordre temporel des données, nous devons construire explicitement leur mémoire historique :

* **Lags de la Cible (``Arrivals``)** : Retards temporels de 1, 2, 6 et 12 mois.
  Le lag de 12 mois (``lags_12``) est crucial : il fournit directement la valeur du même
  mois de l'année précédente, capturant la saisonnalité annuelle sans aucun codage explicite.
* **Lags Macroéconomiques** : Retards de 1 et 3 mois sur les variables externes
  (``REER_lag1``, ``Oil_price_lag3``, ``FDI_lag1``). Modélise le délai de réaction de
  l'économie réelle sur les flux touristiques.


Statistiques Mobiles (Rolling Features)
-----------------------------------------

Les statistiques mobiles décrivent la dynamique locale à court et moyen terme :

* **Moyennes Mobiles** : Calculées sur des fenêtres de 3, 6 et 12 mois.
* **Volatilité Mobile** : Écart-type glissant sur 3, 6 et 12 mois.
* **Prévention du Leakage** : Toutes les fenêtres mobiles sont calculées sur la série
  décalée de 1 mois (``shift(1)``). La valeur contemporaine :math:`y_t` n'est jamais
  incluse dans le calcul des caractéristiques prédictives à l'instant :math:`t`.


Encodage Cyclique de la Saisonnalité
--------------------------------------

Au lieu de fournir le numéro du mois (1 à 12) sous forme linéaire (ce qui ferait croire
aux modèles que décembre (12) et janvier (1) sont opposés alors qu'ils sont contigus),
nous appliquons une transformation trigonométrique :

.. math::

   \text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{Mois}}{12}\right)

   \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{Mois}}{12}\right)

Cette projection circulaire permet aux modèles linéaires (Ridge) et au SVR de comprendre
la nature cyclique des saisons.


Variables Événementielles et Calendaires
------------------------------------------

Des indicateurs binaires spécifiques ont été créés pour isoler les événements récurrents
et exceptionnels :

* **Haute Saison (``is_high_season``)** : Activée pour les mois d'avril, mai, juillet,
  août, octobre et décembre.
* **Été (``is_summer``)** : Activée de juin à août.
* **Jours Fériés (``jours_feries_count``)** : Nombre de jours fériés nationaux par mois.
* **Ramadan Mouvant** : Indicateur binaire ajusté selon le calendrier lunaire
  (période de baisse modérée de l'activité touristique internationale).
* **Événements Coupe du Monde (``cdm_event``)** : Flag d'impact événementiel activé pour
  la CAN Maroc 2025 et la Coupe du Monde FIFA 2030 au Maroc.
* **Événements Spéciaux (``is_special_event``)** : COP22 Marrakech (2016) et Séisme
  Al Haouz (2023).


Détection Non Supervisée des Anomalies
-----------------------------------------

Pour aider les modèles à gérer les crises (notamment la pandémie COVID-19), le pipeline
calcule trois descripteurs d'anomalies complémentaires :

1. **Z-Score (``anomaly_zscore``)** : Calculé sur les différences premières de la série
   cible. Un mois est marqué anomalie si son Z-Score dépasse ±2,2.

2. **Isolation Forest (``anomaly_iforest``)** : Un modèle non supervisé entraîné sur
   les variables ``Arrivals``, ``Oil_price``, ``FDI``, ``REER``. Contamination = 8%.

3. **Décomposition Prophet-like (``anomaly_prophet``)** : La série est décomposée en
   tendance (moyenne mobile 12 mois) + saisonnalité (moyenne par mois). Les résidus
   dépassant ±2,5 écarts-types sont flaggés.

Ces trois méthodes sont **complémentaires** : le Z-Score capte les chocs ponctuels
brutaux, l'Isolation Forest détecte les anomalies multivariées, et la décomposition
Prophet identifie les déviations prolongées par rapport au profil saisonnier normal.


Mise à l'Échelle et Séquences (Deep Learning LSTM)
----------------------------------------------------

Pour l'entraînement du réseau LSTM, le formatage diffère des modèles ML classiques :

* **Mise à l'Échelle (Scaling)** : Un ``MinMaxScaler`` est appliqué sur toutes les variables
  numériques pour les ramener entre :math:`[0, 1]`. Les paramètres de scaling sont calculés
  **uniquement** sur l'ensemble d'entraînement, puis appliqués à l'ensemble de test pour
  éviter le data leakage.

* **Fenêtrage Temporel (Sequencing)** : Les données sont restructurées en séquences 3D de
  format ``(nb_séquences, window_size, nb_features)`` avec ``window_size = 12`` mois.
  Chaque séquence de 12 mois prédit la valeur du 13ème mois.
