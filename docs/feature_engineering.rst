Ingénierie des Caractéristiques (Feature Engineering)
======================================================

Mémoire Temporelle et Lags
---------------------------
Les algorithmes d'apprentissage automatique (ML) n'ayant pas de notion naturelle de l'ordre temporel des données, nous devons construire explicitement leur mémoire historique :

* **Lags de la Cible (`Arrivals`)** : Retards temporels de 1, 2, 3, 6 et 12 mois. Le lag de 12 mois (`Arrivals_lag12`) est crucial car il fournit directement la valeur du même mois de l'année précédente, capturant ainsi la saisonnalité annuelle.
* **Lags Macroéconomiques** : Retards de 1 et 3 mois sur les variables externes (`REER_lag1`, `Oil_price_lag3`, `Total_Receipts_MDH_lag1`). Cela modélise le délai de réaction de l'économie réelle sur les flux touristiques.

Statistiques Mobiles (Rolling Features)
----------------------------------------
Les statistiques mobiles décrivent la dynamique locale à court et moyen terme :

* **Moyennes Mobiles** : Calculées sur des fenêtres glissantes de 3, 6 et 12 mois.
* **Volatilité Mobile** : Écart-type glissant sur 3, 6 et 12 mois.
* **Prévention du Leakage** : Pour éviter toute fuite d'information (*data leakage*), toutes les fenêtres mobiles sont calculées sur la série décalée de 1 mois (`shift(1)`). La valeur contemporaine $y_t$ n'est jamais incluse dans le calcul des caractéristiques prédictives à l'instant $t$.

Encodage Cyclique de la Saisonnalité
------------------------------------
Au lieu de fournir le numéro du mois (1 à 12) sous forme linéaire (ce qui ferait croire aux modèles que décembre (12) et janvier (1) sont opposés alors qu'ils sont contigus), nous appliquons une transformation trigonométrique :

.. math::

   \text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{Mois}}{12}\right)

   \text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{Mois}}{12}\right)

Cette projection circulaire permet aux modèles (notamment les modèles linéaires et SVR) de comprendre la nature cyclique des saisons.

Variables Événementielles et Calendaires
----------------------------------------
Des indicateurs binaires spécifiques ont été créés pour isoler les événements récurrents et exceptionnels :

* **Haute Saison (`is_high_season`)** : Activée pour les mois de juillet et août (pics touristiques).
* **Été (`is_summer`)** : Activée de juin à septembre.
* **Jours Fériés (`jours_feries_count`)** : Nombre de jours fériés nationaux par mois.
* **Ramadan Mouvant** : Indicateur binaire ajusté chaque année selon le calendrier lunaire (période de baisse modérée de l'activité touristique locale).
* **Événements Coupe du Monde (`cdm_event`)** : Flag d'impact événementiel activé pour les pays hôtes (Coupe du Monde Qatar en nov-déc 2022, CAN au Maroc en jan 2025, et Coupe du Monde au Maroc en juin-juillet 2030).

Détection Non Supervisée des Anomalies
--------------------------------------
Pour aider les modèles à gérer les crises (notamment la pandémie de COVID-19), le pipeline calcule trois descripteurs d'anomalies :

1. **Isolation Forest** : Un modèle d'Isolation Forest est entraîné sur les caractéristiques clés pour détecter les observations hors-normes.
2. **Anomalies Prophet** : Utilisation d'un modèle Prophet rapide pour identifier les mois où les résidus réels dépassent les intervalles de prédiction de base.
3. **Z-Score** : Calculé sur les différences premières de la série cible pour capter les variations anormalement brusques.

Mise à l'Échelle et Séquences (Deep Learning)
---------------------------------------------
Pour l'entraînement des réseaux récurrents (LSTM) et des modèles Attention (Transformer), le formatage diffère :

* **Mise à l'Échelle (Scaling)** : Un `MinMaxScaler` est appliqué sur toutes les variables numériques pour les ramener entre $[0, 1]$. Pour éviter toute fuite d'information, les paramètres de mise à l'échelle (minimum, maximum) sont calculés uniquement sur l'ensemble d'entraînement (*Train Set*), puis appliqués sur le test.
* **Fenêtrage Temporel (Sequencing)** : Les données sont restructurées sous forme de séquences à trois dimensions de format : `(nb_séquences, window_size, nb_features)`. Nous avons retenu une fenêtre glissante de 12 mois (`window_size = 12`) pour prédire la valeur du 13ème mois.
