Conclusion et Perspectives
===========================

Bilan du Projet
---------------
Ce projet a permis de développer un pipeline complet d'analyse prédictive et financière appliqué au secteur de l'hospitalité au Maroc. Les principales étapes validées incluent :

1. La mise en place d'une **architecture de données unifiée** regroupant indicateurs économiques, historiques hôteliers et benchmarks internationaux.
2. Un algorithme de **reconstruction historique mensuelle** robuste permettant de désagréger les flux annuels manquants de 1996-2019.
3. Un **feature engineering ciblé** pour les modèles de machine learning et de deep learning, incorporant mémoire temporelle (lags, rolling stats), cyclicalité saisonnière (sin/cos) et flags d'événements.
4. Une **modélisation comparative multi-familles** (SARIMAX, ML classiques, Deep Learning, Attention) validant la suprématie de l'architecture **LSTM** (MAPE de test = 5,8%) pour prédire le pic touristique de 2030.
5. Une étude de rentabilité de **ROI hôtelier sur 10 ans par ville** identifiant Marrakech (+12,5% ROI) et Casablanca (+10,8%) comme les priorités stratégiques d'investissement.

Limites de l'Étude
------------------
Plusieurs limites méthodologiques subsistent dans le pipeline actuel :

* **Manque de données directes de tarification** : L'indisponibilité publique du RevPAR réel par ville a nécessité d'utiliser des approximations de tarifs (ADR moyens estimés).
* **Biais de Benchmark de la Coupe du Monde** : L'effet induit par l'événement FIFA 2030 au Maroc a été extrapolé à partir de la Coupe du Monde 2022 au Qatar et 2018 en Russie. Cela peut introduire un biais de projection compte tenu des différences d'infrastructure de départ.
* **Incertitude temporelle croissante** : Les prévisions à 5 ans intègrent des risques d'incertitude croissants non modélisés.

Perspectives d'Amélioration
----------------------------
Pour affiner les résultats, plusieurs pistes de développement sont envisagées :

1. **Intégration de données en temps réel** : Automatiser le scraping quotidien de tarifs hôteliers sur Booking.com ou TripAdvisor pour disposer de prix réels par ville.
2. **Modèles de prévision désagrégés** : Entraîner des modèles de séries temporelles indépendants pour chaque grande métropole plutôt qu'une série nationale partagée.
3. **Mise en place d'un Dashboard interactif** : Développer une interface Web (Streamlit/Plotly) interactive pour permettre aux investisseurs de modifier les hypothèses financières en temps réel et de visualiser l'impact immédiat sur le ROI.

Références Bibliographiques
---------------------------
.. [1] Ministère du Tourisme, de l'Artisanat et de l'Économie Sociale et Solidaire, *Indicateurs du Tourisme 2025*, 2025.
.. [2] Banque Mondiale, *Morocco Tourism Statistics Database*, 2024.
.. [3] ONU Tourisme (UNWTO), *World Tourism Barometer & Statistics*, 2025.
.. [4] Haut-Commissariat au Plan (HCP), *Tableau de Bord Conjoncturel National*, 2024.
.. [5] Hyndman, R.J., & Athanasopoulos, G., *Forecasting: Principles and Practice*, 3rd Edition, OTexts, 2021.
.. [6] Chen, T., & Guestrin, C., *XGBoost: A Scalable Tree Boosting System*, KDD, 2016.
.. [7] Hochreiter, S., & Schmidhuber, J., *Long Short-Term Memory*, Neural Computation, 9(8), 1997.
