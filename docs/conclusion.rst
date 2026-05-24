Conclusion et Perspectives
===========================

Bilan du Projet
---------------
Ce projet a permis de développer un pipeline complet d'analyse prédictive et financière appliqué au secteur de l'hospitalité au Maroc. Les principales étapes validées incluent :

1. La mise en place d'une **architecture de données unifiée** regroupant indicateurs économiques, historiques hôteliers et benchmarks internationaux.
2. Un algorithme de **reconstruction historique mensuelle** robuste permettant de désagréger les flux annuels manquants de 1996-2019 pour les arrivées et les recettes.
3. Un **feature engineering ciblé et étendu** (jusqu'à 49 variables prédictives) incorporant mémoire temporelle (lags, rolling stats), cyclicalité saisonnière (sin/cos), détection d'anomalies (Z-Score, Isolation Forest, Prophet) et caractéristiques spécifiques aux nuitées.
4. Une **double modélisation prédictive** (Arrivées touristiques et Nuitées nationales) comparant 9 algorithmes de Machine Learning et Deep Learning, validant la performance du modèle Ridge et LSTM pour prédire les pics de la Coupe du Monde FIFA 2030.
5. Une étude de rentabilité de **ROI hôtelier sur 10 ans par ville** identifiant Marrakech (+12,5% ROI) et Casablanca (+10,8%) comme les priorités stratégiques d'investissement.
6. Le développement et le déploiement d'outils interactifs premium : un **simulateur dynamique multi-cible** (Streamlit) facilitant les études de rentabilité par ville, et une **plateforme complète (React + FastAPI)** dotée d'un module d'analyse de risque stochastique par la méthode de Monte Carlo.

Limites de l'Étude
------------------
Plusieurs limites méthodologiques subsistent dans le pipeline actuel :

* **Données de tarification indirectes** : L'indisponibilité publique du RevPAR réel par ville a nécessité d'utiliser des approximations de tarifs (ADR moyens estimés), bien que cette contrainte soit atténuée par la prévision directe des nuitées.
* **Biais de Benchmark de la Coupe du Monde** : L'effet induit par l'événement FIFA 2030 au Maroc a été extrapolé à partir de la Coupe du Monde 2022 au Qatar et 2018 en Russie. Cela peut introduire un biais de projection compte tenu des différences d'infrastructures d'accueil de départ.
* **Incertitude temporelle croissante** : Les prévisions sur l'horizon 2035 intègrent des risques d'incertitude macroéconomique et géopolitique à long terme non modélisables.

Perspectives d'Amélioration
----------------------------
Pour affiner les résultats, plusieurs pistes de développement futur sont envisagées :

1. **Intégration de données en temps réel** : Automatiser le scraping quotidien de tarifs hôteliers sur Booking.com ou TripAdvisor pour disposer de prix réels par ville et par catégorie.
2. **Modèles de prévision désagrégés** : Entraîner des modèles de séries temporelles indépendants pour chaque grande métropole (Marrakech, Casablanca, Agadir) plutôt qu'une série nationale répartie par ratios.
3. **Optimisation stochastique des investissements** : Étendre le moteur Monte Carlo pour optimiser automatiquement la taille de l'hôtel (nombre de chambres) en fonction du coût local de construction et de la distribution de probabilité de la demande.

Références Bibliographiques
---------------------------
.. [1] Ministère du Tourisme, de l'Artisanat et de l'Économie Sociale et Solidaire, *Indicateurs du Tourisme 2025*, 2025.
.. [2] Banque Monde, *Morocco Tourism Statistics Database*, 2024.
.. [3] ONU Tourisme (UNWTO), *World Tourism Barometer & Statistics*, 2025.
.. [4] Haut-Commissariat au Plan (HCP), *Tableau de Bord Conjoncturel National*, 2024.
.. [5] Hyndman, R.J., & Athanasopoulos, G., *Forecasting: Principles and Practice*, 3rd Edition, OTexts, 2021.
.. [6] Chen, T., & Guestrin, C., *XGBoost: A Scalable Tree Boosting System*, KDD, 2016.
.. [7] Hochreiter, S., & Schmidhuber, J., *Long Short-Term Memory*, Neural Computation, 9(8), 1997.
