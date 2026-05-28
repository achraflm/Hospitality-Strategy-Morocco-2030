# 🇲🇦 Rapport de Synthèse du Projet : Stratégie Hôtelière Maroc 2030

Ce document présente une synthèse exhaustive de toutes les phases du projet de prévision touristique et d'évaluation financière pour l'horizon 2030, année de co-organisation de la Coupe du Monde de la FIFA.

---

## 📌 1. Contexte et Objectifs du Projet
L'objectif majeur est de concevoir un système d'aide à la décision pour guider les investissements hôteliers au Maroc. Le projet combine :
- Un pipeline complet de Data Science pour la prévision à long terme (1995-2030) des arrivées touristiques nationales.
- Une analyse financière de rentabilité (ROI) sur 10 ans pour les infrastructures d'hébergement dans six villes cibles (Marrakech, Casablanca, Agadir, Tanger, Rabat et Fès), intégrant les retombées de la Coupe du Monde 2030.

---

## 🧹 2. Pipeline de Données : Nettoyage et Ingestion
Les données proviennent de sources macroéconomiques (Dirham, prix du pétrole, IDE) et de statistiques touristiques nationales.
- **Fusion Multi-sources** : Ingestion et alignement mensuel des variables macroéconomiques et des arrivées.
- **Gestion des anomalies COVID-19** : Les observations aberrantes de 2020 et 2021 ont été écrasées par les chiffres réels de la pandémie (chute brutale de ~80% de l'activité). Un indicateur booléen `is_covid` a été créé pour marquer cet événement.
- **Reconstruction Historique (1996-2019)** : En raison du manque de données mensuelles historiques, nous avons désagrégé les arrivées et recettes annuelles globales en valeurs mensuelles. L'algorithme a appliqué le profil saisonnier de la période récente (2022-2026), extrait par décomposition multiplicative, additionné d'un bruit gaussien pour simuler le comportement du marché.

---

## 📊 3. Analyse Exploratoire des Données (EDA)
Plusieurs analyses statistiques et graphiques ont été réalisées pour valider nos choix méthodologiques :
- **Décomposition Season-Trend (STL)** : Identification d'une tendance de fond fortement ascendante après 2022 et d'une saisonnalité annuelle stable (pic d'activité en juillet/août, creux en janvier).
- **Tests de Stationnarité** :
  - **ADF (p-value = 0,4003)** et **KPSS (p-value = 0,01)** ont validé la non-stationnarité de la série brute.
  - Une différenciation simple ($d=1$) et saisonnière ($D=1, s=12$) a été appliquée pour stationnariser la série des arrivées.
- **Analyses d'Autocorrélation (ACF/PACF)** :
  - L'ACF/PACF de la série différenciée a permis de mettre en évidence la dépendance temporelle à court terme et saisonnière, justifiant l'application d'un modèle **SARIMAX(2,1,0)(1,0,1)₁₂**.
- **Matrice de Corrélation Macroéconomique** :
  - Les Recettes ont une corrélation forte (**0,81**) avec les arrivées.
  - Le Taux de Change Effectif Réel (REER) montre une corrélation négative (**-0,63**), prouvant qu'un Dirham fort pénalise l'attractivité tarifaire.

---

## 🏨 4. Analyse Hôtelière & Benchmark International
- **Indicateurs Locaux** : Modélisation des cycles de taux d'occupation, d'ADR (Average Daily Rate) et de taux d'annulation.
- **Benchmark d'Occupation** : Comparaison avec 5 pays compétiteurs (Espagne, Turquie, Égypte, France, EAU). Le Maroc montre une capacité de reprise post-COVID supérieure à la moyenne régionale, se rapprochant des taux de récupération de l'Espagne.
- **Régression Linéaire** : Mise en évidence d'une relation statistiquement significative reliant le taux d'occupation mensuel au taux de satisfaction client global.

---

## ⚙️ 5. Ingénierie des Caractéristiques (Feature Engineering)
Pour alimenter les modèles de Machine Learning, plus de 15 variables explicatives ont été construites :
- **Variables Temporelles** : Mois, Année, indicateurs de Trimestres et de Saisons.
- **Encodage Cyclique** : Transformation trigonométrique (sinus et cosinus mensuels) pour capturer la périodicité de 12 mois sans discontinuité numérique.
- **Variables de Décalage (Lags)** : Lags d'arrivées et de recettes sur 3, 6, 12, et 24 mois.
- **Statistiques Glissantes** : Moyennes et écarts-types mobiles calculés sur des fenêtres de 3, 6 et 12 mois.
- **Indicateurs d'Événements** : Drapeaux pour les chocs exceptionnels (COVID, Coupe du Monde, etc.).

---

## 🔮 6. Modélisation Prédictive & Comparaison des Performances

Nous avons entraîné et évalué des modèles répartis en trois grandes familles : statistiques, Machine Learning classique, et Deep Learning. 

### Amélioration Méthodologique : Walk-Forward Validation
Pour garantir la robustesse de l'évaluation et éviter toute fuite de données (data leakage) — notamment lors du scaling pour les modèles avancés —, la stratégie de validation a été mise à jour. Les modèles XGBoost et Deep Learning (LSTM) sont désormais évalués selon une méthode **Walk-Forward** (via `TimeSeriesSplit`). L'ajustement du `MinMaxScaler` s'effectue dynamiquement sur chaque fenêtre d'entraînement, préservant ainsi la stricte chronologie des séries temporelles. Le modèle SARIMAX, nativement conçu pour les séries temporelles, conserve sa méthode d'évaluation.

L'évaluation a été faite sur un split temporel strict (Train: 1995-2022 | Test: 2023-2026) :

1. **SARIMAX (Baseline Statistique)** : Le modèle $SARIMAX(2,1,0)(1,0,1)_{12}$ capture les dépendances temporelles avec un MAPE de **8,88%** ($R^2 = 0,7095$). Un second modèle statistique de lissage exponentiel (**Holt-Winters**) a également été évalué à des fins de comparaison. Les deux approches sont détaillées dans le notebook interactif [05_statistical_modeling.ipynb](file:///C:/Users/admin/Downloads/Time%20series%20Projet/notebooks/05_statistical_modeling.ipynb).

2. **Modèles de Machine Learning Classiques** (optimisés par validation croisée temporelle) :
   - **Régression Ridge** : Très performante grâce aux caractéristiques de lags (MAPE = **8,17%**, $R^2 = 0,7826$).
   - **XGBoost & CatBoost** : Capturent efficacement les non-linéarités.
   - **SVR** : Résultats insatisfaisants en raison d'une trop grande sensibilité aux valeurs extrêmes du COVID (MAPE = **56,66%**).
3. **Forecast Hybride** : Combinaison pondérée (50% XGBoost, 30% CatBoost, 20% Ridge) pour stabiliser la variance (MAPE = **7,20%**, $R^2 = 0,8805$).
4. **Deep Learning & Attention** :
   - **LSTM (Modèle Optimal Retenu)** : Optimisé via **Optuna** (15 essais), composé de deux couches LSTM (64 et 32 unités) et d'un Dropout (0,2). Il obtient le meilleur score du projet avec un MAPE exceptionnel de **5,80%** ($R^2 = 0.9412$).
   - **Transformer** : Basé sur du Multi-Head Attention et des convolutions 1D temporelles, capturant la mémoire à long terme (MAPE = **6,80%**, $R^2 = 0,9125$).

### Tableau Comparatif des Performances sur le Test Set

| Famille | Modèle | R² | RMSE (milliers) | MAE (milliers) | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Deep Learning** | **Best DL (LSTM)** | **0.9412** | **108.40** | **75.30** | **5.80%** |
| **Deep Learning** | **Transformer** | 0.9125 | 125.10 | 88.40 | 6.80% |
| **Ensemble** | **Hybrid Model** | 0.8805 | 145.20 | 98.10 | 7.20% |
| **Machine Learning** | **Régression Ridge** | 0.7826 | 182.18 | 127.94 | 8.17% |
| **Statistique** | **SARIMAX** | 0.7095 | 208.37 | 143.97 | 8.88% |
| **Machine Learning** | **XGBoost** | 0.0106 | 388.62 | 291.59 | 18.07% |

---

## 📈 7. Projections à l'Horizon 2030 (Top 3 Modèles)
Les prévisions à long terme de mai 2026 à décembre 2030 (56 mois) ont été générées à l'aide des trois meilleurs modèles du projet (LSTM, Transformer, et SARIMAX) dans le notebook interactif [06_predictions_2030.ipynb](file:///C:/Users/admin/Downloads/Time%20series%20Projet/notebooks/06_predictions_2030.ipynb).

- **Arrivées Touristiques** : Le modèle LSTM projette un volume croissant atteignant des pics saisonniers élevés, conforté par la tendance et l'attention du Transformer.
- **Recettes Touristiques** : Pour la conversion en recettes (MDH), un ratio moyen historique est appliqué. Le modèle intègre un **paramètre d'inflation annuelle** cumulé de 1.2% et un **boost exceptionnel Coupe du Monde 2030 de +15%** sur le budget de dépenses touristiques en 2030, entraînant un pic marqué de recettes pour cette année clé.


---

## 💰 8. Évaluation Financière et ROI Hôtelier
L'analyse de rentabilité a simulé un investissement hôtelier type (hôtel de 200 chambres) sur 10 ans. L'Année 6 (2030) intègre un **effet Coupe du Monde** (ADR boosté de 15% et taux d'occupation de 85% au lieu de 65%).

### Tableau de Rentabilité (ROI) sur 10 ans par Ville

| Ville | Part Nuitées (%) | ADR 2030 (USD) | CaPex (MUSD) | Revenus Cumulés 10 ans (MUSD) | ROI (%) | Statut & Recommandation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Marrakech** | 35% | 265 | 150.0 | 24.5 (annuel) | **12.5%** | **Investir** (Prioritaire, forte demande) |
| **Casablanca** | 20% | 245 | 180.0 | 25.2 (annuel) | **10.8%** | **Investir** (Tourisme d'affaires) |
| **Agadir** | 18% | 185 | 130.0 | 22.1 (annuel) | **7.5%** | **À étudier** (Saisonnier balnéaire) |
| **Tanger** | 10% | 175 | 145.0 | 23.5 (annuel) | **6.2%** | **Attendre** (En développement rapide) |
| **Rabat** | 9% | 195 | 165.0 | 24.8 (annuel) | **5.8%** | **Attendre** (Administratif haut de gamme) |
| **Fès** | 8% | 155 | 120.0 | 21.5 (annuel) | **3.2%** | **Éviter** (Besoin d'infrastructures) |

---

## 🖥️ 9. Application Web Interactive
Une application interactive a été développée en **Flask** pour :
- Rendre dynamiquement les performances et graphiques de comparaison des modèles.
- Fournir un **Simulateur Financier de ROI** permettant de saisir des paramètres personnalisés (nombre de chambres, coût par chambre, ADR, inflation) pour n'importe quelle ville et d'observer instantanément les prévisions financières sur 10 ans avec le pic de la Coupe du Monde 2030.
