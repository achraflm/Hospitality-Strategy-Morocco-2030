# 🇲🇦 Morocco Tourism Strategy 2030 — Plateforme IA de Prévision & Simulateur ROI Hôtelier

Ce projet présente une solution décisionnelle et de prévision à long terme (jusqu'en 2035) appliquée à l'industrie touristique marocaine dans la perspective de la co-organisation de la **Coupe du Monde de la FIFA 2030**. La plateforme combine des pipelines de Data Science pour prédire la demande d'arrivées et les recettes économiques, ainsi qu'un simulateur de rentabilité hôtelière stochastique (Monte Carlo).

---

## 📋 Table des Matières
1. [Fonctionnalités Clés](#-fonctionnalités-clés)
2. [Structure du Projet](#-structure-du-projet)
3. [Architecture de l'Application Web (Détaillée)](#-architecture-de-lapplication-web-détaillée)
4. [Détails des 3 Modèles Prédictifs Optimaux](#-détails-des-3-modèles-prédictifs-optimaux)
5. [Moteur de Simulation ROI Hôtelier](#-moteur-de-simulation-roi-hôtelier)
6. [Moteur de Simulation Stochastique de Monte Carlo](#-moteur-de-simulation-stochastique-de-monte-carlo)
7. [Installation & Démarrage](#-installation--démarrage)
8. [Documentation technique (Sphinx)](#-documentation-technique-sphinx)

---

## ✨ Fonctionnalités Clés

### 1. Ingestion et Nettoyage de Données
* **Fusion automatique** du jeu de données macroéconomique marocain (`Morocco_cleaned.csv`) et des arrivées touristiques réelles (`maroc_tourism_2030_all_arrival_sources.csv`).
* **Imputation des données COVID-19** : Intégration des chiffres réels mensuels de 2020-2021 de la pandémie et activation d'un indicateur booléen `is_covid` pour les modèles.
* **Reconstruction Historique (1996-2019)** : Application d'un algorithme de **désagrégation temporelle** basé sur la décomposition saisonnière (STL) de la période récente (2022-2026) avec injection d'un bruit gaussien contrôlé.

### 2. Ingénierie des Caractéristiques (Feature Engineering)
* Retards temporels (lags 1, 2, 6, 12).
* Statistiques glissantes (moyenne et écart-type sur 3, 6, et 12 mois).
* Taux de croissance annuel (YoY) et encodage cyclique trigonométrique du mois.
* Indicateurs d'événements spéciaux (Coupe du Monde 2030, crises COVID).
* Détection d'anomalies non supervisée : Isolation Forest (`anomaly_iforest`), résidus Prophet (`anomaly_prophet`) et Z-Score sur différences (`anomaly_zscore`).

### 3. Modélisation Prédictive (Top 3 Modèles)
* Sélection stricte des 3 modèles optimaux après évaluation sur split temporel (Train : 1995-2022 | Test : 2023-2026) :
  * **SARIMA** : Modèle statistique de référence ($R^2 = 0.7095$, MAPE = $8.88\%$).
  * **Régression Ridge** : Approche de Machine Learning linéaire régularisée ($R^2 = 0.7826$, MAPE = $8.17\%$).
  * **LSTM (Deep Learning)** : Réseau récurrent à mémoire long-court terme ($R^2 = 0.9412$, MAPE = $5.80\%$).

---

## 📂 Structure du Projet

```text
├── data/                           # Fichiers de données (bruts et finaux)
│   ├── merged_tourism_data_final.csv
│   ├── model_performance_metrics.csv # [GÉNÉRÉ] Métriques comparatives des modèles
│   ├── predictions_2030.csv         # [GÉNÉRÉ] Prévisions d'arrivées 2030
│   ├── receipts_predictions_2030.csv# [GÉNÉRÉ] Prévisions de recettes 2030
│   └── separted/                   # Données divisées en Train/Test
├── docs/                           # Documentation technique Sphinx (.rst)
├── figures/                        # Graphiques et visualisations générés
├── notebooks/                      # Notebooks Jupyter de recherche (Phases 1 à 7)
├── src/                            # Code source modulaire du pipeline principal
│   ├── cleaning.py                 # Nettoyage et reconstruction historique
│   ├── config.py                   # Configuration globale
│   ├── data_loader.py              # Ingestion des fichiers CSV
│   ├── features.py                 # Génération des descripteurs
│   ├── metrics.py                  # Calcul des métriques de validation
│   ├── models/                     # Modèles individuels (sarima.py, ridge.py, lstm.py)
│   └── visualization.py            # Tracés graphiques (Matplotlib)
├── backend/                        # Serveur API FastAPI
│   ├── main.py                     # Point d'entrée de l'API
│   ├── api/                        # Routeurs de l'API (forecast.py, roi.py, monte_carlo.py)
│   └── src/                        # Code source partagé avec le backend (forecasting, ROI)
├── frontend/                       # Client web React
│   ├── src/
│   │   ├── pages/                  # Pages (Dashboard, Forecasting, RoiSimulator, MonteCarlo)
│   │   ├── App.jsx                 # Routage et mise en page principale
│   │   └── main.jsx                # Point d'entrée React
│   ├── package.json
│   └── vite.config.js              # Configuration de Vite avec proxy API
├── main.py                         # Orchestrateur principal du pipeline
├── generate_all_plots.py           # Script utilitaire de génération de figures
└── requirements.txt                # Dépendances Python requises
```

---

## 🖥️ Architecture de l'Application Web (Détaillée)

La plateforme moderne interactive s'appuie sur une architecture découplée **Frontend / Backend** :

```mermaid
graph LR
    subgraph Frontend (React SPA)
        UI[Interface Utilisateur React]
        Chart[Visualisation interactive - Recharts]
    end
    subgraph Backend (FastAPI Server)
        API[FastAPI Router]
        Sim[Calculateur ROI / Moteur Monte Carlo]
        Pred[Moteur de prédiction récursive]
    end
    UI -->|HTTP POST / GET| API
    API --> Sim
    API --> Pred
    Pred -->|JSON Projections| Chart
    Sim -->|JSON Simulation / Risk Analysis| UI
```

### 1. Le Frontend React (SPA)
Développé avec **React 18** et **Vite** pour des performances de build optimales :
* **Composants Dynamiques** : Formulaires de paramétrage économiques équipés de sliders en temps réel.
* **Graphiques Interactifs** : Utilisation de la bibliothèque **Recharts** pour tracer les courbes de projection d'arrivées et de recettes avec infobulles personnalisées, ainsi que les histogrammes de probabilité de Monte Carlo.
* **Aesthetics & Styling** : Thématique sombre et moderne en *Glassmorphism* (effets de flou d'arrière-plan, bordures semi-transparentes, gradients vibrants cyan/teal/emerald et contrastes élevés de texte) conçue en Vanilla CSS avec utilitaires Tailwind.

### 2. Le Backend FastAPI (Python)
Un serveur asynchrone ultra-rapide propulsé par **FastAPI** et **Uvicorn** :
* **Validation des Données** : Utilisation de **Pydantic** pour typer et valider strictement les requêtes entrantes (par exemple, les paramètres économiques de la projection).
* **Parallélisme & Asynchronisme** : Traitement des calculs de simulations stochastiques lourdes de manière efficace.
* **Auto-documentation** : Swagger UI disponible nativement sous `/docs` permettant de tester les points d'accès de l'API en direct.

### 3. Proxy de communication en développement
Toutes les requêtes faites du frontend vers `/api/*` sont automatiquement redirigées vers le serveur FastAPI local (`http://127.0.0.1:8000`) via le proxy configuré dans `vite.config.js`, éliminant ainsi les problèmes de CORS en phase de développement.

---

## 🤖 Détails des 3 Modèles Prédictifs Optimaux

Chaque modèle réside dans sa propre classe au sein de `src/models/` et implémente une interface standard `.fit()` / `.predict()` :

1. **SARIMA (Modèle de Lissage & Tendance)** :
   * Captures des variations saisonnières stables par différenciation saisonnière d'ordre 12 ($D=1$).
   * Idéal comme modèle de référence historique à court et moyen terme.
2. **Régression Ridge (Machine Learning régularisé)** :
   * Modèle linéaire pénalisé L2 limitant le surapprentissage (overfitting) sur les lags temporels.
   * Extrêmement rapide à entraîner et robuste face au bruit des variables macroéconomiques.
3. **LSTM (Deep Learning & Mémoire Temporelle)** :
   * Réseau de neurones récurrents (RNN) à deux couches LSTM, optimisé via des mécanismes d'attention temporelle pour capturer des dépendances complexes à long terme.
   * Il surmonte les limitations des modèles classiques face à des ruptures de tendance, comme la relance post-COVID ou l'attraction d'événements majeurs.

---

## 💰 Moteur de Simulation ROI Hôtelier

Le simulateur financier projette sur une période de 10 ans les flux de trésorerie (Cash Flows) d'un investissement hôtelier type de 200 chambres. Il compare deux scénarios :
* **Scénario de Référence** : Croissance linéaire normale basée sur les tendances historiques.
* **Scénario Coupe du Monde 2030** : Intègre un pic exceptionnel lors de l'Année 6 (2030) avec une hausse de 15% à 40% du tarif journalier moyen (ADR) et une occupation saturée à 85%.

Les indicateurs calculés en temps réel sont :
* **Cash Flow Net Annuel** : $\text{Revenus} - \text{OpEx} - \text{Amortissements}$.
* **Valeur Actuelle Nette (VAN / NPV)** au taux d'actualisation (WACC) choisi :
  $$\text{VAN} = \sum_{t=1}^{10} \frac{\text{Cash Flow}_t}{(1 + \text{WACC})^t} - \text{Investissement Initial}$$
* **Taux de Retour Interne (TRI / IRR)** et **Période de Récupération (Payback Period)**.

---

## 🎲 Moteur de Simulation Stochastique de Monte Carlo

Pour prendre en compte l'incertitude économique, la plateforme embarque un générateur stochastique exécutant jusqu'à 1000 tirages aléatoires :

1. **Échantillonnage des Variables d'Entrée** :
   * **Inflation et Taux d'Occupation de Base** : Échantillonnés suivant une distribution normale $\mathcal{N}(\mu, \sigma)$.
   * **Marge d'OpEx** : Modélisée de manière gaussienne pour refléter l'instabilité des coûts opérationnels.
   * **Boost Coupe du Monde** : Modélisé suivant une distribution triangulaire $\text{Triangular}(min, mode, max)$ pour capturer l'asymétrie positive de l'effet FIFA 2030.
2. **Indicateurs de Risque Clés calculés** :
   * **Probabilité de Perte** : Proportion de tirages où la VAN actualisée finale est négative : $P(\text{VAN} < 0)$.
   * **Value at Risk (VaR 95%)** : Le 5ème percentile de la VAN. Elle indique la perte minimale ou le gain minimal garanti avec un niveau de confiance de 95% dans le pire des scénarios.
   * **Intervalles de Confiance à 90%** : Détermination des percentiles 5% et 95% pour encadrer précisément le rendement futur.

---

## ⚙️ Installation & Démarrage

### Prérequis
* Python 3.10+
* Node.js 18+

### 1. Installation des dépendances Python (Backend & Pipeline)
À la racine du projet :
```bash
pip install -r requirements.txt
```

### 2. Démarrage du Serveur FastAPI (Backend)
```bash
python backend/main.py
```
Le serveur démarre sur `http://127.0.0.1:8000`.

### 3. Démarrage du Client React (Frontend)
Ouvrez un nouveau terminal, puis :
```bash
cd frontend
npm install
npm run dev
```
L'application est accessible sur `http://localhost:5173`.

### 4. Exécution du Pipeline Principal (CLI)
Pour ré-entraîner les modèles et générer les courbes de prédiction globales :
```bash
python main.py
```
*(Utilisez l'option `--quick_run` pour un entraînement accéléré des époques LSTM à des fins de test)*.

---

## 📚 Documentation technique (Sphinx)

La documentation est rédigée en reStructuredText (.rst) et compilée avec Sphinx :
```bash
cd docs
pip install -r requirements.txt
make html
```
Les fichiers HTML générés sont consultables dans `docs/_build/html/index.html`.
