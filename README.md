# ðŸ‡²ðŸ‡¦ Morocco Tourism Strategy 2030 â€” Plateforme IA de PrÃ©vision & Simulateur ROI HÃ´telier

Ce projet prÃ©sente une solution dÃ©cisionnelle et de prÃ©vision Ã  long terme (jusqu'en 2035) appliquÃ©e Ã  l'industrie touristique marocaine dans la perspective de la co-organisation de la **Coupe du Monde de la FIFA 2030**. La plateforme combine des pipelines de Data Science pour prÃ©dire la demande d'arrivÃ©es et les recettes Ã©conomiques, ainsi qu'un simulateur de rentabilitÃ© hÃ´teliÃ¨re stochastique (Monte Carlo).

---

## ðŸ“‹ Table des MatiÃ¨res
1. [FonctionnalitÃ©s ClÃ©s](#-fonctionnalitÃ©s-clÃ©s)
2. [Structure du Projet](#-structure-du-projet)
3. [Architecture de l'Application Web (DÃ©taillÃ©e)](#-architecture-de-lapplication-web-dÃ©taillÃ©e)
4. [DÃ©tails des 3 ModÃ¨les PrÃ©dictifs Optimaux](#-dÃ©tails-des-3-modÃ¨les-prÃ©dictifs-optimaux)
5. [Moteur de Simulation ROI HÃ´telier](#-moteur-de-simulation-roi-hÃ´telier)
6. [Moteur de Simulation Stochastique de Monte Carlo](#-moteur-de-simulation-stochastique-de-monte-carlo)
7. [Installation & DÃ©marrage](#-installation--dÃ©marrage)
8. [Documentation technique (Sphinx)](#-documentation-technique-sphinx)

---

## âœ¨ FonctionnalitÃ©s ClÃ©s

### 1. Ingestion et Nettoyage de DonnÃ©es
* **Fusion automatique** du jeu de donnÃ©es macroÃ©conomique marocain (**Morocco_cleaned.csv**) et des arrivÃ©es touristiques rÃ©elles (**maroc_tourism_2030_all_arrival_sources.csv**).
* **Imputation des donnÃ©es COVID-19** : IntÃ©gration des chiffres rÃ©els mensuels de 2020-2021 de la pandÃ©mie et activation d'un indicateur boolÃ©en **is_covid** pour les modÃ¨les.
* **Reconstruction Historique (1996-2019)** : Application d'un algorithme de **dÃ©sagrÃ©gation temporelle** basÃ© sur la dÃ©composition saisonniÃ¨re (STL) de la pÃ©riode rÃ©cente (2022-2026) avec injection d'un bruit gaussien contrÃ´lÃ©.

### 2. IngÃ©nierie des CaractÃ©ristiques (Feature Engineering)
* Retards temporels (lags 1, 2, 6, 12).
* Statistiques glissantes (moyenne et Ã©cart-type sur 3, 6, et 12 mois).
* Taux de croissance annuel (YoY) et encodage cyclique trigonomÃ©trique du mois.
* Indicateurs d'Ã©vÃ©nements spÃ©ciaux (Coupe du Monde 2030, crises COVID).
* DÃ©tection d'anomalies non supervisÃ©e : Isolation Forest (**anomaly_iforest**), rÃ©sidus Prophet (**anomaly_prophet**) et Z-Score sur diffÃ©rences (**anomaly_zscore**).

### 3. ModÃ©lisation PrÃ©dictive (Top 3 ModÃ¨les)
* SÃ©lection stricte des 3 modÃ¨les optimaux aprÃ¨s Ã©valuation sur split temporel sÃ©parÃ© et stratÃ©gies de Walk-Forward :
  * **XGBoost** : ModÃ¨le basÃ© sur des arbres de dÃ©cision, extrÃªmement performant sur les caractÃ©ristiques structurÃ©es et la gestion des non-linÃ©aritÃ©s.
  * **LSTM (Deep Learning)** : RÃ©seau de neurones rÃ©current capable de mÃ©moriser les dÃ©pendances temporelles longues, idÃ©al pour capturer les tendances macro-Ã©conomiques.
  * **GRU (Deep Learning)** : Alternative optimisÃ©e au LSTM, plus rapide Ã  entraÃ®ner tout en offrant une excellente robustesse face aux variations saisonniÃ¨res.

---

## ðŸ“‚ Structure du Projet

``**text
â”œâ”€â”€ data/                           # Fichiers de donnÃ©es (bruts et finaux)
â”‚   â”œâ”€â”€ merged_tourism_data_final.csv
â”‚   â”œâ”€â”€ model_performance_metrics.csv # [GÃ‰NÃ‰RÃ‰] MÃ©triques comparatives des modÃ¨les
â”‚   â”œâ”€â”€ predictions_2030.csv         # [GÃ‰NÃ‰RÃ‰] PrÃ©visions d'arrivÃ©es 2030
â”‚   â”œâ”€â”€ receipts_predictions_2030.csv# [GÃ‰NÃ‰RÃ‰] PrÃ©visions de recettes 2030
â”‚   â””â”€â”€ separted/                   # DonnÃ©es divisÃ©es en Train/Test
â”œâ”€â”€ docs/                           # Documentation technique Sphinx (.rst)
â”œâ”€â”€ figures/                        # Graphiques et visualisations gÃ©nÃ©rÃ©s
â”œâ”€â”€ notebooks/                      # Notebooks Jupyter de recherche (Phases 1 Ã  7)
â”œâ”€â”€ src/                            # Code source modulaire du pipeline principal
â”‚   â”œâ”€â”€ cleaning.py                 # Nettoyage et reconstruction historique
â”‚   â”œâ”€â”€ config.py                   # Configuration globale
â”‚   â”œâ”€â”€ data_loader.py              # Ingestion des fichiers CSV
â”‚   â”œâ”€â”€ features.py                 # GÃ©nÃ©ration des descripteurs
â”‚   â”œâ”€â”€ metrics.py                  # Calcul des mÃ©triques de validation
â”‚   â”œâ”€â”€ models/                     # ModÃ¨les individuels (sarima.py, ridge.py, lstm.py)
â”‚   â””â”€â”€ visualization.py            # TracÃ©s graphiques (Matplotlib)
â”œâ”€â”€ backend/                        # Serveur API FastAPI
â”‚   â”œâ”€â”€ main.py                     # Point d'entrÃ©e de l'API
â”‚   â”œâ”€â”€ api/                        # Routeurs de l'API (forecast.py, roi.py, monte_carlo.py)
â”‚   â””â”€â”€ src/                        # Code source partagÃ© avec le backend (forecasting, ROI)
â”œâ”€â”€ frontend/                       # Client web React
â”‚   â”œâ”€â”€ src/
â”‚   â”‚   â”œâ”€â”€ pages/                  # Pages (Dashboard, Forecasting, RoiSimulator, MonteCarlo)
â”‚   â”‚   â”œâ”€â”€ App.jsx                 # Routage et mise en page principale
â”‚   â”‚   â””â”€â”€ main.jsx                # Point d'entrÃ©e React
â”‚   â”œâ”€â”€ package.json
â”‚   â””â”€â”€ vite.config.js              # Configuration de Vite avec proxy API
â”œâ”€â”€ main.py                         # Orchestrateur principal du pipeline
â”œâ”€â”€ generate_all_plots.py           # Script utilitaire de gÃ©nÃ©ration de figures
â””â”€â”€ requirements.txt                # DÃ©pendances Python requises
**`**

---

## ðŸ–¥ï¸ Architecture de l'Application Web (DÃ©taillÃ©e)

La plateforme moderne interactive s'appuie sur une architecture dÃ©couplÃ©e **Frontend / Backend** :

**`**mermaid
graph LR
    subgraph Frontend (React SPA)
        UI[Interface Utilisateur React]
        Chart[Visualisation interactive - Recharts]
    end
    subgraph Backend (FastAPI Server)
        API[FastAPI Router]
        Sim[Calculateur ROI / Moteur Monte Carlo]
        Pred[Moteur de prÃ©diction rÃ©cursive]
    end
    UI -->|HTTP POST / GET| API
    API --> Sim
    API --> Pred
    Pred -->|JSON Projections| Chart
    Sim -->|JSON Simulation / Risk Analysis| UI
**`**

### 1. Le Frontend React (SPA)
DÃ©veloppÃ© avec **React 18** et **Vite** pour des performances de build optimales :
* **Composants Dynamiques** : Formulaires de paramÃ©trage Ã©conomiques Ã©quipÃ©s de sliders en temps rÃ©el.
* **Graphiques Interactifs** : Utilisation de la bibliothÃ¨que **Recharts** pour tracer les courbes de projection d'arrivÃ©es et de recettes avec infobulles personnalisÃ©es, ainsi que les histogrammes de probabilitÃ© de Monte Carlo.
* **Aesthetics & Styling** : ThÃ©matique sombre et moderne en *Glassmorphism* (effets de flou d'arriÃ¨re-plan, bordures semi-transparentes, gradients vibrants cyan/teal/emerald et contrastes Ã©levÃ©s de texte) conÃ§ue en Vanilla CSS avec utilitaires Tailwind.

### 2. Le Backend FastAPI (Python)
Un serveur asynchrone ultra-rapide propulsÃ© par **FastAPI** et **Uvicorn** :
* **Validation des DonnÃ©es** : Utilisation de **Pydantic** pour typer et valider strictement les requÃªtes entrantes (par exemple, les paramÃ¨tres Ã©conomiques de la projection).
* **ParallÃ©lisme & Asynchronisme** : Traitement des calculs de simulations stochastiques lourdes de maniÃ¨re efficace.
* **Auto-documentation** : Swagger UI disponible nativement sous **/docs** permettant de tester les points d'accÃ¨s de l'API en direct.

### 3. Proxy de communication en dÃ©veloppement
Toutes les requÃªtes faites du frontend vers **/api/*** sont automatiquement redirigÃ©es vers le serveur FastAPI local (**http://127.0.0.1:8000**) via le proxy configurÃ© dans **vite.config.js**, Ã©liminant ainsi les problÃ¨mes de CORS en phase de dÃ©veloppement.

---

## ðŸ“ˆ StratÃ©gie d'EntraÃ®nement AvancÃ©e (Deep Learning & XGBoost)

Face au volume limitÃ© de donnÃ©es historiques (typiques des sÃ©ries temporelles annuelles ou mensuelles agrÃ©gÃ©es dans le tourisme), nous avons adoptÃ© une mÃ©thodologie rigoureuse pour l'entraÃ®nement des modÃ¨les complexes (**XGBoost, LSTM, GRU, LSTM+CNN**) afin d'Ã©viter le surapprentissage (*overfitting*) et simuler des conditions rÃ©elles de prÃ©vision : la **Validation Walk-Forward**.

### Pourquoi le Walk-Forward Validation ?
Contrairement Ã  un simple split Train/Test (qui rÃ©duit drastiquement les donnÃ©es d'entraÃ®nement), le Walk-Forward permet au modÃ¨le de s'adapter progressivement aux nouvelles donnÃ©es. 
1. **FenÃªtres Glissantes (Sliding Windows)** : Le modÃ¨le est entraÃ®nÃ© sur une fenÃªtre historique, puis prÃ©dit le pas de temps suivant.
2. **Mise Ã  jour continue** : La vraie valeur du pas de temps est ensuite intÃ©grÃ©e dans l'ensemble d'entraÃ®nement pour prÃ©dire le pas suivant.
3. **Robustesse accrue** : Cette technique, bien que trÃ¨s gourmande en calcul, garantit que les rÃ©seaux de neurones profonds (Deep Learning) et XGBoost apprennent les ruptures de tendance rÃ©centes (comme la volatilitÃ© post-COVID) sans "tricher" sur les donnÃ©es futures (Data Leakage).

---

## ðŸ“Š RÃ©sultats des ModÃ¨les (Deep Learning & XGBoost)

Voici les performances obtenues sur l'ensemble de test (post-Covid) avec la stratÃ©gie Walk-Forward :

### Cible : ArrivÃ©es (Arrivals)
* **XGBoost** : $R^2 = 0.532$, MAPE = $11.86\%$
* **LSTM (Standard)** : $R^2 = -0.126$, MAPE = $19.43\%$
* **LSTM (2-Layers)** : $R^2 = -0.126$, MAPE = $19.43\%$
* **GRU** : $R^2 = -0.126$, MAPE = $19.43\%$

### Cible : NuitÃ©es (Nights)
* **XGBoost** : $R^2 = 0.489$, MAPE = $12.10\%$
* **LSTM (Standard)** : $R^2 = 0.352$, MAPE = $14.37\%$
* **LSTM (2-Layers)** : $R^2 = 0.352$, MAPE = $14.37\%$
* **GRU** : $R^2 = 0.352$, MAPE = $14.37\%$

> **Note** : Le manque profond de donnÃ©es historiques a fortement pÃ©nalisÃ© les modÃ¨les de Deep Learning purs (LSTM/GRU) sur la cible "Arrivals" face Ã  des algorithmes de Machine Learning traditionnels. En revanche, sur la cible "Nights", la logique sÃ©quentielle des rÃ©seaux et l'amplification des arbres (XGBoost) ont rÃ©ussi Ã  extraire des tendances valides ($R^2 > 0.35$).

---

## ðŸ† Top 3 des Meilleurs ModÃ¨les par Cible (Toutes mÃ©thodes confondues)

### ðŸ¥‡ Top 3 pour la Cible "ArrivÃ©es" (Arrivals)
1. **RÃ©gression Ridge** : $R^2 = 0.779$ | MAPE = $11.60\%$
   *(Le modÃ¨le linÃ©aire rÃ©gularisÃ© reste le plus robuste face au faible volume de donnÃ©es et au bruit macroÃ©conomique de la pandÃ©mie).*
2. **Decision Tree** : $R^2 = 0.693$ | MAPE = $10.38\%$
3. **Linear Regression** : $R^2 = 0.636$ | MAPE = $15.34\%$

### ðŸ¥‡ Top 3 pour la Cible "NuitÃ©es" (Nights)
1. **XGBoost (Walk-Forward)** : $R^2 = 0.489$ | MAPE = $12.10\%$
   *(Le meilleur compromis non-linÃ©aire sur les NuitÃ©es, gÃ©rant mieux la variance post-COVID grÃ¢ce Ã  l'entraÃ®nement continu).*
2. **LSTM (Walk-Forward)** : $R^2 = 0.352$ | MAPE = $14.37\%$
3. **GRU (Walk-Forward)** : $R^2 = 0.352$ | MAPE = $14.37\%$

---

## ðŸ’° Moteur de Simulation ROI HÃ´telier

Le simulateur financier projette sur une pÃ©riode de 10 ans les flux de trÃ©sorerie (Cash Flows) d'un investissement hÃ´telier type de 200 chambres. Il compare deux scÃ©narios :
* **ScÃ©nario de RÃ©fÃ©rence** : Croissance linÃ©aire normale basÃ©e sur les tendances historiques.
* **ScÃ©nario Coupe du Monde 2030** : IntÃ¨gre un pic exceptionnel lors de l'AnnÃ©e 6 (2030) avec une hausse de 15% Ã  40% du tarif journalier moyen (ADR) et une occupation saturÃ©e Ã  85%.

Les indicateurs calculÃ©s en temps rÃ©el sont :
* **Cash Flow Net Annuel** : $\text{Revenus} - \text{OpEx} - \text{Amortissements}$.
* **Valeur Actuelle Nette (VAN / NPV)** au taux d'actualisation (WACC) choisi :
  $$\text{VAN} = \sum_{t=1}^{10} \frac{\text{Cash Flow}_t}{(1 + \text{WACC})^t} - \text{Investissement Initial}$$
* **Taux de Retour Interne (TRI / IRR)** et **PÃ©riode de RÃ©cupÃ©ration (Payback Period)**.

---

## ðŸŽ² Moteur de Simulation Stochastique de Monte Carlo

Pour prendre en compte l'incertitude Ã©conomique, la plateforme embarque un gÃ©nÃ©rateur stochastique exÃ©cutant jusqu'Ã  1000 tirages alÃ©atoires :

1. **Ã‰chantillonnage des Variables d'EntrÃ©e** :
   * **Inflation et Taux d'Occupation de Base** : Ã‰chantillonnÃ©s suivant une distribution normale $\mathcal{N}(\mu, \sigma)$.
   * **Marge d'OpEx** : ModÃ©lisÃ©e de maniÃ¨re gaussienne pour reflÃ©ter l'instabilitÃ© des coÃ»ts opÃ©rationnels.
   * **Boost Coupe du Monde** : ModÃ©lisÃ© suivant une distribution triangulaire $\text{Triangular}(min, mode, max)$ pour capturer l'asymÃ©trie positive de l'effet FIFA 2030.
2. **Indicateurs de Risque ClÃ©s calculÃ©s** :
   * **ProbabilitÃ© de Perte** : Proportion de tirages oÃ¹ la VAN actualisÃ©e finale est nÃ©gative : $P(\text{VAN} < 0)$.
   * **Value at Risk (VaR 95%)** : Le 5Ã¨me percentile de la VAN. Elle indique la perte minimale ou le gain minimal garanti avec un niveau de confiance de 95% dans le pire des scÃ©narios.
   * **Intervalles de Confiance Ã  90%** : DÃ©termination des percentiles 5% et 95% pour encadrer prÃ©cisÃ©ment le rendement futur.

---

## âš™ï¸ Installation & DÃ©marrage

### PrÃ©requis
* Python 3.10+
* Node.js 18+

### 1. Installation des dÃ©pendances Python (Backend & Pipeline)
Ã€ la racine du projet :
**`**bash
pip install -r requirements.txt
**`**

### 2. DÃ©marrage du Serveur FastAPI (Backend)
**`**bash
python backend/main.py
**`**
Le serveur dÃ©marre sur **http://127.0.0.1:8000**.

### 3. DÃ©marrage du Client React (Frontend)
Ouvrez un nouveau terminal, puis :
**`**bash
cd frontend
npm install
npm run dev
**`**
L'application est accessible sur **http://localhost:5173**.

### 4. ExÃ©cution du Pipeline Principal (CLI)
Pour rÃ©-entraÃ®ner les modÃ¨les et gÃ©nÃ©rer les courbes de prÃ©diction globales :
**`**bash
python main.py
**`**
*(Utilisez l'option **--quick_run** pour un entraÃ®nement accÃ©lÃ©rÃ© des Ã©poques LSTM Ã  des fins de test)*.

---

## ðŸ“š Documentation technique (Sphinx)

La documentation est rÃ©digÃ©e en reStructuredText (.rst) et compilÃ©e avec Sphinx :
**`**bash
cd docs
pip install -r requirements.txt
make html
**`**
Les fichiers HTML gÃ©nÃ©rÃ©s sont consultables dans **docs/_build/html/index.html`.

