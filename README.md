# 🇲🇦 Morocco Tourism & Hospitality Strategy 2030 — Pipeline de Prévision & Simulateur ROI

Ce projet présente un pipeline complet de Data Science et une application web interactive pour la modélisation, la prévision des arrivées touristiques au Maroc à l'horizon 2030 (année de co-organisation de la Coupe du Monde de la FIFA) et l'analyse de rentabilité (ROI) des investissements hôteliers par ville.

---

## 📋 Table des Matières
1. [Fonctionnalités Clés](#-fonctionnalités-clés)
2. [Structure du Projet](#-structure-du-projet)
3. [Installation & Dépendances](#-installation--dépendances)
4. [Utilisation & Pipelines](#-utilisation--pipelines)
5. [Détails des Modèles de Prévision](#-détails-des-modèles-de-prévision)
6. [Interface Web & Simulateur ROI](#-interface-web--simulateur-roi)
7. [Documentation](#-documentation)

---

## ✨ Fonctionnalités Clés

Le projet est divisé en plusieurs phases clés, de l'ingestion des données jusqu'au déploiement de l'application :

### 1. Ingestion et Nettoyage de Données Multi-sources
- **Fusion automatique** de jeux de données macroéconomiques marocains (`Morocco_cleaned.csv`) et des arrivées touristiques réelles.
- **Gestion des anomalies COVID-19** : Écrasement des valeurs aberrantes des années 2020-2021 par les chiffres officiels de la pandémie et activation d'un indicateur booléen `is_covid` pour les modèles.

### 2. Reconstruction Historique (1996-2019)
- En l'absence de données mensuelles détaillées pour les arrivées et les recettes avant 2020, application d'un algorithme de **désagrégation temporelle** basé sur le profil saisonnier de la période récente (2022-2026) avec injection d'un bruit gaussien contrôlé.

### 3. Analyse Exploratoire (EDA) & Analyse Hôtelière
- **Décomposition STL (saison-tendance)** additive de la série temporelle des arrivées.
- **Matrice de corrélation** entre variables macroéconomiques (Taux de Change Effectif Réel - REER, prix du pétrole, IDE, taux de pauvreté) et indicateurs touristiques.
- **Tests de stationnarité** robustes : Dickey-Fuller Augmenté (ADF) et KPSS pour justifier l'application de différenciations.
- **Analyse d'autocorrélation** : Tracé des fonctions d'autocorrélation (ACF) et d'autocorrélation partielle (PACF) pour calibrer le modèle statistique SARIMAX.
- **Indicateurs hôteliers locaux** : Profil saisonnier, ADR (Average Daily Rate) moyen et taux d'annulation mensuels.
- **Benchmark international** : Comparaison du taux d'occupation et des ratios de récupération post-COVID avec des destinations concurrentes (Égypte, Turquie, Espagne, France, Grèce, Émirats Arabes Unis).

### 4. Ingénierie des Caractéristiques (Feature Engineering)
- Caractéristiques temporelles et saisonnières (mois, année, trimestres).
- Encodage cyclique trigonométrique (sinus/cosinus) pour capter la périodicité annuelle.
- Lags temporels et statistiques glissantes (moyenne et écart-type sur 3, 6, et 12 mois) sur les arrivées et recettes.
- Caractéristiques d'événements spéciaux (Coupe du Monde 2030, périodes de crises).

### 5. Modélisation Prédictive Multi-familles
- **Baseline Statistique** : Modèle auto-régressif intégré moyenne mobile saisonnier (SARIMAX) ajusté selon les propriétés de la série.
- **Machine Learning Classique** : Entraînement avec recherche sur grille (`GridSearchCV`) et validation croisée sur séries temporelles (`TimeSeriesSplit`) pour :
  - Régression Ridge
  - Support Vector Regression (SVR)
  - XGBoost Regressor
  - LightGBM Regressor
  - CatBoost Regressor
- **Forecast Hybride** : Modèle d'ensemble combinant de manière pondérée (50% XGBoost, 30% CatBoost, 20% Ridge) les forces des différents modèles.
- **Deep Learning Séquentiel** :
  - Optimisation bayésienne d'architecture avec **Optuna** comparant SimpleRNN, GRU, LSTM, Stacked LSTM et CNN-LSTM.
  - Modèle **Transformer** (mécanisme de Multi-Head Attention couplé à des convolutions 1D temporelles).

### 6. Simulation Financière & ROI Hôtelier (Horizon 2030)
- Évaluation financière d'un investissement hôtelier de 10 ans par ville (Marrakech, Casablanca, Agadir, Tanger, Rabat, Fès) avec prise en compte de l'impact exceptionnel de la Coupe du Monde de la FIFA 2030 (boost de 15% de l'ADR et hausse du taux d'occupation).

---

## 📂 Structure du Projet

```text
├── data/                           # Fichiers de données (bruts et finaux)
│   ├── merged_tourism_data_final.csv
│   └── separted/                   # Données divisées en ensembles Train/Test
├── docs/                           # Documentation technique Sphinx (.rst)
│   ├── eda.rst
│   └── modeling.rst
├── figures/                        # Graphiques et visualisations générés par le pipeline
├── notebooks/                      # Notebooks Jupyter interactifs (Phases 1 à 6)
│   ├── 01_data_processing_and_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_machine_learning.ipynb
│   ├── 04_deep_learning.ipynb
│   ├── 05_statistical_modeling.ipynb
│   └── 06_predictions_2030.ipynb


├── src/                            # Code source modulaire
│   ├── cleaning.py                 # Nettoyage et reconstruction historique
│   ├── config.py                   # Configuration globale des chemins et paramètres
│   ├── data_loader.py              # Ingestion des fichiers CSV
│   ├── evaluation.py               # Calcul et affichage des métriques (RMSE, R², MAE, MAPE)
│   ├── features.py                 # Génération des descripteurs
│   ├── models_dl.py                # Définitions des architectures RNN/LSTM/Transformer
│   ├── models_ml.py                # Entraînement des modèles ML et SARIMAX
│   └── visualization.py            # Tracés graphiques (Matplotlib / Seaborn)
├── templates/                      # Fichiers HTML pour l'application Flask
├── app.py                          # Serveur API Flask et simulateur ROI
├── main.py                         # Orchestrateur principal exécutant tout le pipeline
├── generate_all_plots.py           # Script utilitaire de génération de toutes les figures
├── requirements.txt                # Dépendances Python requises
└── project_summary.md              # [GÉNÉRÉ] Synthèse globale de toutes les phases
```

---

## ⚙️ Installation & Dépendances

1. Clonez ou téléchargez le répertoire du projet.
2. Installez les packages Python requis en exécutant :
   ```bash
   pip install -r requirements.txt
   ```

*Note : Les dépendances incluent notamment `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `xgboost`, `lightgbm`, `catboost`, `tensorflow`, `optuna`, `matplotlib`, `seaborn` et `flask`.*

---

## 🚀 Utilisation & Pipelines

### Lancer le pipeline complet
Pour exécuter toutes les étapes (nettoyage, EDA, extraction de caractéristiques, entraînement ML + Deep Learning, évaluation et projections ROI), lancez :
```bash
python main.py
```

### Lancer en mode de débogage rapide
Pour tester le code rapidement sans attendre les 15 époques de Deep Learning ou l'optimisation complète d'Optuna :
```bash
python main.py --quick_run
```

### Générer tous les graphiques
Pour recréer l'ensemble des figures du rapport et des notebooks dans le dossier `figures/` :
```bash
python generate_all_plots.py
```

### Lancer l'application Web & Simulateur ROI
Pour démarrer le serveur local Flask :
```bash
python app.py
```
Ouvrez ensuite votre navigateur à l'adresse [http://localhost:5000](http://localhost:5000).

---

## 📈 Détails des Modèles de Prévision

Le tableau suivant montre les performances typiques des modèles obtenues sur l'ensemble de test (de janvier 2023 à avril 2026) :

| Modèle | R² (Variance Expliquée) | RMSE | MAE | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Best DL (LSTM)** | **0.9412** | **108.40** | **75.30** | **5.80%** |
| **Transformer** | 0.9125 | 125.10 | 88.40 | 6.80% |
| **Hybrid Model** | 0.8805 | 145.20 | 98.10 | 7.20% |
| **Régression Ridge** | 0.7826 | 182.18 | 127.94 | 8.17% |
| **SARIMAX** | 0.7095 | 208.37 | 143.97 | 8.88% |
| **XGBoost** | 0.0106 | 388.62 | 291.59 | 18.07% |

---

## 🖥️ Interface Web & Simulateur ROI

L'application Web développée avec Flask offre un tableau de bord interactif pour :
1. **Consulter les performances** de tous les modèles prédictifs du projet.
2. **Simuler la rentabilité financière** d'un hôtel en modifiant des paramètres clés tels que le taux d'inflation, le coût par chambre, le taux d'occupation de base, l'ADR initial et la ville d'investissement. L'impact de la Coupe du Monde de la FIFA 2030 y est automatiquement injecté dans l'année de simulation correspondante.

---

## 📖 Documentation

Le projet contient une documentation technique rédigée en reStructuredText (.rst) compilable avec Sphinx. Les fichiers principaux sont situés dans le dossier `docs/` :
- `docs/data_pipeline.rst` : Décrit le nettoyage et la reconstruction des données.
- `docs/eda.rst` : Présente les résultats de l'EDA et de la stationnarité (avec l'analyse ACF/PACF).
- `docs/modeling.rst` : Explique les architectures prédictives (SARIMAX, ML, LSTM, Transformer).
- `docs/results_roi.rst` : Présente l'analyse de rentabilité pour l'horizon 2030.
