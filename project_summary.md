# 🇲🇦 Rapport de Synthèse du Projet : Stratégie Hôtelière Maroc 2030

Ce document présente une synthèse exhaustive de toutes les phases du projet de prévision touristique et d'évaluation financière pour l'horizon 2030, année de co-organisation de la Coupe du Monde de la FIFA.

---

## 📌 1. Contexte et Objectifs du Projet
L'objectif majeur est de concevoir un système d'aide à la décision pour guider les investissements hôteliers au Maroc. Le projet combine :
- Un pipeline complet de Data Science pour la prévision à long terme (1995-2030) des arrivées touristiques nationales.
- Une analyse financière de rentabilité (ROI) sur 10 ans pour les infrastructures d'hébergement dans six villes cibles (Marrakech, Casablanca, Agadir, Tanger, Rabat et Fès), intégrant les retombées de la Coupe du Monde 2030.

---

## 🧹 2. Collecte des Données et Reconstruction des Séries Mensuelles

### Introduction
La qualité d’un modèle de prévision dépend directement de la qualité et de la granularité des données disponibles. Dans le cadre de ce projet, l’objectif principal était de construire une base de données temporelle cohérente permettant :
- l’analyse de la demande touristique marocaine,
- la prévision des arrivées touristiques et des nuitées,
- l’estimation du ROI hôtelier avant la Coupe du Monde FIFA 2030,
- et la simulation de scénarios économiques.

Cependant, plusieurs contraintes importantes ont été rencontrées :
- disponibilité limitée des données officielles mensuelles historiques,
- présence de données annuelles uniquement pour certaines périodes,
- valeurs manquantes,
- granularité différente entre les sources,
- absence de certaines variables économiques consolidées.

Pour résoudre ces problèmes, une stratégie de fusion multi-sources et de reconstruction temporelle a été mise en place.

### Sources des Données
Les données ont été collectées à partir de plusieurs sources nationales et internationales.

**Sources Officielles**
| Source | Type de données |
|---|---|
| ONMT | Statistiques touristiques |
| HCP Maroc | Indicateurs économiques |
| World Bank | Inflation, PIB |
| IMF | Variables macroéconomiques |
| TradingEconomics | Séries temporelles économiques |
| UN Tourism (UNWTO) | Flux touristiques internationaux |

**Sources Ouvertes et Datasets**
| Source | Utilisation |
|---|---|
| Kaggle | Données hôtelières et benchmarking |
| Booking Demand Dataset | Patterns de réservation |
| Tourism Forecasting Datasets | Validation des modèles |
| Qatar 2022 / Russie 2018 | Benchmark événementiel |

### Collecte et Préparation des Données

**1. Fusion Multi-Sources**
Les données provenant de différentes plateformes ont été harmonisées afin de construire un dataset unifié.
Les principales opérations effectuées sont :
- harmonisation des formats temporels,
- conversion des dates,
- suppression des doublons,
- standardisation des noms de colonnes,
- alignement des unités économiques.

**2. Nettoyage des Données**
Un pipeline de nettoyage a été appliqué :
- suppression des valeurs manquantes critiques,
- interpolation des valeurs partielles,
- détection des anomalies,
- correction des incohérences temporelles,
- suppression des doublons.

### Problématique des Données Annuelles

**Contexte**
Une partie importante des données historiques touristiques n’était disponible qu’au format annuel.
Exemple :
- 1995 : 2.6 M d'Arrivées
- 1996 : 2.9 M d'Arrivées
- 1997 : 3.1 M d'Arrivées

Or, les modèles de séries temporelles utilisés dans ce projet (SARIMA, LSTM, XGBoost) nécessitent des données mensuelles afin de :
- capturer la saisonnalité,
- apprendre les cycles touristiques,
- détecter les effets événementiels,
- modéliser les pics de fréquentation.

### Reconstruction des Séries Mensuelles

**Objectif**
Transformer les données annuelles historiques en séries mensuelles réalistes.

**Méthodologie Utilisée**
Une approche hybride a été adoptée :
- extraction des patterns saisonniers des données mensuelles récentes,
- extraction du bruit statistique et des variations naturelles,
- redistribution des valeurs annuelles selon ces patterns.

**Extraction de la Saisonnalité**
Les données mensuelles disponibles sur les périodes récentes ont été utilisées pour identifier :
- les cycles saisonniers,
- les pics estivaux,
- les baisses hivernales,
- les comportements mensuels récurrents.

*Exemple de coefficients extraits :*
Janvier (0.72), Juin (1.18), Juillet (1.42), Août (1.51).
Ces coefficients représentent la proportion relative des arrivées mensuelles par rapport à la moyenne annuelle.

**Redistribution des Données Annuelles**
Chaque valeur annuelle a ensuite été répartie sur les 12 mois selon les coefficients saisonniers.
Le principe utilisé est : `Valeur_mois = Valeur_annuelle × Poids_saisonnier`

Cette méthode permet de conserver :
- la somme annuelle originale,
- les comportements saisonniers réalistes.

**Ajout du Bruit Statistique**
Afin d’éviter des séries artificiellement lisses, un bruit statistique contrôlé a été ajouté. L’objectif est de reproduire :
- les fluctuations naturelles,
- les variations mensuelles réelles,
- les comportements irréguliers observés dans les données modernes.
Le bruit a été généré à partir des distributions observées sur les séries mensuelles réelles.

**Avantages de Cette Approche**
Cette stratégie permet :
- d’augmenter la granularité temporelle,
- d’améliorer l’apprentissage des modèles,
- de conserver les tendances historiques,
- de reproduire des comportements saisonniers réalistes,
- de limiter le manque de données mensuelles historiques.

**Limites de la Méthode**
Certaines limites doivent néanmoins être considérées :
- les données reconstruites restent des approximations,
- les comportements historiques exacts ne peuvent être garantis,
- les événements exceptionnels anciens ne sont pas parfaitement reproduits.

Cependant, cette approche reste largement utilisée dans les problématiques de reconstruction temporelle lorsque les données haute fréquence sont indisponibles.

### Gestion des Chocs Exceptionnels (COVID-19)
Une attention particulière a été accordée à la période COVID-19. Les données réelles de fermeture des frontières et d’effondrement du tourisme ont été intégrées manuellement afin de :
- préserver la cohérence historique,
- éviter un lissage artificiel,
- permettre aux modèles d’apprendre les comportements extrêmes.
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

1. **XGBoost (Walk-Forward)** : Capture efficacement les non-linéarités avec un excellent score de R² = **0.8761** et un MAPE de **7.40%**.
2. **LSTM** : Réseau récurrent performant sur les données séquentielles (R² = **0.6958**, MAPE = **11.05%**).
3. **GRU** : Alternative plus légère au LSTM (R² = **0.5995**, MAPE = **13.31%**).
4. **LSTM 2 Layers** : Architecture LSTM profonde (R² = **0.5418**, MAPE = **14.74%**).
5. **LSTM + CNN** : Modèle hybride combinant convolutions temporelles et récurrence (R² = **0.5320**, MAPE = **12.75%**).

### Tableau Comparatif des Performances (Arrivées) sur le Test Set (Walk-Forward)

| Famille | Modèle | R² | RMSE | MAE | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Machine Learning** | **XGBoost** | **0.8761** | **136 108** | **107 028** | **7.40%** |
| **Deep Learning** | **LSTM** | 0.6958 | 213 247 | 167 149 | 11.05% |
| **Deep Learning** | **GRU** | 0.5995 | 244 656 | 198 297 | 13.31% |
| **Deep Learning** | **LSTM 2 Layers** | 0.5418 | 261 705 | 213 001 | 14.74% |
| **Deep Learning** | **LSTM + CNN** | 0.5320 | 264 485 | 192 598 | 12.75% |

### Tableau Comparatif des Performances (Nuitées) sur le Test Set (Walk-Forward)

| Famille | Modèle | R² | RMSE | MAE | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Machine Learning** | **XGBoost** | **0.4870** | **433 902** | **315 853** | **12.75%** |
| **Deep Learning** | **LSTM** | 0.2622 | 520 339 | 395 140 | 15.38% |
| **Deep Learning** | **GRU** | -0.0792 | 629 343 | 517 907 | 21.31% |
| **Deep Learning** | **LSTM + CNN** | -0.1849 | 659 425 | 498 719 | 19.48% |
| **Deep Learning** | **LSTM 2 Layers** | -0.4356 | 725 851 | 601 326 | 24.76% |

---

## 📈 7. Projections à l'Horizon 2030 (Top Modèles)
Les prévisions à long terme de mai 2026 à décembre 2030 (56 mois) ont été générées à l'aide des meilleurs modèles du projet (XGBoost, LSTM, GRU).

- **Arrivées Touristiques** : Le modèle XGBoost projette un volume croissant atteignant des pics saisonniers élevés, conforté par la tendance de fond.
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
Une application interactive Dashboard / Web app a été développée :
- **Streamlit** : Utilisé pour rendre dynamiquement les performances, l'exploration des données et les graphiques de prévision des 5 modèles retenus (XGBoost, LSTM, GRU, LSTM 2 Layers, LSTM + CNN).
- **React (Frontend)** : Fournit un **Simulateur Financier de ROI** (Monte Carlo) permettant de saisir des paramètres personnalisés (nombre de chambres, coût par chambre, ADR, inflation) pour observer instantanément les prévisions financières.
