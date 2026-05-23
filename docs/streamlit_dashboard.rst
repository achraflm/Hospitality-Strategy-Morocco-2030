Plateforme Web Interactive Premium (React + FastAPI)
=====================================================

Le projet fournit une application web complète, moderne et premium ("Morocco Tourism Investment Intelligence Platform") divisée en un frontend React monopage (SPA) propulsé par Vite et TailwindCSS, et un backend API en Python avec FastAPI.

Architecture de l'Application :
--------------------------------
L'application est structurée en deux sous-dossiers à la racine :
* **backend/** : Contient le code serveur FastAPI (`backend/main.py`), les routeurs d'API (`backend/api/`) et le moteur de calcul scientifique et de simulation stochastique (`backend/src/`).
* **frontend/** : Contient l'application React monopage configurée avec TailwindCSS et Recharts pour les graphiques interactifs premium.

Lancement des Serveurs en Local :
---------------------------------

1. Démarrer le Serveur API (Backend Python) :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Le serveur FastAPI tourne par défaut sur `http://127.0.0.1:8000`.
.. code-block:: bash

   python backend/main.py

La documentation interactive Swagger (OpenAPI) est accessible à l'adresse `http://localhost:8000/docs`.

2. Démarrer le Serveur de Développement Frontend (React + Vite) :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Depuis la racine du projet :
.. code-block:: bash

   cd frontend
   npm run dev

L'application web est accessible dans le navigateur à l'adresse `http://localhost:5173`. Toutes les requêtes vers `/api/*` sont automatiquement redirigées vers le backend local via un reverse-proxy configuré dans `vite.config.js`.

Structure des Pages de la Plateforme :
--------------------------------------

* **Tableau de Bord (Dashboard)** :
  - Métriques de synthèse des flux touristiques et financiers par ville hôte.
  - Graphiques historiques interactifs combinant arrivées et recettes réelles.
  - Référentiel hôtelier par défaut pour 6 villes stratégiques marocaines.

* **Prévisions IA (Forecasting)** :
  - Sélection interactive parmi les 3 modèles optimaux (SARIMA, Ridge, LSTM).
  - Évaluation historique comparative sur l'année test split de votre choix.
  - Projections futures personnalisables jusqu'en 2035 avec inflation et boost FIFA 2030 éditables en direct.
  - Exportation des résultats de prévision au format CSV.

* **Simulateur ROI Hôtelier** :
  - Paramétrage financier complet par sliders en temps réel (nombre de chambres, coût d'investissement, ADR, taux d'occupation, taux d'actualisation WACC).
  - Comparaison dynamique immédiate entre le scénario de référence et le scénario Coupe du Monde 2030 (qui intègre l'effet FIFA 2030).
  - Tableau de cash flows financiers sur 10 ans exportable en CSV.

* **Simulation Monte Carlo (Analyse de Risque)** :
  - Exécution stochastique de 200 à 1000 tirages pour simuler les incertitudes d'inflation, d'occupation et de boost FIFA.
  - Graphique de distribution des fréquences de rentabilité (ROI/VAN).
  - Calcul de la probabilité de perte ($P(\text{VAN} < 0)$) et de la Value at Risk (VaR 95%).

Architecture Technique de la Web App :
--------------------------------------
La plateforme interactive utilise une architecture web moderne à services découplés :

1. **Frontend (Single Page Application)** :
   - Propulsé par **React 18** et **Vite** pour un temps de chargement ultra-rapide.
   - Design moderne premium reposant sur le *Glassmorphism* avec un thème sombre élégant.
   - Graphiques réactifs et interactifs dessinés à l'aide de la bibliothèque **Recharts**.

2. **Backend (API REST asynchrone)** :
   - Construit en Python avec **FastAPI**, assurant une vitesse d'exécution optimale.
   - Validation stricte des requêtes par type via **Pydantic**.
   - Moteur mathématique et financier asynchrone pour les projections récursives et le générateur de Monte Carlo.


