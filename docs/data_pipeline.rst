Pipeline de Traitement des Données
=================================

Architecture Multi-Sources
---------------------------
Pour capturer la dynamique complexe de la demande, le pipeline de données fusionne quatre grandes familles de sources :

1. **Données de Flux Touristiques** : 
   * Historique annuel (1995-2020) des arrivées aux frontières et des recettes touristiques issues des plateformes officielles (HCP, Ministère du Tourisme).
   * Données mensuelles (2010-2025) issues de TradingEconomics servant de série cible principale.
   * Données de récentes performances collectées manuellement (14,5 millions en 2023, 17,4 millions en 2024 et 19,8 millions en 2025).

2. **Données Hôtelières Locales** : 
   * Extraits du dataset comportemental `hotel_bookings_clean.csv` décrivant les indicateurs opérationnels réels (taux d'occupation, ADR moyen, taux d'annulation).

3. **Indicateurs Macroéconomiques Externes** : 
   * Données d'indicateurs économiques du Maroc (`Morocco_cleaned.csv`) incluant le taux de change effectif réel (REER), le prix du pétrole (Brent) pour le coût du transport, l'inflation sectorielle (CPI Hotels), les investissements directs étrangers (FDI) et le taux de pauvreté.

4. **Benchmarks d'Hospitalité Internationaux** : 
   * Données comparatives des destinations concurrentes de la zone méditerranéenne et du Moyen-Orient (Égypte, Turquie, Espagne, France, Grèce, EAU) via `hospitality_benchmark_clean.csv`.

Intégration des Données COVID
-----------------------------
La pandémie du COVID-19 (mars 2020 à décembre 2021) représente un choc exogène massif qui fausse les modèles d'apprentissage automatique s'il n'est pas modélisé.
Le pipeline de nettoyage :

* Écrase les valeurs aberrantes de cette période avec les données mensuelles réelles répertoriées durant la crise (fermeture totale puis partielle).
* Initialise un flag binaire `is_covid = 1` pour cette période. Ce flag permet aux modèles d'isoler l'impact négatif et d'éviter que le choc n'altère la prévision de la tendance à long terme.

Reconstruction Mensuelle Historique
-----------------------------------
Puisque les données mensuelles détaillées ne sont pas disponibles avant 2016 (seuls les totaux annuels sont fiables de 1996 à 2015), nous avons conçu un algorithme de reconstruction temporelle :

1. **Extraction de Saisonnalité** : Nous appliquons une décomposition multiplicative via `seasonal_decompose` de la bibliothèque statsmodels sur la période récente et stable (2022-2026) pour extraire le profil saisonnier typique du tourisme au Maroc (12 coefficients mensuels).
2. **Désagrégation Temporelle** : Les coefficients de saisonnalité sont normalisés (somme = 12) et appliqués sur le niveau moyen mensuel historique (Total Annuel / 12) de chaque année correspondante.
3. **Ajout de Bruit** : Un bruit gaussien, calibré sur la variance résiduelle de la période récente, est injecté pour rendre les données simulées statistiquement représentatives.
4. **Reconstruction des Recettes** : La même logique de désagrégation est appliquée aux recettes touristiques annuelles historiques.

Traitement des Données Hôtelières et Benchmark
----------------------------------------------
* **Données Hôtelières** : Les données comportementales brutes de réservations sont rééchantillonnées à une fréquence mensuelle. Le pipeline applique des fonctions d'agrégation adaptées : la moyenne (`mean`) pour les taux (taux d'occupation, taux d'annulation) et la somme (`sum`) pour les volumes de revenus.
* **Hospitalité Comparative** : Le pipeline filtre les indicateurs des pays de comparaison, calcule le taux d'occupation de référence pré-COVID pour chaque pays et définit un ratio de récupération mensuel post-pandémie.
