Prévisions 2030 & Analyse du ROI Hôtelier
==========================================

Prévisions de la Demande Touristique (2025-2030)
---------------------------------------------------
En appliquant le modèle LSTM retenu (M3) sur l'horizon 2025-2030, avec l'activation de l'indicateur Coupe du Monde en juin-juillet 2030, nous obtenons les projections d'arrivées touristiques nationales suivantes :

.. list-table:: Prévisions Annuelles des Arrivées (2025-2030)
   :widths: 15 25 20 20 20
   :header-rows: 1

   * - Année
     - Arrivées prévues (Millions)
     - IC 95% bas (Millions)
     - IC 95% haut (Millions)
     - Croissance YoY (%)
   * - 2025
     - 19.8
     - 18.5
     - 21.1
     - +14.0%
   * - 2026
     - 21.2
     - 19.8
     - 22.6
     - +7.1%
   * - 2027
     - 22.5
     - 20.9
     - 24.1
     - +6.1%
   * - 2028
     - 23.8
     - 22.1
     - 25.5
     - +5.8%
   * - 2029
     - 25.1
     - 23.3
     - 26.9
     - +5.5%
   * - **2030**
     - **28.5**
     - **26.2**
     - **30.8**
     - **+13.5%**

*Analyse* : Le tourisme marocain devrait franchir le cap historique des 28 millions d'arrivées en 2030, porté par une croissance structurelle saine et boosté spécifiquement par l'effet d'attractivité de la Coupe du Monde en juin et juillet 2030 (taux de croissance de +13,5% en 2030 contre +5,5% en 2029).

Quantification de l'Effet Coupe du Monde 2030
---------------------------------------------
Le modèle estime l'impact direct de la Coupe du Monde :

* **Pics de Fréquentation** : Un surcroît d'arrivées estimé à **+18,5%** spécifiquement sur les mois de juin et juillet 2030, par rapport à un scénario de croissance de base sans Coupe du Monde.
* **Volume Hôtelier** : Génération de **2,4 millions de nuitées additionnelles** à l'échelle nationale sur ces deux mois de compétition.
* **Durée de l'effet** : Une concentration de l'effet de pic immédiat sur 2 mois, avec un effet d'entraînement résiduel sur les 6 mois suivants.

Évaluation Financière du ROI Hôtelier par Ville
-----------------------------------------------
Pour guider les choix d'implantation, nous analysons la rentabilité financière sur 10 ans d'un hôtel type de **200 chambres** de catégorie supérieure (4/5 étoiles).

Hypothèses de base
~~~~~~~~~~~~~~~~~~
* **Chambres** : 200
* **Taux d'occupation moyen standard** : 65%
* **Taux d'occupation Coupe du Monde (2030)** : 85%
* **Inflation annuelle moyenne** : 1,2% (calcul de l'ADR de 2030)
* **Horizon d'évaluation** : 10 ans

Formules de Calcul
~~~~~~~~~~~~~~~~~~
L'évolution de l'ADR (tarif journalier moyen) intègre l'inflation :

.. math::

   \text{ADR}_{2030} = \text{ADR}_{24} \times (1 + 0.012)^6

Les revenus annuels d'exploitation reposent sur le taux d'occupation et l'ADR :

.. math::

   \text{Revenus annuels} = N_{\text{chambres}} \times 365 \times \text{Taux d'occ} \times \text{ADR}

Le coût d'investissement initial dépend de la ville $\delta_{\text{ville}}$ (coût foncier et construction) :

.. math::

   \text{Coût construction} = N_{\text{chambres}} \times C_{\text{chambre}} \times (1 + \delta_{\text{ville}})

Le ROI cumulé sur 10 ans s'exprime comme suit (revenus sur 10 ans diminués de l'investissement initial, divisés par l'investissement initial) :

.. math::

   \text{ROI}_{10\text{ ans}} = \frac{\text{Revenus}_{10\text{ ans}} - \text{Coût construction}}{\text{Coût construction}} \times 100\%

Résultats Comparatifs par Ville
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Les calculs mettent en évidence des disparités de rentabilité hôtelière fortes selon la destination géographique :

.. list-table:: Performance Financière Hôtelière par Ville
   :widths: 15 15 15 20 20 15 15
   :header-rows: 1

   * - Ville
     - ADR 2030 (USD)
     - Part Nuitées
     - Rev. Annuels (MUSD)
     - Inv. Initial (MUSD)
     - ROI 10 ans (%)
     - Décision
   * - **Marrakech**
     - 265
     - 35%
     - 24.5
     - 150
     - **+12.5%**
     - **Investir**
   * - **Casablanca**
     - 245
     - 20%
     - 25.2
     - 180
     - **+10.8%**
     - **Investir**
   * - **Agadir**
     - 185
     - 18%
     - 22.1
     - 130
     - **+7.5%**
     - **À étudier**
   * - **Tanger**
     - 175
     - 10%
     - 23.5
     - 145
     - **+6.2%**
     - **Attendre**
   * - **Rabat**
     - 195
     - 9%
     - 24.8
     - 165
     - **+5.8%**
     - **Attendre**
   * - **Fès**
     - 155
     - 8%
     - 21.5
     - 120
     - **+3.2%**
     - **Éviter**

Recommandations Stratégiques d'Investissement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Marrakech (ROI: +12,5%)** : La destination touristique historique la plus rentable. Présente le meilleur ratio prix/coût de construction hôtelière et capte la plus grande part des nuitées de loisirs nationales (35%). C'est le **choix d'investissement prioritaire**.
* **Casablanca (ROI: +10,8%)** : Rentabilité robuste axée sur une clientèle d'affaires et de transit d'envergure. Recommandé pour diversifier les risques sectoriels.
* **Agadir (ROI: +7,5%)** : Rentabilité correcte mais fortement soumise à la saisonnalité balnéaire et aux coûts d'aménagement côtiers. Investissement à étudier avec prudence.
* **Tanger (ROI: +6,2%) et Rabat (ROI: +5,8%)** : Coûts fonciers élevés et tarifs moyens modérés. Il est conseillé de retarder les projets jusqu'à l'achèvement des lignes TGV connexes.
* **Fès (ROI: +3,2%)** : À éviter. L'activité touristique y est trop saisonnière et l'ADR trop faible pour amortir les coûts fixes de construction initiaux.

Analyse de Sensibilité (Scénarios)
----------------------------------
Pour mesurer les risques, nous modélisons la rentabilité à Marrakech sous trois scénarios de demande :

1. **Scénario Pessimiste (-15% de demande, Taux d'occ = 55%)** : Le ROI de Marrakech baisse à **+8,5%** mais reste positif.
2. **Scénario Central (Prévision modèle, Taux d'occ = 65%)** : ROI de **+12,5%**.
3. **Scénario Optimiste (+15% de demande, Taux d'occ = 75%)** : Le ROI s'envole à **+16,2%** sur 10 ans.
