Amélioration Méthodologique du Pipeline
=========================================

Dans le cadre de l'optimisation continue du pipeline de prévision touristique, une révision majeure de la stratégie de validation a été opérée pour les modèles avancés (Machine Learning avec XGBoost et Deep Learning avec LSTM/RNN). Le modèle SARIMAX, nativement conçu pour traiter la dépendance temporelle, reste inchangé.

Le Problème de l'Ancien Split
-----------------------------
Initialement, la séparation entre les données d'entraînement et de test pouvait s'apparenter à un split classique. Bien qu'un découpage chronologique (ex: 80% train / 20% test) ait été utilisé, les modèles complexes (XGBoost, Deep Learning) étaient évalués de manière statique. 

Plus problématique encore, une potentielle **fuite de données (Data Leakage)** existait dans la normalisation des données (Scaling) pour le Deep Learning. Le `MinMaxScaler` était ajusté (fit) sur l'ensemble complet des données (X et y) avant la séparation en fenêtres temporelles. Cela signifie que le modèle "voyait" les statistiques (minimum, maximum) de l'ensemble de test pendant son entraînement, faussant l'évaluation réelle de ses performances.

Solution : Walk-Forward Validation (TimeSeriesSplit)
----------------------------------------------------
Pour résoudre ce problème de manière rigoureuse, nous avons implémenté une validation **Walk-Forward** (via `TimeSeriesSplit`).

Le principe est le suivant :
1. Le modèle s'entraîne sur une fenêtre temporelle initiale de $T$ mois.
2. Il prédit le mois $T+1$.
3. La fenêtre d'entraînement s'étend pour inclure le mois $T+1$.
4. Le modèle est ré-entraîné et prédit le mois $T+2$, et ainsi de suite.

### Application au XGBoost
Pour XGBoost, l'historique complet est reconstitué, et le `TimeSeriesSplit` est utilisé pour avancer pas à pas sur l'ensemble de test, garantissant qu'à l'instant $t$, le modèle ne connaît aucune donnée future.

### Application au Deep Learning (LSTM)
Pour le Deep Learning, la préparation des données a été refaite. L'ajustement du scaler (`MinMaxScaler`) est désormais effectué **dynamiquement** à l'intérieur de la boucle de validation croisée temporelle :
- `fit_transform` est appliqué uniquement sur `X_train` et `y_train` de l'itération courante.
- `transform` est appliqué sur le `X_test` (la période à prédire).
Cela garantit une imperméabilité totale et supprime tout data leakage.

Impact sur les Performances
---------------------------
Les métriques évaluées par cette nouvelle méthode reflètent désormais le pouvoir prédictif réel des modèles sur des données totalement inconnues.

- **XGBoost** et **LSTM** : Les performances (R², RMSE, MAE, MAPE) ont été recalculées à l'aide de cette méthode. Les résultats générés montrent une variance plus réaliste des erreurs de prédiction (voir les nouveaux tableaux de métriques et le graphique comparatif `walk_forward_comparison.png`).

Conclusion et Recommandations
-----------------------------
L'intégration du Walk-Forward Validation hisse le pipeline au rang des standards de production pour le Time Series Forecasting.
Pour les futurs travaux, il est recommandé de :
- Maintenir cette méthode d'évaluation stricte pour toute nouvelle architecture testée.
- Utiliser la même approche pour l'optimisation des hyperparamètres (par ex. Optuna couplé avec un TimeSeriesSplit sur le set d'entraînement).
