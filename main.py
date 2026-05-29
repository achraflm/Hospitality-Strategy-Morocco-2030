"""
Orchestrator Principal du Pipeline de Prévision Touristique Maroc 2030
====================================================================

Ce script exécute l'intégralité du pipeline de Data Science :
1. Chargement, fusion et nettoyage des données multi-sources.
2. Reconstruction historique des arrivées et recettes mensuelles (1996-2019).
3. Ingénierie des caractéristiques temporelles, cycliques et d'anomalies.
4. Entraînement et évaluation de 13 modèles de prévision (SARIMA, Prophet, 9 ML, 2 DL).
5. Sauvegarde des métriques de performance comparatives.
6. Projection récursive des arrivées et recettes à l'horizon 2030.
7. Simulation financière (ROI) d'un projet hôtelier de référence à Marrakech.
8. Génération de toutes les visualisations clés dans le dossier `figures/`.
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Ajout du répertoire courant au PYTHONPATH
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import FIGURES_DIR, DATA_DIR, TARGET_COL, START_DATE, END_DATE
import src.data_loader as loader
import src.cleaning as cleaner
import src.visualization as viz
import src.features as feat
import src.metrics as metrics_mod

# Import des modèles individuels
from src.models.sarima import SarimaModel
from src.models.ridge import RidgeModel
from src.models.lstm import LstmModel

def setup_logging(output_dir):
    """Configuration du double logger (Console et Fichier log)."""
    log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    log_file = os.path.join(output_dir, "pipeline.log")
    
    # Configuration racine
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("Morocco2030Pipeline")

def parse_arguments():
    """Analyseur d'arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description="Pipeline de Prévision Touristique Maroc 2030")
    parser.add_argument(
        "--test_year", 
        type=int, 
        default=2023, 
        help="Année de début pour le split de test (default: 2023)"
    )
    parser.add_argument(
        "--dl_epochs", 
        type=int, 
        default=50, 
        help="Nombre d'époques d'entraînement pour les modèles DL (default: 50)"
    )
    parser.add_argument(
        "--quick_run", 
        action="store_true", 
        help="Lancer en mode rapide (DL limité à 2 époques pour débugger)"
    )
    return parser.parse_args()

def run_data_pipeline(logger):
    """Étape 1 & 2 : Ingestion et nettoyage."""
    logger.info("Étape 1 & 2 : Chargement et fusion des données touristiques...")
    try:
        merged_df = loader.load_and_merge_tourism_data()
        logger.info(f"Dimensions initiales après fusion : {merged_df.shape}")
        
        logger.info("Intégration des données COVID réelles répertoriées...")
        merged_df = cleaner.integrate_covid_data(merged_df)
        
        logger.info("Reconstruction historique des arrivées (1996-2019)...")
        merged_df = cleaner.reconstruct_historical_arrivals(merged_df)
        
        logger.info("Reconstruction historique des recettes (1996-2019)...")
        merged_df = cleaner.reconstruct_historical_receipts(merged_df)
        
        # Sauvegarde du dataset principal
        output_file = os.path.join(DATA_DIR, 'merged_tourism_data_final.csv')
        merged_df.to_csv(output_file, index=False)
        logger.info(f"Dataset final sauvegardé -> {output_file}")
        
        return merged_df
    except Exception as e:
        logger.error(f"Échec dans l'étape de préparation des données : {e}", exc_info=True)
        sys.exit(1)

def run_exploratory_plots(df, logger):
    """Génération des visualisations EDA."""
    logger.info("Génération des figures d'analyse exploratoire (EDA)...")
    try:
        # 1. Évolution historique des arrivées
        viz.plot_arrivals_evolution(df, title="Évolution des Arrivées Touristiques au Maroc (1995-2026)")
        
        # 2. Décomposition Season-Trend STL
        decomp_data = df.set_index('Date')['Arrivals']
        decomposition = seasonal_decompose(decomp_data, model='additive', period=12)
        viz.plot_seasonal_decomposition(decomposition, title="Décomposition Additive des Arrivées Touristiques")
        
        # 2b. Autocorrélation (ACF/PACF)
        viz.plot_acf_pacf(decomp_data, lags=24, title="Arrivées Touristiques")
        
        # 3. Corrélations économiques
        corr_columns = ['REER', 'Oil_price', 'FDI', 'Poverty_rate', 'Arrivals', 'Total_Receipts_MDH', 'is_covid']
        viz.plot_correlation_matrix(df, corr_columns, title="Matrice de Corrélation des Variables Économiques")
        
        logger.info("Figures EDA sauvegardées dans le dossier figures/.")
    except Exception as e:
        logger.warning(f"Impossible de générer l'ensemble des figures d'EDA : {e}", exc_info=True)

def forecast_recursive_ml(model, df_historical, future_dates, valid_features, target_col='Arrivals'):
    """Prédiction récursive pas-à-pas pour les modèles de Machine Learning."""
    history_df = df_historical[['Date', target_col, 'Oil_price', 'FDI', 'Poverty_rate', 'REER']].copy()
    history_df['Date'] = pd.to_datetime(history_df['Date'])
    
    last_row = df_historical.iloc[-1]
    last_oil = last_row.get('Oil_price', 75.0)
    last_fdi = last_row.get('FDI', 2.0)
    last_poverty = last_row.get('Poverty_rate', 4.0)
    last_reer = last_row.get('REER', 100.0)
    
    predictions = []
    
    for date in future_dates:
        new_row = {
            'Date': date,
            target_col: np.nan,
            'Oil_price': last_oil,
            'FDI': last_fdi,
            'Poverty_rate': last_poverty,
            'REER': last_reer,
            'is_covid': 1 if (date >= pd.to_datetime('2020-03-01') and date <= pd.to_datetime('2021-12-01')) else 0
        }
        
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        idx = history_df.index[-1]
        
        # Recalculer les lags
        for k in [1, 2, 6, 12]:
            history_df.loc[idx, f'lags_{k}'] = history_df[target_col].iloc[-1 - k]
            
        # Moyennes mobiles
        for w in [3, 6, 12]:
            history_df.loc[idx, f'roll_mean_{w}'] = history_df[target_col].iloc[-1-w:-1].mean()
            history_df.loc[idx, f'roll_std_{w}'] = history_df[target_col].iloc[-1-w:-1].std()
            
        # Croissance YoY
        val_prev = history_df[target_col].iloc[-2]
        val_yoy = history_df[target_col].iloc[-14]
        history_df.loc[idx, 'growth_yoy'] = ((val_prev - val_yoy) / (val_yoy if val_yoy != 0 else 1.0)) * 100
        
        # Temporel et saisonnier
        m = date.month
        y = date.year
        q = (m - 1) // 3 + 1
        history_df.loc[idx, 'month_sin'] = np.sin(2 * np.pi * m / 12)
        history_df.loc[idx, 'month_cos'] = np.cos(2 * np.pi * m / 12)
        history_df.loc[idx, 'quarter'] = q
        history_df.loc[idx, 'year'] = y
        
        # Saison One-Hot
        season = 0 if m in [12, 1, 2] else 1 if m in [3, 4, 5] else 2 if m in [6, 7, 8] else 3
        history_df.loc[idx, 'saison_1'] = 1 if season == 1 else 0
        history_df.loc[idx, 'saison_2'] = 1 if season == 2 else 0
        history_df.loc[idx, 'saison_3'] = 1 if season == 3 else 0
        
        history_df.loc[idx, 'is_summer'] = 1 if m in [6, 7, 8] else 0
        history_df.loc[idx, 'is_high_season'] = 1 if m in [4, 5, 7, 8, 10, 12] else 0
        history_df.loc[idx, 'is_vacances'] = 1 if m in [7, 8, 12] else 0
        
        holiday_map = {1: 2, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1, 8: 3, 9: 0, 10: 0, 11: 2, 12: 0}
        history_df.loc[idx, 'jours_feries_count'] = holiday_map.get(m, 0)
        
        ramadan_months = {
            (2026, 2), (2026, 3), (2027, 2), (2028, 1), (2028, 2), (2029, 1), (2030, 1), (2030, 12)
        }
        history_df.loc[idx, 'is_ramadan'] = 1 if (y, m) in ramadan_months else 0
        history_df.loc[idx, 'is_special_event'] = 0
        
        for var in ['Oil_price', 'FDI', 'Poverty_rate', 'REER']:
            history_df.loc[idx, f'{var}_lag1'] = history_df[var].iloc[-2]
            history_df.loc[idx, f'{var}_lag3'] = history_df[var].iloc[-4]
            
        history_df.loc[idx, 'cdm_event'] = 1 if y == 2030 and m in [6, 7] else 0
        history_df.loc[idx, 'anomaly_zscore'] = 0
        history_df.loc[idx, 'anomaly_iforest'] = 0
        history_df.loc[idx, 'anomaly_prophet'] = 0
        
        # Vecteur de caractéristiques
        feat_vec = history_df[valid_features].iloc[[-1]].fillna(0)
        
        # Prédiction
        pred = model.predict(feat_vec)[0]
        pred = max(0.0, pred)
        
        history_df.loc[idx, target_col] = pred
        predictions.append(pred)
        
    return np.array(predictions)

def forecast_recursive_dl(model, df_historical, future_dates, valid_features, target_col='Arrivals'):
    """Prédiction récursive pas-à-pas pour les modèles Deep Learning (LSTM, RNN)."""
    history_df = df_historical[['Date', target_col, 'Oil_price', 'FDI', 'Poverty_rate', 'REER']].copy()
    history_df['Date'] = pd.to_datetime(history_df['Date'])
    
    last_row = df_historical.iloc[-1]
    last_oil = last_row.get('Oil_price', 75.0)
    last_fdi = last_row.get('FDI', 2.0)
    last_poverty = last_row.get('Poverty_rate', 4.0)
    last_reer = last_row.get('REER', 100.0)
    
    predictions = []
    
    for date in future_dates:
        new_row = {
            'Date': date,
            target_col: np.nan,
            'Oil_price': last_oil,
            'FDI': last_fdi,
            'Poverty_rate': last_poverty,
            'REER': last_reer,
            'is_covid': 1 if (date >= pd.to_datetime('2020-03-01') and date <= pd.to_datetime('2021-12-01')) else 0
        }
        
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        idx = history_df.index[-1]
        
        # Calculer les lags et features (identique à ML)
        for k in [1, 2, 6, 12]:
            history_df.loc[idx, f'lags_{k}'] = history_df[target_col].iloc[-1 - k]
        for w in [3, 6, 12]:
            history_df.loc[idx, f'roll_mean_{w}'] = history_df[target_col].iloc[-1-w:-1].mean()
            history_df.loc[idx, f'roll_std_{w}'] = history_df[target_col].iloc[-1-w:-1].std()
            
        val_prev = history_df[target_col].iloc[-2]
        val_yoy = history_df[target_col].iloc[-14]
        history_df.loc[idx, 'growth_yoy'] = ((val_prev - val_yoy) / (val_yoy if val_yoy != 0 else 1.0)) * 100
        
        m = date.month
        y = date.year
        q = (m - 1) // 3 + 1
        history_df.loc[idx, 'month_sin'] = np.sin(2 * np.pi * m / 12)
        history_df.loc[idx, 'month_cos'] = np.cos(2 * np.pi * m / 12)
        history_df.loc[idx, 'quarter'] = q
        history_df.loc[idx, 'year'] = y
        
        season = 0 if m in [12, 1, 2] else 1 if m in [3, 4, 5] else 2 if m in [6, 7, 8] else 3
        history_df.loc[idx, 'saison_1'] = 1 if season == 1 else 0
        history_df.loc[idx, 'saison_2'] = 1 if season == 2 else 0
        history_df.loc[idx, 'saison_3'] = 1 if season == 3 else 0
        
        history_df.loc[idx, 'is_summer'] = 1 if m in [6, 7, 8] else 0
        history_df.loc[idx, 'is_high_season'] = 1 if m in [4, 5, 7, 8, 10, 12] else 0
        history_df.loc[idx, 'is_vacances'] = 1 if m in [7, 8, 12] else 0
        
        holiday_map = {1: 2, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1, 8: 3, 9: 0, 10: 0, 11: 2, 12: 0}
        history_df.loc[idx, 'jours_feries_count'] = holiday_map.get(m, 0)
        
        ramadan_months = {
            (2026, 2), (2026, 3), (2027, 2), (2028, 1), (2028, 2), (2029, 1), (2030, 1), (2030, 12)
        }
        history_df.loc[idx, 'is_ramadan'] = 1 if (y, m) in ramadan_months else 0
        history_df.loc[idx, 'is_special_event'] = 0
        
        for var in ['Oil_price', 'FDI', 'Poverty_rate', 'REER']:
            history_df.loc[idx, f'{var}_lag1'] = history_df[var].iloc[-2]
            history_df.loc[idx, f'{var}_lag3'] = history_df[var].iloc[-4]
            
        history_df.loc[idx, 'cdm_event'] = 1 if y == 2030 and m in [6, 7] else 0
        history_df.loc[idx, 'anomaly_zscore'] = 0
        history_df.loc[idx, 'anomaly_iforest'] = 0
        history_df.loc[idx, 'anomaly_prophet'] = 0
        
        # Construire la séquence et normaliser
        last_window_df = history_df[valid_features].iloc[-model.window_size:].fillna(0)
        scaled_seq = model.scaler_x.transform(last_window_df.values)
        
        # Prédiction scaled
        pred_scaled = model.model.predict(scaled_seq[np.newaxis, :, :], verbose=0)[0, 0]
        
        # Inversion d'échelle
        pred = model.scaler_y.inverse_transform(np.array([[pred_scaled]]))[0, 0]
        pred = max(0.0, pred)
        
        history_df.loc[idx, target_col] = pred
        predictions.append(pred)
        
    return np.array(predictions)

def run_hotel_roi_simulation(logger):
    """Simule le ROI sur 10 ans d'un hôtel 5 étoiles de 200 chambres à Marrakech (Coupe du Monde vs Base)."""
    from src.roi_simulator import HotelROISimulator
    
    simulator = HotelROISimulator(
        rooms=200,
        investment_usd=40000000.0,
        opex_margin=0.65,
        discount_rate=0.08,
        base_occupancy=0.68,
        wc_occupancy_2030=0.75,
        base_adr=250.0,
        wc_adr_boost_pct=0.40,
        inflation_rate=0.025
    )
    
    df_roi = simulator.simulate_10years(start_year=2026)
    metrics = simulator.calculate_metrics(df_roi)
    
    logger.info(f"Hôtel ROI Base        : NPV = {metrics['NPV_Base_MUSD']:.2f} M$ | IRR = {metrics['IRR_Base_Pct']:.2f}%")
    logger.info(f"Hôtel ROI Coupe Monde : NPV = {metrics['NPV_WC_MUSD']:.2f} M$ | IRR = {metrics['IRR_WC_Pct']:.2f}%")
    
    # Save CSV
    roi_path = os.path.join(DATA_DIR, 'hotel_roi_simulation.csv')
    df_roi.to_csv(roi_path, index=False)
    logger.info(f"Table de simulation hôtelière sauvegardée dans -> {roi_path}")
    
    # Plot Cumulative Cash Flow
    simulator.plot_comparison(df_roi, metrics, save_path=os.path.join(FIGURES_DIR, '08_hotel_roi_simulation.png'))
    
    return df_roi

def main():
    """Point d'entrée du pipeline."""
    args = parse_arguments()
    
    # Configuration répertoire de logs et de figures
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    logger = setup_logging(DATA_DIR)
    
    logger.info("================================================================")
    logger.info(f"DÉMARRAGE DU PIPELINE TOURISME MAROC 2030 (TEST YEAR: {args.test_year})")
    logger.info("================================================================")
    start_time = time.time()
    
    # Quick run adjusts DL parameters
    dl_epochs = 2 if args.quick_run else args.dl_epochs
    logger.info(f"DL Epochs set to: {dl_epochs}")
    
    # 1. Pipeline de chargement, nettoyage et reconstruction
    merged_df = run_data_pipeline(logger)
    
    # 2. Analyse exploratoire graphique
    run_exploratory_plots(merged_df, logger)
    
    # 3. Feature Engineering et split
    df_featured = feat.build_features(merged_df)
    features_list = feat.get_feature_list()
    
    df_ml = df_featured.dropna(subset=[TARGET_COL]).copy()
    valid_features = [c for c in features_list if c in df_ml.columns]
    # Utiliser les données séparées du dossier backend/data/separted
    X_train_sep, X_test_sep, y_train_sep, y_test_sep = loader.get_separated_data(TARGET_COL)
    
    actual_features = [f for f in valid_features if f in X_train_sep.columns]
    X_train = X_train_sep[actual_features].fillna(X_train_sep[actual_features].median())
    X_test = X_test_sep[actual_features].fillna(X_test_sep[actual_features].median())
    
    y_train = y_train_sep
    y_test = y_test_sep
    
    logger.info("Données séparées chargées pour l'entraînement.")
    
    # 4. ENTRAÎNEMENT DES 3 MODÈLES
    predictions = {}
    
    # --- Modèles Statistiques ---
    logger.info("Entraînement SARIMA...")
    sarima = SarimaModel()
    sarima.fit(y_train)
    predictions['SARIMA'] = sarima.predict(steps=len(y_test))
    
    # --- Modèles de Machine Learning ---
    logger.info("Entraînement Ridge...")
    ridge = RidgeModel().fit(X_train, y_train)
    predictions['Ridge'] = ridge.predict(X_test)
    
    # --- Modèles Deep Learning ---
    logger.info("Entraînement LSTM...")
    lstm = LstmModel(epochs=dl_epochs)
    lstm.fit(X_train, y_train)
    predictions['LSTM'] = lstm.predict(X_test, X_train_history=X_train)
    
    # 5. Évaluation et enregistrement des métriques
    logger.info("Calcul des métriques pour l'ensemble des 3 modèles...")
    results_df = metrics_mod.compare_models(predictions, y_test)
    logger.info("\nTableau Comparatif des Performances :\n" + results_df.round(4).to_string())
    
    metrics_path = os.path.join(DATA_DIR, 'model_performance_metrics.csv')
    results_df.to_csv(metrics_path, index=False)
    logger.info(f"Bilan complet des performances sauvegardé -> {metrics_path}")
    
    # 6. Graphique 04 : Comparaison des prédictions sur le test set
    test_dates = df_ml[df_ml['Date'] >= test_start_date]['Date'].values[:len(y_test)]
    viz.plot_predictions_comparison(y_test, predictions, test_dates, title="Comparaison des Modèles sur l'Ensemble de Test")
    
    # 7. Importance des caractéristiques (Ridge)
    logger.info("Génération de l'importance des caractéristiques...")
    feature_importances = pd.Series(np.abs(ridge.model.coef_), index=X_train.columns).sort_values(ascending=False)
    viz.plot_feature_importance(feature_importances, title="Importance des Caractéristiques (Abs coef Ridge)")
    
    # 8. PROJECTION DE PREVISIONS JUSQU'EN 2030
    logger.info("Début des projections à l'horizon 2030 (mai 2026 à décembre 2030)...")
    steps = 56
    future_dates = pd.date_range(start='2026-05-01', periods=steps, freq='MS')
    
    # Identifier les meilleurs modèles de chaque catégorie
    best_ml_name = 'Ridge'
    best_dl_name = 'LSTM'
    
    logger.info(f"Meilleur modèle ML : {best_ml_name}")
    logger.info(f"Meilleur modèle DL : {best_dl_name}")
    
    # Associer les instances
    ml_models_dict = {
        'Ridge': ridge
    }
    dl_models_dict = {
        'LSTM': lstm
    }
    
    best_ml_model = ml_models_dict[best_ml_name]
    best_dl_model = dl_models_dict[best_dl_name]
    
    # Lancement des projections d'arrivées
    logger.info("Projection récursive avec le meilleur modèle ML...")
    arrivals_proj_ml = forecast_recursive_ml(best_ml_model, df_ml, future_dates, valid_features)
    
    logger.info("Projection récursive avec le meilleur modèle DL...")
    arrivals_proj_dl = forecast_recursive_dl(best_dl_model, df_ml, future_dates, valid_features)
    
    logger.info("Projection avec SARIMA...")
    # Re-fit sur le dataset complet
    sarima_full = SarimaModel().fit(df_ml[TARGET_COL])
    arrivals_proj_sarima = sarima_full.predict(steps=steps)
    
    # Invalidation de valeurs négatives
    arrivals_proj_ml = np.clip(arrivals_proj_ml, 0, None)
    arrivals_proj_dl = np.clip(arrivals_proj_dl, 0, None)
    arrivals_proj_sarima = np.clip(arrivals_proj_sarima, 0, None)
    
    # Save projections
    proj_df = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m-%d'),
        f'{best_ml_name} (Best ML)': arrivals_proj_ml,
        f'{best_dl_name} (Best DL)': arrivals_proj_dl,
        'SARIMA': arrivals_proj_sarima
    })
    proj_df_path = os.path.join(DATA_DIR, 'predictions_2030.csv')
    proj_df.to_csv(proj_df_path, index=False)
    logger.info(f"Prévisions 2030 d'arrivées enregistrées -> {proj_df_path}")
    
    # 9. PROJECTION DES RECETTES TOURISTIQUES
    # Calcul du ratio recettes / arrivées sur la période récente 2022-2025 (hors anomalies covid)
    recent_data = df_ml[(df_ml['Date'] >= '2022-01-01') & (df_ml['Date'] <= '2025-12-31')]
    mean_ratio = (recent_data['Total_Receipts_MDH'] / recent_data['Arrivals']).mean()
    logger.info(f"Ratio recettes/arrivées de base calculé : {mean_ratio:.5f} MDH par arrivée")
    
    inflation_rate = 0.015  # 1.5% d'inflation annuelle cumulée
    world_cup_boost = 0.20  # 20% boost sur les dépenses moyennes en 2030 (Coupe du Monde)
    
    def project_receipts(arrivals_pred):
        receipts = []
        for val, date in zip(arrivals_pred, future_dates):
            years_since_2026 = date.year - 2026
            # Inflation cumulée
            ratio = mean_ratio * ((1 + inflation_rate) ** years_since_2026)
            # Boost Coupe du monde
            if date.year == 2030:
                ratio = ratio * (1 + world_cup_boost)
            receipts.append(val * ratio)
        return np.array(receipts)
        
    receipts_proj_ml = project_receipts(arrivals_proj_ml)
    receipts_proj_dl = project_receipts(arrivals_proj_dl)
    receipts_proj_sarima = project_receipts(arrivals_proj_sarima)
    
    # Save receipts projections
    receipts_df = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m-%d'),
        f'{best_ml_name} (Best ML)': receipts_proj_ml,
        f'{best_dl_name} (Best DL)': receipts_proj_dl,
        'SARIMA': receipts_proj_sarima
    })
    receipts_df_path = os.path.join(DATA_DIR, 'receipts_predictions_2030.csv')
    receipts_df.to_csv(receipts_df_path, index=False)
    logger.info(f"Prévisions 2030 de recettes enregistrées -> {receipts_df_path}")
    
    # 10. TRACÉ DES PREVISIONS 2030
    # Graphique Arrivées 2030
    plt.figure(figsize=(14, 7))
    plt.plot(df_ml['Date'], df_ml['Arrivals'], color='black', label='Historique Réel', linewidth=1.5)
    plt.plot(future_dates, arrivals_proj_ml, color='teal', linestyle='-', label=f'{best_ml_name} (Best ML)', linewidth=2.0)
    plt.plot(future_dates, arrivals_proj_dl, color='orange', linestyle='--', label=f'{best_dl_name} (Best DL)', linewidth=2.0)
    plt.plot(future_dates, arrivals_proj_sarima, color='red', linestyle=':', label='SARIMA', linewidth=2.0)
    # Highlight 2030 World Cup period
    plt.axvspan('2030-06-01', '2030-07-31', color='teal', alpha=0.15, label='Coupe du Monde FIFA 2030')
    plt.title("Projections des Arrivées Touristiques au Maroc jusqu'à fin 2030")
    plt.xlabel("Année")
    plt.ylabel("Nombre d'Arrivées")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '06_arrivals_forecast_2030.png'), dpi=150)
    plt.close()
    
    # Graphique Recettes 2030
    plt.figure(figsize=(14, 7))
    plt.plot(df_ml['Date'], df_ml['Total_Receipts_MDH'], color='black', label='Historique Réel', linewidth=1.5)
    plt.plot(future_dates, receipts_proj_ml, color='teal', linestyle='-', label=f'{best_ml_name} (Best ML)', linewidth=2.0)
    plt.plot(future_dates, receipts_proj_dl, color='orange', linestyle='--', label=f'{best_dl_name} (Best DL)', linewidth=2.0)
    plt.plot(future_dates, receipts_proj_sarima, color='red', linestyle=':', label='SARIMA', linewidth=2.0)
    plt.axvspan('2030-06-01', '2030-07-31', color='teal', alpha=0.15, label='Coupe du Monde FIFA 2030')
    plt.title("Projections des Recettes Touristiques du Maroc jusqu'à fin 2030 (MDH)")
    plt.xlabel("Année")
    plt.ylabel("Recettes (MDH)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '07_receipts_forecast_2030.png'), dpi=150)
    plt.close()
    
    logger.info("Figures de prévisions 2030 générées.")
    
    # 11. Simulation financière hôtel Marrakech
    run_hotel_roi_simulation(logger)
    
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info("================================================================")
    logger.info(f"PIPELINE COMPLÉTÉ ENTIÈREMENT EN {elapsed:.2f} SECONDES !")
    logger.info(f"Toutes les figures de forecasting sont stockées dans : {FIGURES_DIR}")
    logger.info("================================================================")

if __name__ == '__main__':
    main()
