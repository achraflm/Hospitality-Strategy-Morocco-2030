import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import linregress

# Ensure root is in python path
sys.path.append(os.path.abspath('.'))

from src.config import FIGURES_DIR, DATA_DIR, TARGET_COL, START_DATE, END_DATE, TEST_START_DATE, TRAIN_END_DATE
import src.data_loader as loader
import src.cleaning as cleaner
import src.visualization as viz
import src.features as feat
import src.models_ml as ml
import src.evaluation as eval_mod

def main():
    print("==================================================")
    print("Starting Programmatic Plot Generation Script")
    print("==================================================")
    
    # ----------------------------------------------------
    # PHASE 1 & 2: DATA LOADING, CLEANING, AND RECONSTRUCTION
    # ----------------------------------------------------
    print("\n[Phase 1 & 2] Loading and merging data...")
    merged_df = loader.load_and_merge_tourism_data()
    print(f"Initial merged shape: {merged_df.shape}")
    
    print("Integrating COVID real data...")
    merged_df = cleaner.integrate_covid_data(merged_df)
    
    print("Reconstructing historical arrivals and receipts...")
    merged_df = cleaner.reconstruct_historical_arrivals(merged_df)
    merged_df = cleaner.reconstruct_historical_receipts(merged_df)
    
    # Save cleaned data
    output_file = os.path.join(DATA_DIR, 'merged_tourism_data_final.csv')
    merged_df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")
    
    # ----------------------------------------------------
    # PLOT 1: Arrivals Evolution
    # ----------------------------------------------------
    print("\nGenerating Plot 1: Arrivals Evolution...")
    viz.plot_arrivals_evolution(merged_df, title="Évolution des Arrivées Touristiques au Maroc (1995-2026)")
    
    # ----------------------------------------------------
    # PLOT 2: Seasonal Decomposition
    # ----------------------------------------------------
    print("Generating Plot 2: Seasonal Decomposition...")
    decomp_data = merged_df.set_index('Date')['Arrivals']
    decomposition = seasonal_decompose(decomp_data, model='additive', period=12)
    viz.plot_seasonal_decomposition(decomposition, title="Décomposition Additive des Arrivées Touristiques")
    
    # ----------------------------------------------------
    # PLOT 3: Economic Variables Correlation Matrix
    # ----------------------------------------------------
    print("Generating Plot 3: Correlation Matrix of Economic Variables...")
    corr_columns = ['REER', 'Oil_price', 'FDI', 'Poverty_rate', 'Arrivals', 'Total_Receipts_MDH', 'is_covid']
    viz.plot_correlation_matrix(merged_df, corr_columns, title="Matrice de Corrélation des Variables Économiques")
    
    # ----------------------------------------------------
    # HOTEL ANALYSIS
    # ----------------------------------------------------
    print("\nProcessing hotel bookings...")
    raw_hotel = loader.load_hotel_bookings()
    hotel_monthly = cleaner.clean_and_resample_hotel_data(raw_hotel)
    
    # ----------------------------------------------------
    # PLOT 4: Hotel Monthly Trends
    # ----------------------------------------------------
    print("Generating Plot 4: Hotel Monthly Trends...")
    viz.plot_hotel_trends(hotel_monthly)
    
    # Save seasonal profile
    seasonal_profile = cleaner.get_hotel_seasonal_profile(hotel_monthly)
    seasonal_profile.to_csv('hotel_seasonal_profile.csv', index=False)
    print("Hotel seasonal profile saved to: hotel_seasonal_profile.csv")
    
    # ----------------------------------------------------
    # BENCHMARK ANALYSIS
    # ----------------------------------------------------
    print("\nProcessing benchmark data...")
    raw_bench = loader.load_hospitality_benchmark()
    bench_comp, bench_monthly = cleaner.process_hospitality_benchmark(raw_bench)
    
    # ----------------------------------------------------
    # PLOT 5: Comparative Benchmark Occupancy
    # ----------------------------------------------------
    print("Generating Plot 5: Comparative Benchmark Occupancy...")
    comparable_countries = ['Egypt', 'Turkey', 'Spain', 'France', 'UAE', 'Greece']
    viz.plot_benchmark_occupancy(bench_comp, comparable_countries)
    
    # ----------------------------------------------------
    # PLOT 6: Occupancy Correlation Heatmap between Countries
    # ----------------------------------------------------
    print("Generating Plot 6: Country Occupancy Correlation Heatmap...")
    bench_pivot = bench_comp.pivot_table(index='date', columns='Country', values='occupancy', aggfunc='mean')
    viz.plot_correlation_matrix(bench_pivot, comparable_countries, title="Corrélations d'Occupation entre Pays")
    
    # Save benchmark comparable data
    bench_monthly.to_csv('benchmark_comparable.csv', index=False)
    print("Benchmark comparable monthly data saved to: benchmark_comparable.csv")
    
    # ----------------------------------------------------
    # PHASE 3: FEATURE ENGINEERING
    # ----------------------------------------------------
    print("\n[Phase 3] Running Feature Engineering...")
    df_featured = feat.build_features(merged_df)
    features_list = feat.get_feature_list()
    
    # Save featured dataset
    output_featured_path = os.path.join(DATA_DIR, 'DATASET_ML_FEATURES.csv')
    df_featured.to_csv(output_featured_path, index=False)
    print(f"Featured dataset saved to: {output_featured_path}")
    
    # Splits Train/Test
    df_ml = df_featured.dropna(subset=[TARGET_COL]).copy()
    valid_features = [c for c in features_list if c in df_ml.columns]
    
    X_train = df_ml[df_ml['Date'] <= TRAIN_END_DATE][valid_features]
    y_train = df_ml[df_ml['Date'] <= TRAIN_END_DATE][TARGET_COL]
    X_test = df_ml[df_ml['Date'] >= TEST_START_DATE][valid_features]
    y_test = df_ml[df_ml['Date'] >= TEST_START_DATE][TARGET_COL]
    
    separated_dir = os.path.join(DATA_DIR, 'separted')
    os.makedirs(separated_dir, exist_ok=True)
    X_train.to_csv(os.path.join(separated_dir, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(separated_dir, 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(separated_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(separated_dir, 'y_test.csv'), index=False)
    print("Train/Test splits saved to 'data/separted'")
    
    # ----------------------------------------------------
    # PHASE 4 & 5: MACHINE LEARNING MODELS
    # ----------------------------------------------------
    print("\n[Phase 4 & 5] Training Machine Learning Models...")
    print("Fitting Ridge regression...")
    ridge_grid = ml.train_ridge(X_train, y_train, cv=3)
    
    print("Fitting SVR model...")
    svr_grid = ml.train_svr(X_train, y_train, cv=3)
    
    print("Fitting XGBoost model...")
    xgb_grid = ml.train_xgboost(X_train, y_train, cv=3)
    
    print("Fitting LightGBM model...")
    lgb_grid = ml.train_lightgbm(X_train, y_train, cv=3)
    
    print("Fitting CatBoost model...")
    cat_grid = ml.train_catboost(X_train, y_train, cv=3)
    
    # Make ML predictions
    predictions = {}
    predictions['Ridge'] = ridge_grid.predict(X_test)
    predictions['SVR'] = svr_grid.predict(X_test)
    if xgb_grid:
        predictions['XGBoost'] = xgb_grid.predict(X_test)
    if lgb_grid:
        predictions['LightGBM'] = lgb_grid.predict(X_test)
    if cat_grid:
        predictions['CatBoost'] = cat_grid.predict(X_test)
        
    # Hybrid Model
    weights = {'XGBoost': 0.5, 'CatBoost': 0.3, 'Ridge': 0.2}
    hybrid_forecaster = ml.HybridForecaster(weights=weights)
    predictions['Hybrid Model'] = hybrid_forecaster.predict(predictions)
    
    results_df = eval_mod.compare_models(predictions, y_test)
    print("\nML models performance comparison:")
    print(results_df.round(4).to_string())
    
    # Save ML metrics
    metrics_output_path = os.path.join(DATA_DIR, 'model_performance_metrics_ML.csv')
    results_df.to_csv(metrics_output_path, index=False)
    
    # ----------------------------------------------------
    # PLOT 7: ML and Hybrid predictions vs actual
    # ----------------------------------------------------
    print("\nGenerating Plot 7: ML Predictions Comparison...")
    test_dates = merged_df[merged_df['Date'] >= TEST_START_DATE]['Date'].values[:len(y_test)]
    viz.plot_predictions_comparison(y_test, predictions, test_dates, title="Comparaison des Modèles ML & Hybride")
    
    # ----------------------------------------------------
    # PLOT 8: Feature Importance
    # ----------------------------------------------------
    if xgb_grid:
        print("Generating Plot 8: XGBoost Feature Importance...")
        importance = ml.get_feature_importance(xgb_grid.best_estimator_, X_train.columns)
        viz.plot_feature_importance(importance, title="Importance des Caractéristiques (XGBoost)")
                
    # ----------------------------------------------------
    # PHASE 6: DEEP LEARNING MODELS
    # ----------------------------------------------------
    print("\n[Phase 6] Running Deep Learning pipeline...")
    # Import dynamically to ensure checked dependencies are loaded
    import src.models_dl as dl
    
    print("Preparing DL sequences...")
    X_train_dl, X_test_dl, y_train_dl, y_test_dl, scaler = dl.prepare_dl_sequences(merged_df, window_size=12, split_ratio=0.8)
    print(f"DL sequences train shape: {X_train_dl.shape}, test shape: {X_test_dl.shape}")
    
    # Run a quick Optuna search (only 3 trials and 5 epochs to run fast but check correctness)
    print("Running quick Optuna hyperparameter optimization (3 trials, 5 epochs)...")
    study = dl.optimize_dl_hyperparameters(X_train_dl, y_train_dl, X_test_dl, y_test_dl, n_trials=3, epochs=5)
    best_p = study.best_params
    print(f"Best trial parameters: {best_p}")
    
    # Train the best model on 15 epochs
    print(f"Training best model ({best_p['model_type']}) on 15 epochs...")
    input_shape = (X_train_dl.shape[1], X_train_dl.shape[2])
    best_model = dl.build_dl_model(
        model_type=best_p['model_type'],
        units=best_p['units'],
        dropout=best_p['dropout'],
        lr=best_p['lr'],
        input_shape=input_shape
    )
    best_model.fit(X_train_dl, y_train_dl, epochs=15, batch_size=16, verbose=0)
    
    # Predict and invert scaling
    y_pred_scaled = best_model.predict(X_test_dl)
    dummy_train = np.zeros((len(y_pred_scaled), X_train_dl.shape[2]))
    dummy_train[:, 0] = y_pred_scaled.flatten()
    y_pred_inv = scaler.inverse_transform(dummy_train)[:, 0]
    
    dummy_test = np.zeros((len(y_test_dl), X_train_dl.shape[2]))
    dummy_test[:, 0] = y_test_dl
    y_test_inv = scaler.inverse_transform(dummy_test)[:, 0]
    
    best_dl_metrics = eval_mod.calculate_metrics(y_test_inv, y_pred_inv)
    print(f"\nBest DL ({best_p['model_type']}) performance:")
    print(best_dl_metrics)
    
    # Transformer Model (trained on 15 epochs)
    print("Training Transformer model on 15 epochs...")
    transformer_model = dl.build_transformer_model(input_shape)
    import tensorflow as tf
    transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    transformer_model.fit(X_train_dl, y_train_dl, epochs=15, batch_size=16, validation_split=0.1, verbose=0)
    
    # Predict and invert scaling
    y_pred_trans_scaled = transformer_model.predict(X_test_dl)
    dummy_trans = np.zeros((len(y_pred_trans_scaled), X_train_dl.shape[2]))
    dummy_trans[:, 0] = y_pred_trans_scaled.flatten()
    y_pred_trans_inv = scaler.inverse_transform(dummy_trans)[:, 0]
    
    trans_metrics = eval_mod.calculate_metrics(y_test_inv, y_pred_trans_inv)
    print("\nTransformer model performance:")
    print(trans_metrics)
    
    # ----------------------------------------------------
    # PLOT 10: Deep Learning Predictions Comparison
    # ----------------------------------------------------
    print("\nGenerating Plot 10: Deep Learning Predictions Comparison...")
    dl_predictions = {
        f"Best DL ({best_p['model_type']})": y_pred_inv,
        "Transformer": y_pred_trans_inv
    }
    dl_test_dates = merged_df['Date'].values[-len(y_test_inv):]
    viz.plot_predictions_comparison(y_test_inv, dl_predictions, dl_test_dates, title="Comparaison des Modèles Deep Learning")
    
    # Combine final metrics
    global_results = pd.read_csv(metrics_output_path)
    best_dl_row = {
        'Model': f"Best DL ({best_p['model_type']})",
        'R2': best_dl_metrics['R2'],
        'RMSE': best_dl_metrics['RMSE'],
        'MAE': best_dl_metrics['MAE'],
        'MAPE': best_dl_metrics['MAPE']
    }
    trans_row = {
        'Model': 'Transformer',
        'R2': trans_metrics['R2'],
        'RMSE': trans_metrics['RMSE'],
        'MAE': trans_metrics['MAE'],
        'MAPE': trans_metrics['MAPE']
    }
    global_results = pd.concat([global_results, pd.DataFrame([best_dl_row, trans_row])], ignore_index=True)
    global_results = global_results.sort_values(by='R2', ascending=False)
    
    final_metrics_path = os.path.join(DATA_DIR, 'model_performance_metrics.csv')
    global_results.to_csv(final_metrics_path, index=False)
    print(f"\nFinal combined metrics saved to: {final_metrics_path}")
    print(global_results.round(4).to_string())
    
    print("\n==================================================")
    print("All figures and predictions generated successfully!")
    print("Check the figures directory:", FIGURES_DIR)
    print("==================================================")

if __name__ == '__main__':
    main()
