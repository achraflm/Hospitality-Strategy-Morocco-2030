"""
Application Web Interactive de Prévisions Touristiques (Sans ROI)
================================================================

Ce tableau de bord se concentre exclusivement sur les prévisions des arrivées
et des nuitées touristiques, avec une validation différenciée selon les modèles
(Normale pour le ML classique, Walk-Forward + AutoResearch pour le Deep Learning).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import TARGET_COL, NIGHTS_COL
import src.data_loader as loader
import src.cleaning as cleaner
import src.features as feat
from src.autoresearch import AutoResearchEvaluator

from src.models.sarima import SarimaModel
from src.models.prophet import ProphetModel
from src.models.ridge import RidgeModel
from src.models.random_forest import RandomForestModel
from src.models.extra_trees import ExtraTreesModel
from src.models.gradient_boosting import GradientBoostingModel
from src.models.adaboost import AdaBoostModel
from src.models.xgboost import XgboostModel
from src.models.lightgbm import LightgbmModel
from src.models.catboost import CatboostModel
from src.models.svr import SvrModel
from src.models.lstm import LstmModel
from src.models.rnn import RnnModel

st.set_page_config(page_title="Morocco Tourism Forecasting", page_icon="🇲🇦", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    h1, h2, h3 { color: #0f172a; font-family: 'Outfit', sans-serif; font-weight: 700; }
    div.stButton > button:first-child {
        background-color: #0d9488; color: white; font-weight: 600;
        border-radius: 8px; border: none; padding: 12px; width: 100%;
    }
    .metric-card {
        background-color: white; padding: 15px; border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border-top: 4px solid #0d9488;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_clean_tourism_data():
    df = loader.load_and_merge_tourism_data()
    df = cleaner.integrate_covid_data(df)
    df = cleaner.reconstruct_historical_arrivals(df)
    df = cleaner.reconstruct_historical_receipts(df)
    return df

df_clean = get_clean_tourism_data()

def forecast_model(model_name, X_train, y_train, df_ml_train, test_dates, selected_features, dl_epochs=10):
    """ Entraîne et prédit en mode récursif pour la période de test / futur """
    from main import forecast_recursive_ml, forecast_recursive_dl
    
    if model_name == 'SARIMA':
        model = SarimaModel().fit(y_train)
        preds = model.predict(steps=len(test_dates))
        return np.clip(preds, 0, None)
    
    elif model_name == 'Prophet':
        exog_cols = [c for c in selected_features if c in df_ml_train.columns and c not in [y_train.name, 'Date']]
        model = ProphetModel().fit(df_ml_train, target_col=y_train.name, exog_cols=exog_cols)
        future_df = pd.DataFrame({'Date': test_dates})
        for col in exog_cols:
            future_df[col] = df_ml_train[col].iloc[-1]
        preds = model.predict(future_df)
        return np.clip(preds, 0, None)
        
    elif model_name in ['LSTM', 'SimpleRNN']:
        dl_class_map = {'LSTM': LstmModel, 'SimpleRNN': RnnModel}
        model = dl_class_map[model_name](epochs=dl_epochs).fit(X_train, y_train)
        preds = forecast_recursive_dl(model, df_ml_train, test_dates, selected_features)
        return np.clip(preds, 0, None)
        
    else:
        ml_class_map = {
            'Ridge': RidgeModel, 'Random Forest': RandomForestModel,
            'Extra Trees': ExtraTreesModel, 'Gradient Boosting': GradientBoostingModel,
            'AdaBoost': AdaBoostModel, 'XGBoost': XgboostModel,
            'LightGBM': LightgbmModel, 'CatBoost': CatboostModel, 'SVR': SvrModel
        }
        if model_name in ml_class_map:
            model = ml_class_map[model_name]().fit(X_train, y_train)
            preds = forecast_recursive_ml(model, df_ml_train, test_dates, selected_features)
            return np.clip(preds, 0, None)
    return None

# ---- SIDEBAR ----
st.sidebar.header("🛠️ Configuration des Prévisions")

st.sidebar.info(
    "🤖 **Mode Autoresearch**\n\n"
    "Le module d'IA autonome analyse automatiquement les modèles de Deep Learning "
    "(Walk-Forward) pour générer des insights textuels sur leur performance."
)

pred_target = st.sidebar.radio("Cible de Prédiction", ["Arrivées touristiques", "Nuitées hôtelières"])
use_nights = pred_target.startswith("Nuitées")
active_target_col = NIGHTS_COL if use_nights else TARGET_COL

st.sidebar.subheader("🤖 Modélisation")
available_models = ['Ridge', 'XGBoost', 'LSTM', 'SARIMA', 'Prophet', 'Random Forest', 'Gradient Boosting', 'LightGBM']
selected_models = st.sidebar.multiselect("Modèles à comparer", available_models, default=['Ridge', 'XGBoost', 'LSTM'])

dl_epochs = st.sidebar.slider("Époques Deep Learning", 1, 50, 10, 5)

st.sidebar.subheader("🔮 Horizon de Projection")
proj_end_year = st.sidebar.slider("Année de fin de prévision", 2028, 2040, 2030)

# Features
default_features = feat.get_nights_feature_list() if use_nights else feat.get_feature_list()
selected_features = st.sidebar.multiselect("Variables d'entrée (Features)", default_features, default=[f for f in default_features if f in ['month_sin', 'month_cos', 'year', 'cdm_event', 'is_covid', 'lags_1', 'lags_12', 'nights_lag_1']])

# ---- MAIN PAGE ----
st.title("🇲🇦 Morocco Tourism Forecasting Dashboard")
st.markdown("Ce tableau de bord permet de tester les modèles normaux et Deep Learning. Les modèles de Deep Learning bénéficient d'un traitement spécial : le **Walk-Forward Training** et l'analyse **AutoResearch**.")

sim_btn = st.button("🚀 Lancer l'Évaluation & Projections")

if sim_btn:
    if not selected_models or not selected_features:
        st.error("Sélectionnez au moins un modèle et une feature.")
    else:
        # Preprocessing
        df_featured = feat.build_features(df_clean)
        df_ml = df_featured.dropna(subset=[active_target_col]).copy()
        
        # Isoler l'historique jusqu'en 2025 pour l'évaluation et l'entraînement final
        df_history = df_ml[df_ml['Date'].dt.year <= 2025].copy()
        
        valid_sel = [f for f in selected_features if f in df_history.columns]
        X = df_history[valid_sel].fillna(df_history[valid_sel].median())
        y = df_history[active_target_col]
        
        wf_metrics = {m: {'r2': 0, 'mae': 0, 'insights': '', 'val_type': ''} for m in selected_models}
        
        # Initialize AutoResearch Evaluator
        ar_evaluator = AutoResearchEvaluator(output_dir="notebooks/results/autoresearch_output")
        target_name_str = "Nuitées" if use_nights else "Arrivées"
        
        with st.spinner("⏳ Évaluation des modèles en cours (Standard pour ML, Walk-Forward pour Deep Learning)..."):
            for model in selected_models:
                is_dl = model in ['LSTM', 'SimpleRNN']
                
                if is_dl:
                    # Walk-Forward Training + AutoResearch pour Deep Learning
                    tscv = TimeSeriesSplit(n_splits=3)
                    all_y_true = []
                    all_y_pred = []
                    
                    for split_idx, (train_index, test_index) in enumerate(tscv.split(X)):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                        df_train_wf = df_history.iloc[train_index]
                        test_dates = df_history.iloc[test_index]['Date']
                        
                        preds = forecast_model(model, X_train, y_train, df_train_wf, test_dates, valid_sel, dl_epochs)
                        if preds is not None:
                            all_y_true.extend(y_test)
                            all_y_pred.extend(preds)
                            
                    if all_y_true:
                        wf_metrics[model]['r2'] = r2_score(all_y_true, all_y_pred)
                        wf_metrics[model]['mae'] = mean_absolute_error(all_y_true, all_y_pred)
                        wf_metrics[model]['val_type'] = "Walk-Forward (3 splits)"
                        
                        # Apply AutoResearch
                        res = ar_evaluator.evaluate_model(target_name_str, model, all_y_true, all_y_pred, is_walk_forward=True)
                        wf_metrics[model]['insights'] = res['Insights']
                
                else:
                    # Traitement Normal (80% Train / 20% Test) pour Modèles ML Classiques
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                    df_train_wf = df_history.iloc[:split_idx]
                    test_dates = df_history.iloc[split_idx:]['Date']
                    
                    preds = forecast_model(model, X_train, y_train, df_train_wf, test_dates, valid_sel, dl_epochs)
                    if preds is not None:
                        wf_metrics[model]['r2'] = r2_score(y_test, preds)
                        wf_metrics[model]['mae'] = mean_absolute_error(y_test, preds)
                        wf_metrics[model]['val_type'] = "Standard (Train/Test)"
                        
                        # Optionnel : Générer un insight basique même pour ML
                        res = ar_evaluator.evaluate_model(target_name_str, model, y_test, preds, is_walk_forward=False)
                        wf_metrics[model]['insights'] = res['Insights']
                        
        st.subheader("🧪 Résultats d'Évaluation & AutoResearch Insights")
        cols = st.columns(len(selected_models))
        for idx, model in enumerate(selected_models):
            with cols[idx % len(cols)]:
                m_r2 = wf_metrics[model]['r2']
                m_mae = wf_metrics[model]['mae']
                m_val = wf_metrics[model]['val_type']
                m_ins = wf_metrics[model]['insights']
                
                st.markdown(f"<div class='metric-card'><h4>{model}</h4>"
                            f"<p style='color: gray; font-size: 0.9em;'>Évaluation: {m_val}</p>"
                            f"<b>R²:</b> {m_r2:.3f}<br>"
                            f"<b>MAE:</b> {m_mae:,.0f}<br><hr>"
                            f"<i>💡 Insights (AutoResearch): {m_ins}</i></div>", unsafe_allow_html=True)
                            
        with st.spinner(f"🔮 Entraînement final et Projections jusqu'en {proj_end_year}..."):
            future_dates = pd.date_range(start='2026-01-01', end=f'{proj_end_year}-12-01', freq='MS')
            projections = {}
            
            for model in selected_models:
                preds = forecast_model(model, X, y, df_history, future_dates, valid_sel, dl_epochs)
                projections[model] = preds
                
            st.subheader("📈 Projections Futures")
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Historique
            ax.plot(df_history['Date'], df_history[active_target_col], color='black', linewidth=2, label='Historique (Réel)')
            
            # Projections
            colors = ['#0d9488', '#d97706', '#2563eb', '#db2777', '#7c3aed', '#10b981']
            for idx, model in enumerate(selected_models):
                if projections[model] is not None:
                    ax.plot(future_dates, projections[model], color=colors[idx % len(colors)], linestyle='--', linewidth=2, label=f"Prévision {model}")
            
            target_name = "Nuitées Hôtelières" if use_nights else "Arrivées Touristiques"
            ax.set_title(f"Historique et Projections des {target_name} (Horizon {proj_end_year})", fontsize=14, fontweight='bold')
            ax.set_ylabel(target_name)
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
