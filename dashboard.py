"""
Application Streamlit de Modélisation et Simulation Touristique Maroc 2030
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit

# S'assurer que le répertoire racine est dans le python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import DATA_DIR
import src.data_loader as loader
import src.cleaning as cleaner
import src.features as feat
import src.metrics as metrics_mod

# Import de tous les modèles
from src.models.xgboost import XgboostModel
from src.models.lstm import LstmModel
from src.models.gru import GruModel
from src.models.lstm_deep import LstmDeepModel
from src.models.lstm_cnn import LstmCnnModel
from src.models.sarima import SarimaModel
from src.models.ridge import RidgeModel
from src.roi_simulator import HotelROISimulator

from main import forecast_recursive_ml, forecast_recursive_dl

st.set_page_config(page_title="Morocco Tourism Forecasting Dashboard", page_icon="🇲🇦", layout="wide", initial_sidebar_state="expanded")

# CSS
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    h1, h2, h3 { color: #0f172a; font-family: 'Inter', sans-serif; font-weight: 700; }
    .metric-card { background-color: white; padding: 24px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-top: 4px solid #0d9488; margin-bottom: 15px; }
    .metric-card h4 { margin: 0 0 8px 0; color: #475569; font-size: 14px; font-weight: 600; }
    .metric-card p.val { margin: 0; font-size: 28px; font-weight: 700; color: #0f172a; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_clean_tourism_data():
    df = loader.load_and_merge_tourism_data()
    df = cleaner.integrate_covid_data(df)
    df = cleaner.reconstruct_historical_arrivals(df)
    df = cleaner.reconstruct_historical_receipts(df)
    return df

st.title("🇲🇦 Tableau de Bord Prévisionnel du Tourisme - Horizon 2030 & Au-delà")

st.sidebar.header("🛠️ Configuration des Expériences")

target_variable = st.sidebar.selectbox(
    "Cible de Prévision",
    options=["Arrivals", "Nights"],
    index=0
)
TARGET_COL = target_variable

split_year = st.sidebar.slider("Année de début du test split", min_value=2020, max_value=2025, value=2023)

eval_method = st.sidebar.radio(
    "Méthode d'évaluation",
    options=["Walk-Forward", "Normal"]
)

st.sidebar.subheader("📋 Feature Engineering")
if TARGET_COL == "Arrivals":
    default_features = feat.get_feature_list()
else:
    default_features = feat.get_nights_feature_list()

selected_features = st.sidebar.multiselect(
    "Choisir les colonnes", options=default_features,
    default=[f for f in default_features if f in ['lags_1', 'lags_2', 'lags_12', 'growth_yoy', 'month_sin', 'month_cos', 'year', 'is_high_season', 'cdm_event']]
)

st.sidebar.subheader("🤖 Modèles Prédictifs")
selected_models = st.sidebar.multiselect(
    "Modèles à évaluer", 
    ["XGBoost", "LSTM", "LSTM 2 Layers", "LSTM + CNN", "GRU", "SARIMA", "Ridge"],
    default=["XGBoost", "LSTM", "LSTM 2 Layers", "LSTM + CNN", "GRU", "SARIMA", "Ridge"]
)

dl_epochs = st.sidebar.number_input("Époques DL", min_value=1, max_value=100, value=5, step=5)

df_clean = get_clean_tourism_data()

tab_data, tab_train, tab_forecast = st.tabs([
    "📊 Exploration", "🤖 Entraînement", "📈 Projections 2030"
])

with tab_data:
    st.header(f"Analyse des {TARGET_COL}")
    st.line_chart(df_clean.set_index('Date')[TARGET_COL])

with tab_train:
    st.header(f"Entraînement Dynamique ({eval_method})")
    run_btn = st.button("🚀 Entraîner les Modèles")
    
    if run_btn and selected_models and selected_features:
        with st.spinner("Modélisation en cours..."):
            df_featured = feat.build_features(df_clean)
            df_ml = df_featured.dropna(subset=[TARGET_COL]).copy()
            
            train_end_date = f"{split_year - 1}-12-31"
            test_start_date = f"{split_year}-01-01"
            
            train_df = df_ml[df_ml['Date'] <= train_end_date]
            test_df = df_ml[df_ml['Date'] >= test_start_date]
            
            X_train = train_df[selected_features].fillna(train_df[selected_features].median())
            y_train = train_df[TARGET_COL]
            X_test = test_df[selected_features].fillna(train_df[selected_features].median())
            y_test = test_df[TARGET_COL]
            
            X_all_raw = pd.concat([X_train, X_test])
            y_all_raw = pd.concat([y_train, y_test]).values
            train_size = len(X_train)
            n_splits = len(X_test)
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=1)
            
            ml_class_map = {
                'XGBoost': XgboostModel,
                'SARIMA': SarimaModel,
                'Ridge': RidgeModel
            }
            dl_class_map = {
                'LSTM': LstmModel,
                'LSTM 2 Layers': LstmDeepModel,
                'LSTM + CNN': LstmCnnModel,
                'GRU': GruModel
            }

            predictions = {}
            for m_name in selected_models:
                if m_name in ml_class_map:
                    if eval_method == "Walk-Forward":
                        y_pred = []
                        my_bar = st.progress(0, text=f"Walk-Forward {m_name}...")
                        for idx, (train_index, test_index) in enumerate(tscv.split(X_all_raw)):
                            if test_index[0] < train_size: continue
                            X_tr, y_tr = X_all_raw.iloc[train_index], y_all_raw[train_index]
                            X_te = X_all_raw.iloc[test_index]
                            model = ml_class_map[m_name]().fit(X_tr, y_tr)
                            pred = model.predict(X_te)
                            y_pred.append(pred[0])
                            my_bar.progress((idx + 1) / n_splits)
                        my_bar.empty()
                        predictions[m_name] = np.array(y_pred)
                    else:
                        with st.spinner(f"Normal Train {m_name}..."):
                            model = ml_class_map[m_name]().fit(X_train, y_train)
                            predictions[m_name] = np.array(model.predict(X_test))
                            
                elif m_name in dl_class_map:
                    if eval_method == "Walk-Forward":
                        y_pred = []
                        my_bar = st.progress(0, text=f"Walk-Forward {m_name}...")
                        for idx, (train_index, test_index) in enumerate(tscv.split(X_all_raw)):
                            if test_index[0] < train_size: continue
                            X_tr, y_tr = X_all_raw.iloc[train_index], y_all_raw[train_index]
                            X_te = X_all_raw.iloc[test_index]
                            model = dl_class_map[m_name](epochs=dl_epochs).fit(X_tr, y_tr)
                            pred = model.predict(X_te, X_train_history=X_tr)
                            y_pred.append(pred[0])
                            my_bar.progress((idx + 1) / n_splits)
                        my_bar.empty()
                        predictions[m_name] = np.array(y_pred)
                    else:
                        with st.spinner(f"Normal Train {m_name}..."):
                            model = dl_class_map[m_name](epochs=dl_epochs).fit(X_train, y_train)
                            predictions[m_name] = np.array(model.predict(X_test, X_train_history=X_train))
            
            results_df = metrics_mod.compare_models(predictions, y_test)
            st.session_state['results_df'] = results_df
            st.session_state['predictions'] = predictions
            st.session_state['y_test'] = y_test
            st.session_state['test_dates'] = test_df['Date'].values
            st.session_state['eval_method_used'] = eval_method
            
    if 'results_df' in st.session_state:
        eval_used = st.session_state.get('eval_method_used', 'Walk-Forward')
        st.subheader(f"Performance sur Test ({eval_used})")
        styled_df = st.session_state['results_df'].style.highlight_max(subset=['R2'], color='#115e59')
        st.dataframe(styled_df)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(st.session_state['test_dates'], st.session_state['y_test'], label='Réel', color='black', lw=2)
        for name, pred in st.session_state['predictions'].items():
            ax.plot(st.session_state['test_dates'], pred.flatten(), label=name, linestyle='--')
        plt.legend()
        st.pyplot(fig)

with tab_forecast:
    st.header(f"Projections 2030 : {TARGET_COL}")
    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']
        
        ml_names = ["XGBoost", "Ridge"]
        dl_names = ["LSTM", "LSTM 2 Layers", "LSTM + CNN", "GRU"]
        
        avail_ml = results_df[results_df['Model'].isin(ml_names)]
        avail_dl = results_df[results_df['Model'].isin(dl_names)]
        
        best_ml_name = avail_ml['Model'].iloc[0] if len(avail_ml) > 0 else None
        best_dl_name = avail_dl['Model'].iloc[0] if len(avail_dl) > 0 else None
        
        status = []
        if best_ml_name: status.append(best_ml_name)
        if best_dl_name: status.append(best_dl_name)
            
        st.info(f"Modèles retenus pour projection : {' | '.join(status)}")
        
        st.markdown("### ⚙️ Paramètres d'Inflation et Coupe du Monde 2030")
        col1, col2, col3 = st.columns(3)
        with col1:
            inflation_rate = st.slider("Taux d'inflation annuel moyen (%)", 0.0, 5.0, 2.5, 0.5) / 100
        with col2:
            wc_boost = st.slider("Boost d'attractivité/Prix Coupe du Monde (%)", 0, 80, 20, 5) / 100
        with col3:
            wc_opex_inflation = st.slider("Choc Inflation OPEX (2030) (%)", 0.0, 20.0, 6.3, 0.1) / 100
            st.caption("💡 *Recommandation : 6.3% (Moyenne historique des chocs d'inflation lors des Coupes du Monde 2014, 2018, 2022)*")
        
        if st.button("Lancer la Projection"):
            with st.spinner("Projection récursive..."):
                df_featured = feat.build_features(df_clean)
                df_ml = df_featured.dropna(subset=[TARGET_COL]).copy()
                X_train_full = df_ml[selected_features].fillna(df_ml[selected_features].median())
                y_train_full = df_ml[TARGET_COL]
                
                future_dates = pd.date_range(start='2026-05-01', end='2030-12-01', freq='MS')
                
                proj_results = {}
                
                ml_class_map = {
                    'XGBoost': XgboostModel,
                    'SARIMA': SarimaModel,
                    'Ridge': RidgeModel
                }
                if best_ml_name in ml_class_map:
                    model = ml_class_map[best_ml_name]().fit(X_train_full, y_train_full)
                    proj = forecast_recursive_ml(model, df_ml, future_dates, selected_features, target_col=TARGET_COL)
                    proj_results[best_ml_name] = np.clip(proj, 0, None)
                    
                dl_class_map = {'LSTM': LstmModel, 'LSTM 2 Layers': LstmDeepModel, 'LSTM + CNN': LstmCnnModel, 'GRU': GruModel}
                if best_dl_name:
                    model = dl_class_map[best_dl_name](epochs=dl_epochs).fit(X_train_full, y_train_full)
                    proj = forecast_recursive_dl(model, df_ml, future_dates, selected_features, target_col=TARGET_COL)
                    proj_results[best_dl_name] = np.clip(proj, 0, None)
                    
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(df_ml['Date'], df_ml[TARGET_COL], label='Historique', color='black')
                for name, proj in proj_results.items():
                    ax.plot(future_dates, proj, label=f"Proj {name}", linestyle='--')
                ax.axvspan('2030-06-01', '2030-07-31', color='teal', alpha=0.15, label='Coupe du Monde')
                plt.title(f"Projection des {TARGET_COL} jusqu'en 2030")
                plt.legend()
                st.pyplot(fig)
                
                # Si on prédit les arrivées, on peut estimer les recettes (avec inflation)
                if TARGET_COL == 'Arrivals' and 'Total_Receipts_MDH' in df_clean.columns:
                    st.subheader("💰 Projection des Recettes Touristiques (MDH) avec Inflation")
                    mean_ratio = (df_clean['Total_Receipts_MDH'] / df_clean['Arrivals']).mean()
                    
                    fig_rec, ax_rec = plt.subplots(figsize=(14, 6))
                    ax_rec.plot(df_ml['Date'], df_ml['Total_Receipts_MDH'], label='Recettes Historiques', color='green')
                    
                    for name, proj in proj_results.items():
                        receipts = []
                        for arr, date in zip(proj, future_dates):
                            years_since_2026 = date.year - 2026
                            ratio = mean_ratio * ((1 + inflation_rate) ** years_since_2026)
                            if date.year == 2030:
                                ratio = ratio * (1 + wc_boost)
                            # Remarque: l'inflation OPEX (wc_opex_inflation) impacte le ROI, pas directement les recettes macro.
                            receipts.append(arr * ratio)
                        ax_rec.plot(future_dates, receipts, label=f"Recettes Proj {name}", linestyle='--')
                        
                    ax_rec.axvspan('2030-06-01', '2030-07-31', color='teal', alpha=0.15, label='Coupe du Monde')
                    plt.title("Projection des Recettes Touristiques Globales (Effet Prix + Inflation)")
                    plt.legend()
                    st.pyplot(fig_rec)
