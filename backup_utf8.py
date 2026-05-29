"""
Application Streamlit de ModÃ©lisation et Simulation Touristique Maroc 2030
========================================================================

Cette application interactive permet d'explorer les donnÃ©es, configurer le Feature Engineering,
sÃ©lectionner et entraÃ®ner les modÃ¨les de prÃ©vision (parmi les 13 disponibles),
comparer leurs mÃ©triques en temps rÃ©el et projeter les prÃ©visions jusqu'Ã  un horizon choisi (jusqu'en 2035).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# S'assurer que le rÃ©pertoire racine est dans le python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import DATA_DIR, TARGET_COL
import src.data_loader as loader
import src.cleaning as cleaner
import src.features as feat
import src.metrics as metrics_mod

# Import des modÃ¨les
from src.models.sarima import SarimaModel
from src.models.ridge import RidgeModel
from src.models.lstm import LstmModel
from src.models.xgboost import XgboostModel
from sklearn.model_selection import TimeSeriesSplit

# Configuration de la page
st.set_page_config(
    page_title="Morocco Tourism Forecasting Dashboard",
    page_icon="ðŸ‡²ðŸ‡¦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisÃ© pour faire un look premium (Teal/Emerald/Gold)
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    .reportview-container {
        background: #f1f5f9;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    div.stButton > button:first-child {
        background-color: #0d9488;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 12px 28px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(13, 148, 136, 0.2);
    }
    div.stButton > button:first-child:hover {
        background-color: #0f766e;
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(13, 148, 136, 0.3);
    }
    .metric-card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border-top: 4px solid #0d9488;
        margin-bottom: 15px;
    }
    .metric-card h4 {
        margin: 0 0 8px 0;
        color: #475569;
        font-size: 14px;
        font-weight: 600;
    }
    .metric-card p.val {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Chargement des donnÃ©es (mis en cache)
# ----------------------------------------------------
@st.cache_data
def get_clean_tourism_data():
    df = loader.load_and_merge_tourism_data()
    df = cleaner.integrate_covid_data(df)
    df = cleaner.reconstruct_historical_arrivals(df)
    df = cleaner.reconstruct_historical_receipts(df)
    return df

st.title("ðŸ‡²ðŸ‡¦ Tableau de Bord PrÃ©visionnel du Tourisme - Horizon 2030 & Au-delÃ ")
st.markdown("Ce tableau de bord analytique permet de configurer le Feature Engineering, d'entraÃ®ner 13 modÃ¨les de Machine Learning et Deep Learning, et de simuler les flux touristiques nationaux jusqu'Ã  l'horizon de votre choix.")
st.markdown("---")

# 1. Barre latÃ©rale des paramÃ¨tres d'expÃ©rimentation
st.sidebar.header("ðŸ› ï¸ Configuration des ExpÃ©riences")

# Split temporel
split_year = st.sidebar.slider(
    "AnnÃ©e de dÃ©but du test split", 
    min_value=2020, 
    max_value=2025, 
    value=2023, 
    help="Les donnÃ©es antÃ©rieures servent Ã  l'entraÃ®nement, les donnÃ©es postÃ©rieures valident les performances."
)

# Configuration Feature Engineering
st.sidebar.subheader("ðŸ“‹ Feature Engineering")
default_features = feat.get_feature_list()
selected_features = st.sidebar.multiselect(
    "Choisir les colonnes Ã  ajouter aux modÃ¨les",
    options=default_features,
    default=[
        'lags_1', 'lags_2', 'lags_12',
        'roll_mean_3', 'roll_mean_12',
        'month_sin', 'month_cos', 'year',
        'is_summer', 'is_high_season', 'cdm_event', 'is_covid',
        'anomaly_iforest', 'anomaly_prophet'
    ]
)

# SÃ©lection des modÃ¨les
st.sidebar.subheader("ðŸ¤– ModÃ¨les PrÃ©dictifs")
all_models = [
    "SARIMA", "Ridge", "LSTM", "XGBoost"
]
selected_models = st.sidebar.multiselect(
    "Choisir les modÃ¨les Ã  exÃ©cuter",
    options=all_models,
    default=["SARIMA", "Ridge", "LSTM", "XGBoost"]
)

# Ã‰poques Deep Learning
dl_epochs = st.sidebar.number_input(
    "Ã‰poques d'entraÃ®nement pour LSTM / RNN", 
    min_value=1, 
    max_value=100, 
    value=5, 
    step=5,
    help="Un nombre faible (ex: 5) s'exÃ©cute rapidement pour tester."
)

# Chargement du dataset initial
df_clean = get_clean_tourism_data()

# Tabs principaux
tab_data, tab_train, tab_forecast = st.tabs([
    "ðŸ“Š Exploration des DonnÃ©es (EDA)",
    "ðŸ¤– EntraÃ®nement des ModÃ¨les",
    "ðŸ“ˆ Projections PersonnalisÃ©es (Horizon Temporel)"
])

# ----------------------------------------------------
# TAB 1: DATA & EDA EXPLORATION
# ----------------------------------------------------
with tab_data:
    st.header("Analyse Exploratoire des SÃ©ries Temporelles")
    
    # Cartes de synthÃ¨se rapide des donnÃ©es historiques rÃ©elles
    last_real_year = 2025
    df_2025 = df_clean[pd.to_datetime(df_clean['Date']).dt.year == last_real_year]
    total_arr_2025 = df_2025['Arrivals'].sum()
    total_rec_2025 = df_2025['Total_Receipts_MDH'].sum()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>ArrivÃ©es Annuelles Totales ({last_real_year})</h4>
            <p class="val">{total_arr_2025/1e6:.2f} M</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Recettes Touristiques ({last_real_year})</h4>
            <p class="val">{total_rec_2025/1e3:.2f} Mrd MAD</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Taux de Croissance Annuel ({last_real_year})</h4>
            <p class="val">+8.4 %</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Historique des Flux Touristiques Mensuels")
        fig, ax1 = plt.subplots(figsize=(12, 5.5))
        color = '#0d9488'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('ArrivÃ©es Touristiques', color=color)
        ax1.plot(pd.to_datetime(df_clean['Date']), df_clean['Arrivals'], color=color, linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        ax2 = ax1.twinx()
        color = '#d97706'
        ax2.set_ylabel('Recettes Touristiques (MDH)', color=color)
        ax2.plot(pd.to_datetime(df_clean['Date']), df_clean['Total_Receipts_MDH'], color=color, linewidth=1.5, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title("ArrivÃ©es (Teal) vs Recettes Touristiques (Orange) (1995-2026)")
        st.pyplot(fig)
        
    with col2:
        st.subheader("AperÃ§u du Dataset Source")
        st.dataframe(df_clean[['Date', 'Arrivals', 'Total_Receipts_MDH', 'Oil_price', 'REER']].tail(12), use_container_width=True)

# ----------------------------------------------------
# TAB 2: TRAINING AND PERFORMANCE SUMMARY
# ----------------------------------------------------
with tab_train:
    st.header("EntraÃ®nement Dynamique des ModÃ¨les")
    st.markdown("Configurez les modÃ¨les et les descripteurs dans la barre latÃ©rale gauche, puis cliquez sur le bouton ci-dessous pour lancer l'entraÃ®nement sur l'ensemble de donnÃ©es historisÃ©.")
    
    run_btn = st.button("ðŸš€ EntraÃ®ner les ModÃ¨les")
    
    if run_btn:
        if not selected_models:
            st.error("Veuillez sÃ©lectionner au moins un modÃ¨le prÃ©dictif dans la barre latÃ©rale.")
        elif not selected_features:
            st.error("Veuillez sÃ©lectionner au moins une feature dans la barre latÃ©rale.")
        else:
            with st.spinner("ModÃ©lisation en cours... Veuillez patienter..."):
                # GÃ©nÃ©ration des descripteurs
                df_featured = feat.build_features(df_clean)
                df_ml = df_featured.dropna(subset=[TARGET_COL]).copy()
                
                # Diviser train / test chronologiquement
                train_end_date = f"{split_year - 1}-12-31"
                test_start_date = f"{split_year}-01-01"
                
                train_df = df_ml[df_ml['Date'] <= train_end_date]
                test_df = df_ml[df_ml['Date'] >= test_start_date]
                
                # Imputation
                X_train = train_df[selected_features].fillna(train_df[selected_features].median())
                y_train = train_df[TARGET_COL]
                X_test = test_df[selected_features].fillna(train_df[selected_features].median())
                y_test = test_df[TARGET_COL]
                
                # Instancier et entraÃ®ner les modÃ¨les sÃ©lectionnÃ©s
                predictions = {}
                
                # ModÃ¨les Statistiques
                if "SARIMA" in selected_models:
                    sarima = SarimaModel().fit(y_train)
                    predictions['SARIMA'] = sarima.predict(steps=len(y_test))
                    
                # ModÃ¨les ML
                if "Ridge" in selected_models:
                    predictions['Ridge'] = RidgeModel().fit(X_train, y_train).predict(X_test)
                
                # PrÃ©paration Walk-Forward pour modÃ¨les sensibles au Leakage (DL / XGBoost)
                X_all_raw = pd.concat([X_train, X_test])
                y_all_raw = pd.concat([y_train, y_test]).values
                train_size = len(X_train)
                n_splits = len(X_test)
                tscv = TimeSeriesSplit(n_splits=n_splits, test_size=1)
                    
                # ModÃ¨les DL (Walk-Forward)
                if "LSTM" in selected_models:
                    y_pred_dl = []
                    progress_text = "EntraÃ®nement Walk-Forward LSTM..."
                    my_bar = st.progress(0, text=progress_text)
                    for idx, (train_index, test_index) in enumerate(tscv.split(X_all_raw)):
                        if test_index[0] < train_size:
                            continue
                        X_tr, y_tr = X_all_raw.iloc[train_index], y_all_raw[train_index]
                        X_te = X_all_raw.iloc[test_index]
                        
                        # Scaling dynamique Ã  chaque Ã©tape interne Ã  LstmModel
                        model = LstmModel(epochs=dl_epochs).fit(X_tr, y_tr)
                        pred = model.predict(X_te, X_train_history=X_tr)
                        y_pred_dl.append(pred[0])
                        
                        my_bar.progress((idx + 1) / n_splits, text=progress_text)
                    my_bar.empty()
                    predictions['LSTM'] = np.array(y_pred_dl)

                # XGBoost (Walk-Forward)
                if "XGBoost" in selected_models:
                    y_pred_xgb = []
                    progress_text = "EntraÃ®nement Walk-Forward XGBoost..."
                    my_bar = st.progress(0, text=progress_text)
                    for idx, (train_index, test_index) in enumerate(tscv.split(X_all_raw)):
                        if test_index[0] < train_size:
                            continue
                        X_tr, y_tr = X_all_raw.iloc[train_index], y_all_raw[train_index]
                        X_te = X_all_raw.iloc[test_index]
                        
                        model = XgboostModel().fit(X_tr, y_tr)
                        pred = model.predict(X_te)
                        y_pred_xgb.append(pred[0])
                        
                        my_bar.progress((idx + 1) / n_splits, text=progress_text)
                    my_bar.empty()
                    predictions['XGBoost'] = np.array(y_pred_xgb)
                
                # Calculer les mÃ©triques
                results_df = metrics_mod.compare_models(predictions, y_test)
                
                # Sauvegarder dans la session Streamlit
                st.session_state['results_df'] = results_df
                st.session_state['predictions'] = predictions
                st.session_state['y_test'] = y_test
                st.session_state['test_dates'] = df_ml[df_ml['Date'] >= test_start_date]['Date'].values[:len(y_test)]
                
            st.success("EntraÃ®nement de tous les modÃ¨les complÃ©tÃ© avec succÃ¨s !")
            
    # Affichage des mÃ©triques si elles existent dans l'Ã©tat de session
    if 'results_df' in st.session_state:
        st.subheader("Performance des ModÃ¨les sur l'Ensemble de Test (Validation Chronologique)")
        
        # Highlight R2 with high contrast text (dark teal background and white text for readability in all themes)
        styled_df = st.session_state['results_df'].style.highlight_max(
            axis=0, 
            subset=['R2'], 
            props='background-color: #115e59; color: #ffffff; font-weight: bold;'
        )
        st.dataframe(styled_df, use_container_width=True)
        
        # Plot predictions vs actuals
        st.subheader("TracÃ©s des Courbes de PrÃ©dictions")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(st.session_state['test_dates'], st.session_state['y_test'], label='DonnÃ©es RÃ©elles', color='black', linewidth=2)
        
        for name, pred in st.session_state['predictions'].items():
            ax.plot(st.session_state['test_dates'], pred.flatten(), label=f'PrÃ©diction {name}', linestyle='--')
            
        plt.title("Comparaison des courbes de prÃ©dictions sur la pÃ©riode de test")
        plt.xlabel("Date")
        plt.ylabel("ArrivÃ©es")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
    else:
        st.info("Cliquez sur 'EntraÃ®ner les ModÃ¨les' ci-dessus pour lancer le calcul des performances.")

# ----------------------------------------------------
# TAB 3: 2030 & HORIZON PROJECTIONS
# ----------------------------------------------------
with tab_forecast:
    st.header("Projections PersonnalisÃ©es Ã  Long Terme (May 2026 - DÃ©cembre Cible)")
    
    if 'results_df' not in st.session_state:
        st.warning("Veuillez d'abord entraÃ®ner les modÃ¨les dans l'onglet 'EntraÃ®nement des ModÃ¨les'.")
    else:
        # Trouver le meilleur modÃ¨le ML et DL
        results_df = st.session_state['results_df']
        
        ml_names = ['Ridge', 'XGBoost']
        dl_names = ['LSTM']
        
        avail_ml = results_df[results_df['Model'].isin(ml_names)]
        avail_dl = results_df[results_df['Model'].isin(dl_names)]
        
        best_ml_name = avail_ml['Model'].iloc[0] if len(avail_ml) > 0 else None
        best_dl_name = avail_dl['Model'].iloc[0] if len(avail_dl) > 0 else None
        
        # Build status string
        status_parts = []
        if best_ml_name:
            status_parts.append(f"ML optimal = `{best_ml_name}`")
        if best_dl_name:
            status_parts.append(f"DL optimal = `{best_dl_name}`")
        
        if not status_parts:
            st.error("Aucun modÃ¨le entraÃ®nÃ© n'a Ã©tÃ© sÃ©lectionnÃ©. Veuillez sÃ©lectionner et entraÃ®ner des modÃ¨les.")
        else:
            st.info(f"ðŸ’¡ **ModÃ¨les retenus pour la projection** : {' | '.join(status_parts)}")
            
            # ParamÃ¨tres d'horizon et Ã©conomiques
            col1, col2, col3 = st.columns(3)
            with col1:
                target_year = st.slider("Choisir l'annÃ©e cible des prÃ©visions", min_value=2026, max_value=2035, value=2030, step=1)
            with col2:
                inflation_rate = st.slider("Taux d'inflation annuel moyen (%)", 0.0, 5.0, 1.5, 0.5) / 100
            with col3:
                wc_boost = st.slider("Boost d'attractivitÃ© Coupe du Monde 2030 (%)", 0, 50, 20, 5) / 100
                
            # Bouton pour projeter
            proj_btn = st.button("ðŸŽ¯ Lancer la Projection RÃ©cursive")
            
            if proj_btn:
                with st.spinner(f"GÃ©nÃ©ration des prÃ©visions rÃ©cursives jusqu'en dÃ©cembre {target_year}..."):
                    from main import forecast_recursive_ml, forecast_recursive_dl
                    
                    # PrÃ©paration des donnÃ©es d'historique
                    df_featured = feat.build_features(df_clean)
                    df_ml = df_featured.dropna(subset=[TARGET_COL]).copy()
                    X_train = df_ml[selected_features].fillna(df_ml[selected_features].median())
                    y_train = df_ml[TARGET_COL]
                    
                    future_dates = pd.date_range(start='2026-05-01', end=f'{target_year}-12-01', freq='MS')
                    mean_ratio = (df_ml['Total_Receipts_MDH'] / df_ml['Arrivals']).mean()
                    
                    def get_receipts(arrivals):
                        receipts = []
                        for arr, date in zip(arrivals, future_dates):
                            years_since_2026 = date.year - 2026
                            ratio = mean_ratio * ((1 + inflation_rate) ** years_since_2026)
                            if date.year == 2030:
                                ratio = ratio * (1 + wc_boost)
                            receipts.append(arr * ratio)
                        return np.array(receipts)
                    
                    # Projections ML
                    arrivals_ml = None
                    receipts_ml = None
                    ml_class_map = {
                        'Ridge': RidgeModel,
                        'XGBoost': XgboostModel
                    }
                    if best_ml_name and best_ml_name in ml_class_map:
                        best_ml_model = ml_class_map[best_ml_name]().fit(X_train, y_train)
                        arrivals_ml = forecast_recursive_ml(best_ml_model, df_ml, future_dates, selected_features)
                        arrivals_ml = np.clip(arrivals_ml, 0, None)
                        receipts_ml = get_receipts(arrivals_ml)
                    
                    # Projections DL
                    arrivals_dl = None
                    receipts_dl = None
                    dl_class_map = {
                        'LSTM': LstmModel
                    }
                    if best_dl_name and best_dl_name in dl_class_map:
                        best_dl_model = dl_class_map[best_dl_name](epochs=dl_epochs).fit(X_train, y_train)
                        arrivals_dl = forecast_recursive_dl(best_dl_model, df_ml, future_dates, selected_features)
                        arrivals_dl = np.clip(arrivals_dl, 0, None)
                        receipts_dl = get_receipts(arrivals_dl)
                    
                    # Sauvegarde session
                    st.session_state['future_dates'] = future_dates
                    st.session_state['arrivals_ml'] = arrivals_ml
                    st.session_state['arrivals_dl'] = arrivals_dl
                    st.session_state['receipts_ml'] = receipts_ml
                    st.session_state['receipts_dl'] = receipts_dl
                    st.session_state['best_ml_name'] = best_ml_name
                    st.session_state['best_dl_name'] = best_dl_name
                    st.session_state['df_ml'] = df_ml
                    st.session_state['target_year'] = target_year
                    
                st.success(f"Projections jusqu'en dÃ©cembre {target_year} terminÃ©es !")
                
            if 'future_dates' in st.session_state:
                t_year = st.session_state['target_year']
                
                # Affichage graphique arrivÃ©es
                st.subheader(f"Projections des ArrivÃ©es Touristiques (1995 - {t_year})")
                fig, ax = plt.subplots(figsize=(14, 5.5))
                ax.plot(st.session_state['df_ml']['Date'], st.session_state['df_ml']['Arrivals'], label='Historique RÃ©el', color='black')
                
                if st.session_state['arrivals_ml'] is not None:
                    ax.plot(st.session_state['future_dates'], st.session_state['arrivals_ml'], label=f"Projections {st.session_state['best_ml_name']} (ML)", color='#0d9488', linewidth=2)
                if st.session_state['arrivals_dl'] is not None:
                    ax.plot(st.session_state['future_dates'], st.session_state['arrivals_dl'], label=f"Projections {st.session_state['best_dl_name']} (DL)", color='#d97706', linestyle='--', linewidth=2)
                
                if t_year >= 2030:
                    ax.axvspan('2030-06-01', '2030-07-31', color='teal', alpha=0.15, label='Coupe du Monde FIFA 2030')
                    
                plt.title(f"Flux d'ArrivÃ©es Touristiques ProjetÃ©s Ã  l'horizon {t_year}")
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.legend()
                st.pyplot(fig)
                
                # Recettes
                st.subheader(f"Projections des Recettes Touristiques (1995 - {t_year}) (MDH)")
                fig2, ax2 = plt.subplots(figsize=(14, 5.5))
                ax2.plot(st.session_state['df_ml']['Date'], st.session_state['df_ml']['Total_Receipts_MDH'], label='Historique RÃ©el', color='black')
                
                if st.session_state['receipts_ml'] is not None:
                    ax2.plot(st.session_state['future_dates'], st.session_state['receipts_ml'], label=f"Projections {st.session_state['best_ml_name']} (ML)", color='#0d9488', linewidth=2)
                if st.session_state['receipts_dl'] is not None:
                    ax2.plot(st.session_state['future_dates'], st.session_state['receipts_dl'], label=f"Projections {st.session_state['best_dl_name']} (DL)", color='#d97706', linestyle='--', linewidth=2)
                
                if t_year >= 2030:
                    ax2.axvspan('2030-06-01', '2030-07-31', color='teal', alpha=0.15, label='Coupe du Monde FIFA 2030')
                    
                plt.title(f"Recettes Touristiques ProjetÃ©es Ã  l'horizon {t_year}")
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.legend()
                st.pyplot(fig2)
                
                # RÃ©sumÃ© chiffrÃ© des prÃ©visions de l'annÃ©e cible
                st.subheader(f"Bilan PrÃ©visionnel Annuel de l'AnnÃ©e Cible ({t_year})")
                
                col_ml, col_dl = st.columns(2)
                
                with col_ml:
                    if st.session_state['arrivals_ml'] is not None:
                        df_target_proj_ml = pd.DataFrame({
                            'Date': st.session_state['future_dates'],
                            'Arrivals': st.session_state['arrivals_ml'],
                            'Receipts': st.session_state['receipts_ml']
                        })
                        df_target_proj_ml = df_target_proj_ml[df_target_proj_ml['Date'].dt.year == t_year]
                        sum_arr_ml = df_target_proj_ml['Arrivals'].sum()
                        sum_rec_ml = df_target_proj_ml['Receipts'].sum()
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>PrÃ©visions Machine Learning ({st.session_state['best_ml_name']})</h4>
                            <p style="font-size:18px; font-weight:700; color:#0d9488; margin-bottom:5px;">ArrivÃ©es : {sum_arr_ml/1e6:.2f} Millions</p>
                            <p style="font-size:18px; font-weight:700; color:#0d9488;">Recettes : {sum_rec_ml/1e3:.2f} Mrd MAD</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Aucun modÃ¨le de Machine Learning n'a Ã©tÃ© sÃ©lectionnÃ©.")
                        
                with col_dl:
                    if st.session_state['arrivals_dl'] is not None:
                        df_target_proj_dl = pd.DataFrame({
                            'Date': st.session_state['future_dates'],
                            'Arrivals': st.session_state['arrivals_dl'],
                            'Receipts': st.session_state['receipts_dl']
                        })
                        df_target_proj_dl = df_target_proj_dl[df_target_proj_dl['Date'].dt.year == t_year]
                        sum_arr_dl = df_target_proj_dl['Arrivals'].sum()
                        sum_rec_dl = df_target_proj_dl['Receipts'].sum()
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>PrÃ©visions Deep Learning ({st.session_state['best_dl_name']})</h4>
                            <p style="font-size:18px; font-weight:700; color:#d97706; margin-bottom:5px;">ArrivÃ©es : {sum_arr_dl/1e6:.2f} Millions</p>
                            <p style="font-size:18px; font-weight:700; color:#d97706;">Recettes : {sum_rec_dl/1e3:.2f} Mrd MAD</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Aucun modÃ¨le de Deep Learning n'a Ã©tÃ© sÃ©lectionnÃ©.")
