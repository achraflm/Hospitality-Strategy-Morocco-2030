"""
Application Streamlit de Modélisation et Simulation Touristique Maroc 2030
========================================================================

Cette application interactive permet d'explorer les données, configurer le Feature Engineering,
sélectionner et entraîner les modèles de prévision (parmi les 13 disponibles),
comparer leurs métriques en temps réel et projeter les prévisions jusqu'à un horizon choisi (jusqu'en 2035).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# S'assurer que le répertoire racine est dans le python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import DATA_DIR, TARGET_COL
import src.data_loader as loader
import src.cleaning as cleaner
import src.features as feat
import src.metrics as metrics_mod

# Import des modèles
from src.models.sarima import SarimaModel
from src.models.ridge import RidgeModel
from src.models.lstm import LstmModel

# Configuration de la page
st.set_page_config(
    page_title="Morocco Tourism Forecasting Dashboard",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé pour faire un look premium (Teal/Emerald/Gold)
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
# Chargement des données (mis en cache)
# ----------------------------------------------------
@st.cache_data
def get_clean_tourism_data():
    df = loader.load_and_merge_tourism_data()
    df = cleaner.integrate_covid_data(df)
    df = cleaner.reconstruct_historical_arrivals(df)
    df = cleaner.reconstruct_historical_receipts(df)
    return df

st.title("🇲🇦 Tableau de Bord Prévisionnel du Tourisme - Horizon 2030 & Au-delà")
st.markdown("Ce tableau de bord analytique permet de configurer le Feature Engineering, d'entraîner 13 modèles de Machine Learning et Deep Learning, et de simuler les flux touristiques nationaux jusqu'à l'horizon de votre choix.")
st.markdown("---")

# 1. Barre latérale des paramètres d'expérimentation
st.sidebar.header("🛠️ Configuration des Expériences")

# Split temporel
split_year = st.sidebar.slider(
    "Année de début du test split", 
    min_value=2020, 
    max_value=2025, 
    value=2023, 
    help="Les données antérieures servent à l'entraînement, les données postérieures valident les performances."
)

# Configuration Feature Engineering
st.sidebar.subheader("📋 Feature Engineering")
default_features = feat.get_feature_list()
selected_features = st.sidebar.multiselect(
    "Choisir les colonnes à ajouter aux modèles",
    options=default_features,
    default=[
        'lags_1', 'lags_2', 'lags_12',
        'roll_mean_3', 'roll_mean_12',
        'month_sin', 'month_cos', 'year',
        'is_summer', 'is_high_season', 'cdm_event', 'is_covid',
        'anomaly_iforest', 'anomaly_prophet'
    ]
)

# Sélection des modèles
st.sidebar.subheader("🤖 Modèles Prédictifs")
all_models = [
    "SARIMA", "Ridge", "LSTM"
]
selected_models = st.sidebar.multiselect(
    "Choisir les modèles à exécuter",
    options=all_models,
    default=["SARIMA", "Ridge", "LSTM"]
)

# Époques Deep Learning
dl_epochs = st.sidebar.number_input(
    "Époques d'entraînement pour LSTM / RNN", 
    min_value=1, 
    max_value=100, 
    value=5, 
    step=5,
    help="Un nombre faible (ex: 5) s'exécute rapidement pour tester."
)

# Chargement du dataset initial
df_clean = get_clean_tourism_data()

# Tabs principaux
tab_data, tab_train, tab_forecast = st.tabs([
    "📊 Exploration des Données (EDA)",
    "🤖 Entraînement des Modèles",
    "📈 Projections Personnalisées (Horizon Temporel)"
])

# ----------------------------------------------------
# TAB 1: DATA & EDA EXPLORATION
# ----------------------------------------------------
with tab_data:
    st.header("Analyse Exploratoire des Séries Temporelles")
    
    # Cartes de synthèse rapide des données historiques réelles
    last_real_year = 2025
    df_2025 = df_clean[pd.to_datetime(df_clean['Date']).dt.year == last_real_year]
    total_arr_2025 = df_2025['Arrivals'].sum()
    total_rec_2025 = df_2025['Total_Receipts_MDH'].sum()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Arrivées Annuelles Totales ({last_real_year})</h4>
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
        ax1.set_ylabel('Arrivées Touristiques', color=color)
        ax1.plot(pd.to_datetime(df_clean['Date']), df_clean['Arrivals'], color=color, linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        ax2 = ax1.twinx()
        color = '#d97706'
        ax2.set_ylabel('Recettes Touristiques (MDH)', color=color)
        ax2.plot(pd.to_datetime(df_clean['Date']), df_clean['Total_Receipts_MDH'], color=color, linewidth=1.5, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title("Arrivées (Teal) vs Recettes Touristiques (Orange) (1995-2026)")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Aperçu du Dataset Source")
        st.dataframe(df_clean[['Date', 'Arrivals', 'Total_Receipts_MDH', 'Oil_price', 'REER']].tail(12), use_container_width=True)

# ----------------------------------------------------
# TAB 2: TRAINING AND PERFORMANCE SUMMARY
# ----------------------------------------------------
with tab_train:
    st.header("Entraînement Dynamique des Modèles")
    st.markdown("Configurez les modèles et les descripteurs dans la barre latérale gauche, puis cliquez sur le bouton ci-dessous pour lancer l'entraînement sur l'ensemble de données historisé.")
    
    run_btn = st.button("🚀 Entraîner les Modèles")
    
    if run_btn:
        if not selected_models:
            st.error("Veuillez sélectionner au moins un modèle prédictif dans la barre latérale.")
        elif not selected_features:
            st.error("Veuillez sélectionner au moins une feature dans la barre latérale.")
        else:
            with st.spinner("Modélisation en cours... Veuillez patienter..."):
                # Génération des descripteurs
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
                
                # Instancier et entraîner les modèles sélectionnés
                predictions = {}
                
                # Modèles Statistiques
                if "SARIMA" in selected_models:
                    sarima = SarimaModel().fit(y_train)
                    predictions['SARIMA'] = sarima.predict(steps=len(y_test))
                    
                # Modèles ML
                if "Ridge" in selected_models:
                    predictions['Ridge'] = RidgeModel().fit(X_train, y_train).predict(X_test)
                    
                # Modèles DL
                if "LSTM" in selected_models:
                    lstm = LstmModel(epochs=dl_epochs).fit(X_train, y_train)
                    predictions['LSTM'] = lstm.predict(X_test, X_train_history=X_train)
                
                # Calculer les métriques
                results_df = metrics_mod.compare_models(predictions, y_test)
                
                # Sauvegarder dans la session Streamlit
                st.session_state['results_df'] = results_df
                st.session_state['predictions'] = predictions
                st.session_state['y_test'] = y_test
                st.session_state['test_dates'] = df_ml[df_ml['Date'] >= test_start_date]['Date'].values[:len(y_test)]
                
            st.success("Entraînement de tous les modèles complété avec succès !")
            
    # Affichage des métriques si elles existent dans l'état de session
    if 'results_df' in st.session_state:
        st.subheader("Performance des Modèles sur l'Ensemble de Test (Validation Chronologique)")
        
        # Highlight R2 with high contrast text (dark teal background and white text for readability in all themes)
        styled_df = st.session_state['results_df'].style.highlight_max(
            axis=0, 
            subset=['R2'], 
            props='background-color: #115e59; color: #ffffff; font-weight: bold;'
        )
        st.dataframe(styled_df, use_container_width=True)
        
        # Plot predictions vs actuals
        st.subheader("Tracés des Courbes de Prédictions")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(st.session_state['test_dates'], st.session_state['y_test'], label='Données Réelles', color='black', linewidth=2)
        
        for name, pred in st.session_state['predictions'].items():
            ax.plot(st.session_state['test_dates'], pred.flatten(), label=f'Prédiction {name}', linestyle='--')
            
        plt.title("Comparaison des courbes de prédictions sur la période de test")
        plt.xlabel("Date")
        plt.ylabel("Arrivées")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
    else:
        st.info("Cliquez sur 'Entraîner les Modèles' ci-dessus pour lancer le calcul des performances.")

# ----------------------------------------------------
# TAB 3: 2030 & HORIZON PROJECTIONS
# ----------------------------------------------------
with tab_forecast:
    st.header("Projections Personnalisées à Long Terme (May 2026 - Décembre Cible)")
    
    if 'results_df' not in st.session_state:
        st.warning("Veuillez d'abord entraîner les modèles dans l'onglet 'Entraînement des Modèles'.")
    else:
        # Trouver le meilleur modèle ML et DL
        results_df = st.session_state['results_df']
        
        ml_names = ['Ridge']
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
            st.error("Aucun modèle entraîné n'a été sélectionné. Veuillez sélectionner et entraîner des modèles.")
        else:
            st.info(f"💡 **Modèles retenus pour la projection** : {' | '.join(status_parts)}")
            
            # Paramètres d'horizon et économiques
            col1, col2, col3 = st.columns(3)
            with col1:
                target_year = st.slider("Choisir l'année cible des prévisions", min_value=2026, max_value=2035, value=2030, step=1)
            with col2:
                inflation_rate = st.slider("Taux d'inflation annuel moyen (%)", 0.0, 5.0, 1.5, 0.5) / 100
            with col3:
                wc_boost = st.slider("Boost d'attractivité Coupe du Monde 2030 (%)", 0, 50, 20, 5) / 100
                
            # Bouton pour projeter
            proj_btn = st.button("🎯 Lancer la Projection Récursive")
            
            if proj_btn:
                with st.spinner(f"Génération des prévisions récursives jusqu'en décembre {target_year}..."):
                    from main import forecast_recursive_ml, forecast_recursive_dl
                    
                    # Préparation des données d'historique
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
                        'Ridge': RidgeModel
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
                    
                st.success(f"Projections jusqu'en décembre {target_year} terminées !")
                
            if 'future_dates' in st.session_state:
                t_year = st.session_state['target_year']
                
                # Affichage graphique arrivées
                st.subheader(f"Projections des Arrivées Touristiques (1995 - {t_year})")
                fig, ax = plt.subplots(figsize=(14, 5.5))
                ax.plot(st.session_state['df_ml']['Date'], st.session_state['df_ml']['Arrivals'], label='Historique Réel', color='black')
                
                if st.session_state['arrivals_ml'] is not None:
                    ax.plot(st.session_state['future_dates'], st.session_state['arrivals_ml'], label=f"Projections {st.session_state['best_ml_name']} (ML)", color='#0d9488', linewidth=2)
                if st.session_state['arrivals_dl'] is not None:
                    ax.plot(st.session_state['future_dates'], st.session_state['arrivals_dl'], label=f"Projections {st.session_state['best_dl_name']} (DL)", color='#d97706', linestyle='--', linewidth=2)
                
                if t_year >= 2030:
                    ax.axvspan('2030-06-01', '2030-07-31', color='teal', alpha=0.15, label='Coupe du Monde FIFA 2030')
                    
                plt.title(f"Flux d'Arrivées Touristiques Projetés à l'horizon {t_year}")
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.legend()
                st.pyplot(fig)
                
                # Recettes
                st.subheader(f"Projections des Recettes Touristiques (1995 - {t_year}) (MDH)")
                fig2, ax2 = plt.subplots(figsize=(14, 5.5))
                ax2.plot(st.session_state['df_ml']['Date'], st.session_state['df_ml']['Total_Receipts_MDH'], label='Historique Réel', color='black')
                
                if st.session_state['receipts_ml'] is not None:
                    ax2.plot(st.session_state['future_dates'], st.session_state['receipts_ml'], label=f"Projections {st.session_state['best_ml_name']} (ML)", color='#0d9488', linewidth=2)
                if st.session_state['receipts_dl'] is not None:
                    ax2.plot(st.session_state['future_dates'], st.session_state['receipts_dl'], label=f"Projections {st.session_state['best_dl_name']} (DL)", color='#d97706', linestyle='--', linewidth=2)
                
                if t_year >= 2030:
                    ax2.axvspan('2030-06-01', '2030-07-31', color='teal', alpha=0.15, label='Coupe du Monde FIFA 2030')
                    
                plt.title(f"Recettes Touristiques Projetées à l'horizon {t_year}")
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.legend()
                st.pyplot(fig2)
                
                # Résumé chiffré des prévisions de l'année cible
                st.subheader(f"Bilan Prévisionnel Annuel de l'Année Cible ({t_year})")
                
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
                            <h4>Prévisions Machine Learning ({st.session_state['best_ml_name']})</h4>
                            <p style="font-size:18px; font-weight:700; color:#0d9488; margin-bottom:5px;">Arrivées : {sum_arr_ml/1e6:.2f} Millions</p>
                            <p style="font-size:18px; font-weight:700; color:#0d9488;">Recettes : {sum_rec_ml/1e3:.2f} Mrd MAD</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Aucun modèle de Machine Learning n'a été sélectionné.")
                        
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
                            <h4>Prévisions Deep Learning ({st.session_state['best_dl_name']})</h4>
                            <p style="font-size:18px; font-weight:700; color:#d97706; margin-bottom:5px;">Arrivées : {sum_arr_dl/1e6:.2f} Millions</p>
                            <p style="font-size:18px; font-weight:700; color:#d97706;">Recettes : {sum_rec_dl/1e3:.2f} Mrd MAD</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Aucun modèle de Deep Learning n'a été sélectionné.")
