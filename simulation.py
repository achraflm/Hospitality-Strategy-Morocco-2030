"""
Application Web Interactive de Simulation de ROI Hôtelier - Maroc 2030 & Horizon 2035
=====================================================================================

Cette application Streamlit autonome est dédiée à la simulation financière et à l'analyse
de scénarios d'investissement hôtelier sur 10 ans pour les villes marocaines, pilotée
dynamiquement par les prévisions des top 3 modèles prédictifs étudiés.
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
from src.roi_simulator import HotelROISimulator

# Import des modèles
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

# Configuration de la page
st.set_page_config(
    page_title="Morocco Hotel ROI Simulator - Horizon 2035",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé pour faire un look premium (Teal & Gold)
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    h1, h2, h3, h4 {
        color: #0f172a;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
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
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #0f766e;
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(13, 148, 136, 0.3);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-top: 4px solid #d97706; /* Gold/Amber */
        margin-bottom: 15px;
    }
    .metric-card.teal {
        border-top: 4px solid #0d9488; /* Teal */
    }
    .metric-card h4 {
        margin: 0 0 8px 0;
        color: #475569;
        font-size: 14px;
        font-weight: 600;
    }
    .metric-card p.val {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
    }
    .city-badge {
        background-color: #e2e8f0;
        color: #334155;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
        margin-top: 5px;
    }
    .city-badge.active {
        background-color: #ccfbf1;
        color: #0f766e;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate projections dynamically for any model
def generate_projections(model_name, df_ml, future_dates, selected_features, X_train, y_train, dl_epochs=10):
    """
    Fits the specified model on the full historical dataset and generates recursive projections.
    """
    from main import forecast_recursive_ml, forecast_recursive_dl
    
    # 1. Statistical models
    if model_name == 'SARIMA':
        model = SarimaModel().fit(df_ml[TARGET_COL])
        preds = model.predict(steps=len(future_dates))
        return np.clip(preds, 0, None)
        
    elif model_name == 'Prophet':
        exog_cols = [c for c in selected_features if c in df_ml.columns and c not in [TARGET_COL, 'Date']]
        model = ProphetModel().fit(df_ml, target_col=TARGET_COL, exog_cols=exog_cols)
        
        future_df = pd.DataFrame({'Date': future_dates})
        for col in exog_cols:
            # Forward fill the last historical value for future regressors
            future_df[col] = df_ml[col].iloc[-1]
            
        preds = model.predict(future_df)
        return np.clip(preds, 0, None)
        
    # 2. Deep Learning models
    elif model_name in ['LSTM', 'SimpleRNN']:
        dl_class_map = {
            'LSTM': LstmModel,
            'SimpleRNN': RnnModel
        }
        model = dl_class_map[model_name](epochs=dl_epochs).fit(X_train, y_train)
        preds = forecast_recursive_dl(model, df_ml, future_dates, selected_features)
        return np.clip(preds, 0, None)
        
    # 3. Machine Learning models
    else:
        ml_class_map = {
            'Ridge': RidgeModel, 'Random Forest': RandomForestModel, 'Extra Trees': ExtraTreesModel,
            'Gradient Boosting': GradientBoostingModel, 'AdaBoost': AdaBoostModel, 'XGBoost': XgboostModel,
            'LightGBM': LightgbmModel, 'CatBoost': CatboostModel, 'SVR': SvrModel
        }
        if model_name in ml_class_map:
            model = ml_class_map[model_name]().fit(X_train, y_train)
            preds = forecast_recursive_ml(model, df_ml, future_dates, selected_features)
            return np.clip(preds, 0, None)
            
    return None

# Chargement du dataset initial
@st.cache_data
def get_clean_tourism_data():
    df = loader.load_and_merge_tourism_data()
    df = cleaner.integrate_covid_data(df)
    df = cleaner.reconstruct_historical_arrivals(df)
    df = cleaner.reconstruct_historical_receipts(df)
    return df

df_clean = get_clean_tourism_data()

# ----------------------------------------------------
# CONFIGURATION SIDEBAR
# ----------------------------------------------------
st.sidebar.header("🛠️ Paramètres Généraux du Projet")

# Choix de la ville et chargement des constantes
city_defaults = {
    'Marrakech': {'capex': 150.0, 'adr': 250.0, 'part': 0.35, 'rec': 'Investir (Prioritaire, forte demande)'},
    'Casablanca': {'capex': 180.0, 'adr': 230.0, 'part': 0.20, 'rec': "Investir (Tourisme d'affaires)"},
    'Agadir': {'capex': 130.0, 'adr': 165.0, 'part': 0.18, 'rec': "À étudier (Saisonnier balnéaire)"},
    'Tanger': {'capex': 145.0, 'adr': 155.0, 'part': 0.10, 'rec': "Attendre (En développement rapide)"},
    'Rabat': {'capex': 165.0, 'adr': 175.0, 'part': 0.09, 'rec': "Attendre (Administratif haut de gamme)"},
    'Fès': {'capex': 120.0, 'adr': 135.0, 'part': 0.08, 'rec': "Éviter (Besoin d'infrastructures)"}
}

selected_city = st.sidebar.selectbox("🌆 Ville Cible", list(city_defaults.keys()))
defaults = city_defaults[selected_city]

# Configuration financière
st.sidebar.subheader("💰 Hypothèses Financières")
capex = st.sidebar.number_input("Investissement (MUSD)", min_value=10.0, max_value=500.0, value=defaults['capex'], step=5.0)
base_adr = st.sidebar.number_input("Tarif Initial ADR (USD)", min_value=50.0, max_value=1000.0, value=defaults['adr'], step=10.0)
rooms = st.sidebar.number_input("Nombre de Chambres", min_value=10, max_value=1000, value=200, step=10)

base_occupancy = st.sidebar.slider("Taux d'Occupation de Base (%)", 30.0, 95.0, 65.0, 1.0) / 100
discount_rate = st.sidebar.slider("Taux d'Actualisation WACC (%)", 2.0, 20.0, 8.0, 0.5) / 100
opex_margin = st.sidebar.slider("Marge Opérationnelle OpEx (%)", 40.0, 90.0, 65.0, 1.0) / 100

# Paramètres Coupe du Monde
st.sidebar.subheader("⚽ Coupe du Monde FIFA 2030")
enable_wc_boost = st.sidebar.checkbox("Activer les impacts 2030", value=True)
wc_adr_boost = st.sidebar.slider("Boost ADR 2030 (%)", 0, 100, 40, 5) / 100
inflation_rate = st.sidebar.slider("Taux d'inflation annuel moyen (%)", 0.0, 10.0, 2.5, 0.1) / 100

# Paramètres modélisation
st.sidebar.subheader("🤖 Modélisation & Features")
dl_epochs = st.sidebar.slider("Époques Deep Learning", 1, 50, 10, 5)

default_features = feat.get_feature_list()
selected_features = st.sidebar.multiselect(
    "Variables d'entrée (Features)",
    options=default_features,
    default=[
        'lags_1', 'lags_2', 'lags_12',
        'roll_mean_3', 'roll_mean_12',
        'month_sin', 'month_cos', 'year',
        'is_summer', 'is_high_season', 'cdm_event', 'is_covid',
        'anomaly_iforest', 'anomaly_prophet'
    ]
)

# ----------------------------------------------------
# CORPS DE LA PAGE
# ----------------------------------------------------
st.title("🇲🇦 Simulateur Interactif de ROI Hôtelier - Horizon 2035")
st.markdown("""
Ce tableau de bord permet de simuler la profitabilité d'un investissement hôtelier sur 10 ans (**2026-2035**) 
en s'appuyant dynamiquement sur les prévisions des **top 3 modèles de prédiction** les plus performants.
""")

# Affichage des métriques de la ville
col_badge1, col_badge2, col_badge3 = st.columns(3)
with col_badge1:
    st.markdown(f'<div class="metric-card teal"><h4>Recommandation de Référence</h4><p class="val" style="color:#0d9488; font-size:18px;">{defaults["rec"]}</p></div>', unsafe_allow_html=True)
with col_badge2:
    st.markdown(f'<div class="metric-card"><h4>Part des Nuitées Nationales</h4><p class="val" style="color:#d97706; font-size:18px;">{defaults["part"]*100:.1f} %</p></div>', unsafe_allow_html=True)
with col_badge3:
    st.markdown(f'<div class="metric-card teal"><h4>Capitaux d\'Investissement Estimés</h4><p class="val" style="color:#0f172a; font-size:18px;">{capex:.1f} Millions USD</p></div>', unsafe_allow_html=True)

st.markdown("---")

# 1. Identification automatique des Top 3 Modèles
model_name_mapping = {
    'SARIMA': 'SARIMA',
    'SARIMAX': 'SARIMA',
    'Prophet': 'Prophet',
    'Ridge': 'Ridge',
    'Random Forest': 'Random Forest',
    'Extra Trees': 'Extra Trees',
    'Gradient Boosting': 'Gradient Boosting',
    'AdaBoost': 'AdaBoost',
    'XGBoost': 'XGBoost',
    'LightGBM': 'LightGBM',
    'CatBoost': 'CatBoost',
    'SVR': 'SVR',
    'LSTM': 'LSTM',
    'Best DL (LSTM)': 'LSTM',
    'SimpleRNN': 'SimpleRNN'
}

metrics_path = os.path.join("data", "model_performance_metrics.csv")
top_3_models = []

if os.path.exists(metrics_path):
    try:
        metrics_df = pd.read_csv(metrics_path)
        metrics_df['Mapped_Model'] = metrics_df['Model'].map(model_name_mapping)
        valid_df = metrics_df.dropna(subset=['Mapped_Model'])
        # Trier par R2 décroissant
        valid_df = valid_df.sort_values(by='R2', ascending=False)
        top_3_models = valid_df['Mapped_Model'].unique().tolist()
    except Exception as e:
        st.warning(f"Impossible de lire le fichier de métriques : {e}")

# Compléter avec des fallbacks si nécessaire
fallback_list = ['Ridge', 'CatBoost', 'Random Forest', 'XGBoost', 'LightGBM']
for fallback in fallback_list:
    if len(top_3_models) >= 3:
        break
    if fallback not in top_3_models:
        top_3_models.append(fallback)
top_3_models = top_3_models[:3]

st.info(f"💡 **Top 3 modèles d'apprentissage les plus performants retenus pour l'étude** : {', '.join([f'`{m}`' for m in top_3_models])}")

# 2. Bouton d'action pour la simulation
sim_btn = st.button("🚀 Lancer la Simulation Comparée")

if sim_btn:
    if not selected_features:
        st.error("Veuillez sélectionner au moins une feature dans la barre latérale pour la modélisation.")
    else:
        with st.spinner("Génération des prévisions 2026-2035 et calcul des cash flows..."):
            # Charger les données réelles de 2025 pour la croissance relative
            df_2025 = df_clean[pd.to_datetime(df_clean['Date']).dt.year == 2025]
            arrivals_2025 = df_2025['Arrivals'].sum()
            
            # Définir l'horizon de 10 ans : mai 2026 à décembre 2035
            future_dates = pd.date_range(start='2026-05-01', end='2035-12-01', freq='MS')
            
            # Préparer X_train, y_train
            df_featured = feat.build_features(df_clean)
            df_ml = df_featured.dropna(subset=[TARGET_COL]).copy()
            X_train = df_ml[selected_features].fillna(df_ml[selected_features].median())
            y_train = df_ml[TARGET_COL]
            
            # Dictionnaires de stockage
            model_sim_dfs = {}
            model_metrics = {}
            
            # Instancier le simulateur
            sim = HotelROISimulator(
                rooms=rooms,
                investment_usd=capex * 1e6,
                opex_margin=opex_margin,
                discount_rate=discount_rate,
                base_occupancy=base_occupancy,
                wc_adr_boost_pct=wc_adr_boost,
                inflation_rate=inflation_rate
            )
            
            # Extraire le cumul réel pour les mois Jan-Avr 2026
            df_2026_real = df_clean[pd.to_datetime(df_clean['Date']).dt.year == 2026]
            real_arrivals_2026 = df_2026_real['Arrivals'].sum()
            
            # Loop sur le top 3 modèles
            for model_name in top_3_models:
                preds = generate_projections(
                    model_name=model_name,
                    df_ml=df_ml,
                    future_dates=future_dates,
                    selected_features=selected_features,
                    X_train=X_train,
                    y_train=y_train,
                    dl_epochs=dl_epochs
                )
                
                if preds is not None:
                    # Agréger par année civile
                    df_temp = pd.DataFrame({'Date': future_dates, 'Arrivals': preds})
                    
                    annual_arrivals = {}
                    for y in range(2026, 2036):
                        if y == 2026:
                            pred_2026 = df_temp[df_temp['Date'].dt.year == 2026]['Arrivals'].sum()
                            annual_arrivals[2026] = real_arrivals_2026 + pred_2026
                        else:
                            annual_arrivals[y] = df_temp[df_temp['Date'].dt.year == y]['Arrivals'].sum()
                            
                    # Simuler cash flows
                    df_sim = sim.simulate_with_forecast(
                        annual_arrivals_dict=annual_arrivals,
                        baseline_arrivals_2025=arrivals_2025,
                        start_year=2026,
                        wc_boost_2030=enable_wc_boost
                    )
                    
                    # Calculer indicateurs
                    metrics = sim.calculate_metrics_for_gop(df_sim['GOP_USD'])
                    
                    model_sim_dfs[model_name] = df_sim
                    model_metrics[model_name] = metrics
            
            # Sauvegarder dans la session Streamlit
            st.session_state['sim_model_sim_dfs'] = model_sim_dfs
            st.session_state['sim_model_metrics'] = model_metrics
            st.session_state['sim_selected_city'] = selected_city
            st.session_state['sim_capex'] = capex
            st.session_state['sim_adr'] = base_adr
            st.session_state['sim_rooms'] = rooms

# Affichage des résultats si calculés
if 'sim_model_metrics' in st.session_state and st.session_state['sim_model_metrics']:
    roi_metrics = st.session_state['sim_model_metrics']
    roi_sim_dfs = st.session_state['sim_model_sim_dfs']
    city_name = st.session_state['sim_selected_city']
    cap = st.session_state['sim_capex']
    adr = st.session_state['sim_adr']
    rms = st.session_state['sim_rooms']
    
    st.subheader(f"📊 Bilan des Indicateurs ROI Hôtelier - {city_name}")
    st.markdown(f"Hôtel de `{rms}` chambres — Coût du foncier et construction : `{cap}` M$ USD")
    
    # Table comparative stylée
    summary_records = []
    for name, m in roi_metrics.items():
        summary_records.append({
            'Modèle': name,
            'Net Present Value (NPV)': f"{m['NPV_MUSD']:.2f} Millions USD",
            'Internal Rate of Return (IRR)': f"{m['IRR_Pct']:.2f} %" if not np.isnan(m['IRR_Pct']) else "N/A",
            'Retour sur Inv. (Payback)': f"{m['Payback_Years']} ans" if not np.isnan(m['Payback_Years']) else ">10 ans",
            'ROI Cumulé sur 10 ans (%)': f"{m['ROI_Pct']:.2f} %"
        })
    
    summary_df = pd.DataFrame(summary_records)
    # Styler R2 et ROI
    styled_summary = summary_df.style.highlight_max(
        axis=0,
        subset=['ROI Cumulé sur 10 ans (%)'],
        props='background-color: #115e59; color: #ffffff; font-weight: bold;'
    )
    st.dataframe(styled_summary, use_container_width=True)
    
    # Graphique cumulé
    st.subheader("📈 Courbes Comparatives de Profit Cumulé (2025-2035)")
    
    fig, ax = plt.subplots(figsize=(12, 5.5))
    years = [2025] + list(range(2026, 2036))
    colors_list = ['#0d9488', '#d97706', '#2563eb']
    markers_list = ['o', 's', '^']
    
    for idx, (name, df_sim) in enumerate(roi_sim_dfs.items()):
        cf = [-cap] + (df_sim['GOP_USD'] / 1e6).tolist()
        cum_cf = np.cumsum(cf)
        
        lbl = f"{name} (NPV={roi_metrics[name]['NPV_MUSD']:.1f}M$, IRR={roi_metrics[name]['IRR_Pct']:.1f}%)"
        ax.plot(years, cum_cf, marker=markers_list[idx % 3], color=colors_list[idx % 3], label=lbl, linewidth=2.5)
        
    ax.axhline(0, color='black', linestyle=':', alpha=0.7)
    ax.set_title(f"Évolution Comparée des Profits Cumulés — Hôtel {city_name}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Année de projet")
    ax.set_ylabel("Profit Cumulé (Millions USD)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle='--', alpha=0.4)
    
    st.pyplot(fig)
    
    # Recommandation d'expert dynamique
    best_model_roi = max(roi_metrics.keys(), key=lambda x: roi_metrics[x]['ROI_Pct'])
    max_roi = roi_metrics[best_model_roi]['ROI_Pct']
    irr_val = roi_metrics[best_model_roi]['IRR_Pct']
    payback_val = roi_metrics[best_model_roi]['Payback_Years']
    
    st.subheader("📝 Note d'Analyse et Conseils Stratégiques")
    
    decision = "FAVORABLE" if max_roi >= 80.0 else "À ÉTUDIER" if max_roi >= 40.0 else "DEFAVORABLE"
    decision_color = "#059669" if decision == "FAVORABLE" else "#d97706" if decision == "À ÉTUDIER" else "#dc2626"
    
    st.markdown(f"""
    <div style="background-color: #f8fafc; padding: 22px; border-radius: 10px; border-left: 6px solid {decision_color}; margin-top: 15px;">
        <h4 style="color: {decision_color}; margin-top: 0; font-size:18px;">Décision Financière Préconisée : {decision}</h4>
        <p>En couplant les volumes d'arrivées touristiques modélisés par <b>{best_model_roi}</b> (scénario le plus optimiste) avec les spécificités de la ville de <b>{city_name}</b>, le projet présente :</p>
        <ul>
            <li>Un <b>Taux de Rentabilité Interne (IRR) de {irr_val:.2f} %</b> (comparé au taux WACC de d'actualisation de {(discount_rate*100):.1f}%).</li>
            <li>Une valeur actualisée nette (NPV) positive estimée à <b>{roi_metrics[best_model_roi]['NPV_MUSD']:.2f} M$ USD</b>.</li>
            <li>Un délai de récupération sur capital investi de <b>{payback_val} ans</b>.</li>
        </ul>
        <p style="margin-bottom:0;"><i>Note sectorielle :</i> La ville de {city_name} capte environ <b>{(defaults['part']*100):.0f}%</b> du volume de nuitées nationales. 
        C'est un choix stratégique d'implantation qui présente une excellente résilience face aux cycles touristiques et bénéficie grandement de l'attractivité nationale pour l'événement Coupe du Monde 2030.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Veuillez configurer vos paramètres dans la barre latérale gauche, puis cliquez sur 'Lancer la Simulation Comparée' ci-dessus.")
