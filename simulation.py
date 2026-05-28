"""
Application Web Interactive de Simulation de ROI Hôtelier - Maroc 2030 & Horizon 2035
=====================================================================================

Cette application Streamlit autonome est dédiée à la simulation financière et à l'analyse
de scénarios d'investissement hôtelier sur 10 ans pour les villes marocaines, pilotée
dynamiquement par les prévisions des top 3 modèles prédictifs étudiés.

Deux cibles de prédiction sont disponibles :
  - **Arrivées touristiques** (Arrivals) — occupation déduite par ratio de croissance
  - **Nuitées** (Nights / overnight stays) — occupation déduite directement (Occ = Nights / Rooms×365)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# S'assurer que le répertoire racine est dans le python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import DATA_DIR, TARGET_COL, NIGHTS_COL
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
    .main { background-color: #f8fafc; }
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
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        border-top: 4px solid #d97706;
        margin-bottom: 15px;
    }
    .metric-card.teal   { border-top: 4px solid #0d9488; }
    .metric-card.purple { border-top: 4px solid #7c3aed; }
    .metric-card h4 { margin: 0 0 8px 0; color: #475569; font-size: 14px; font-weight: 600; }
    .metric-card p.val  { margin: 0; font-size: 26px; font-weight: 700; color: #0f172a; }
    .target-badge { display: inline-block; padding: 4px 14px; border-radius: 9999px;
                    font-size: 13px; font-weight: 700; margin-bottom: 8px; }
    .target-arrivals { background-color: #ccfbf1; color: #0f766e; }
    .target-nights   { background-color: #ede9fe; color: #6d28d9; }
    .city-badge { background-color: #e2e8f0; color: #334155; padding: 4px 10px;
                  border-radius: 9999px; font-size: 13px; font-weight: 600;
                  display: inline-block; margin-right: 8px; margin-top: 5px; }
    .city-badge.active { background-color: #ccfbf1; color: #0f766e; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper : générer des projections pour n'importe quel modèle
# ---------------------------------------------------------------------------
def generate_projections(model_name, target_col, df_ml, future_dates,
                          selected_features, X_train, y_train, dl_epochs=10):
    """
    Fits the specified model on the full historical dataset and generates
    recursive projections for the given target (Arrivals or Nights).
    """
    from main import forecast_recursive_ml, forecast_recursive_dl

    # 1. Statistical models
    if model_name == 'SARIMA':
        model = SarimaModel().fit(df_ml[target_col])
        preds = model.predict(steps=len(future_dates))
        return np.clip(preds, 0, None)

    elif model_name == 'Prophet':
        exog_cols = [c for c in selected_features
                     if c in df_ml.columns and c not in [target_col, 'Date']]
        model = ProphetModel().fit(df_ml, target_col=target_col, exog_cols=exog_cols)
        future_df = pd.DataFrame({'Date': future_dates})
        for col in exog_cols:
            future_df[col] = df_ml[col].iloc[-1]
        preds = model.predict(future_df)
        return np.clip(preds, 0, None)

    # 2. Deep Learning models
    elif model_name in ['LSTM', 'SimpleRNN']:
        dl_class_map = {'LSTM': LstmModel, 'SimpleRNN': RnnModel}
        model = dl_class_map[model_name](epochs=dl_epochs).fit(X_train, y_train)
        preds = forecast_recursive_dl(model, df_ml, future_dates, selected_features)
        return np.clip(preds, 0, None)

    # 3. Machine Learning models
    else:
        ml_class_map = {
            'Ridge': RidgeModel, 'Random Forest': RandomForestModel,
            'Extra Trees': ExtraTreesModel, 'Gradient Boosting': GradientBoostingModel,
            'AdaBoost': AdaBoostModel, 'XGBoost': XgboostModel,
            'LightGBM': LightgbmModel, 'CatBoost': CatboostModel, 'SVR': SvrModel
        }
        if model_name in ml_class_map:
            model = ml_class_map[model_name]().fit(X_train, y_train)
            preds = forecast_recursive_ml(model, df_ml, future_dates, selected_features)
            return np.clip(preds, 0, None)

    return None


# ---------------------------------------------------------------------------
# Chargement du dataset initial (mis en cache)
# ---------------------------------------------------------------------------
@st.cache_data
def get_clean_tourism_data():
    df = loader.load_and_merge_tourism_data()
    df = cleaner.integrate_covid_data(df)
    df = cleaner.reconstruct_historical_arrivals(df)
    df = cleaner.reconstruct_historical_receipts(df)
    return df

df_clean = get_clean_tourism_data()

# ---------------------------------------------------------------------------
# CONFIGURATION SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.header("🛠️ Paramètres Généraux du Projet")

# ---- Cible de prédiction ----
st.sidebar.subheader("🎯 Cible de Prédiction")
pred_target = st.sidebar.radio(
    "Variable à prédire",
    options=["Arrivées touristiques (Arrivals)", "Nuitées (Nights)"],
    help=(
        "**Arrivées** : prédit les entrées de touristes. L'occupation est déduite "
        "par le ratio de croissance par rapport à 2025.\n\n"
        "**Nuitées** : prédit les nuits passées. L'occupation est calculée directement "
        "(Occ = Nuitées / Chambres × 365) — plus précis pour le calcul du RevPAR."
    )
)
use_nights = pred_target.startswith("Nuitées")
active_target_col = NIGHTS_COL if use_nights else TARGET_COL

if use_nights:
    st.sidebar.markdown(
        '<span class="target-badge target-nights">🌙 Mode Nuitées actif</span>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<span class="target-badge target-arrivals">✈️ Mode Arrivées actif</span>',
        unsafe_allow_html=True
    )

# ---- Choix de la ville ----
city_defaults = {
    'Marrakech': {'capex': 150.0, 'adr': 250.0, 'part': 0.35,
                  'rec': 'Investir (Prioritaire, forte demande)'},
    'Casablanca': {'capex': 180.0, 'adr': 230.0, 'part': 0.20,
                   'rec': "Investir (Tourisme d'affaires)"},
    'Agadir':     {'capex': 130.0, 'adr': 165.0, 'part': 0.18,
                   'rec': "À étudier (Saisonnier balnéaire)"},
    'Tanger':     {'capex': 145.0, 'adr': 155.0, 'part': 0.10,
                   'rec': "Attendre (En développement rapide)"},
    'Rabat':      {'capex': 165.0, 'adr': 175.0, 'part': 0.09,
                   'rec': "Attendre (Administratif haut de gamme)"},
    'Fès':        {'capex': 120.0, 'adr': 135.0, 'part': 0.08,
                   'rec': "Éviter (Besoin d'infrastructures)"}
}

selected_city = st.sidebar.selectbox("🌆 Ville Cible", list(city_defaults.keys()))
defaults = city_defaults[selected_city]

# ---- Configuration financière ----
st.sidebar.subheader("💰 Hypothèses Financières")
capex = st.sidebar.number_input(
    "Investissement (MUSD)", min_value=10.0, max_value=500.0, value=defaults['capex'], step=5.0)
base_adr = st.sidebar.number_input(
    "Tarif Initial ADR (USD)", min_value=50.0, max_value=1000.0, value=defaults['adr'], step=10.0)
rooms = st.sidebar.number_input(
    "Nombre de Chambres", min_value=10, max_value=1000, value=200, step=10)
base_occupancy = st.sidebar.slider("Taux d'Occupation de Base (%)", 30.0, 95.0, 65.0, 1.0) / 100
discount_rate  = st.sidebar.slider("Taux d'Actualisation WACC (%)", 2.0, 20.0, 8.0, 0.5) / 100
opex_margin    = st.sidebar.slider("Marge Opérationnelle OpEx (%)", 40.0, 90.0, 65.0, 1.0) / 100

# ---- Coupe du Monde ----
st.sidebar.subheader("⚽ Coupe du Monde FIFA 2030")
enable_wc_boost = st.sidebar.checkbox("Activer les impacts 2030", value=True)
wc_adr_boost    = st.sidebar.slider("Boost ADR 2030 (%)", 0, 100, 40, 5) / 100
inflation_rate  = st.sidebar.slider("Taux d'inflation annuel moyen (%)", 0.0, 10.0, 2.5, 0.1) / 100

# ---- Paramètres modélisation ----
st.sidebar.subheader("🤖 Modélisation & Features")
dl_epochs = st.sidebar.slider("Époques Deep Learning", 1, 50, 10, 5)

default_features_arrivals = feat.get_feature_list()
default_features_nights   = feat.get_nights_feature_list()

if use_nights:
    options_features = default_features_nights
    default_sel = [
        'nights_lag_1', 'nights_lag_2', 'nights_lag_12',
        'nights_roll_mean_3', 'nights_roll_mean_12',
        'nuitees_per_arrival',
        'month_sin', 'month_cos', 'year',
        'is_summer', 'is_high_season', 'cdm_event', 'is_covid',
        'anomaly_iforest', 'anomaly_prophet'
    ]
else:
    options_features = default_features_arrivals
    default_sel = [
        'lags_1', 'lags_2', 'lags_12',
        'roll_mean_3', 'roll_mean_12',
        'month_sin', 'month_cos', 'year',
        'is_summer', 'is_high_season', 'cdm_event', 'is_covid',
        'anomaly_iforest', 'anomaly_prophet'
    ]

selected_features = st.sidebar.multiselect(
    "Variables d'entrée (Features)",
    options=options_features,
    default=[f for f in default_sel if f in options_features]
)

# ---------------------------------------------------------------------------
# CORPS DE LA PAGE
# ---------------------------------------------------------------------------
st.title("🇲🇦 Simulateur Interactif de ROI Hôtelier - Horizon 2035")
st.markdown(
    f"Ce tableau de bord simule la profitabilité d'un investissement hôtelier sur 10 ans "
    f"(**2026–2035**) en s'appuyant sur les **Top 3 modèles** les plus performants pour la "
    f"cible sélectionnée : "
    f"<b>{'Nuitées 🌙' if use_nights else 'Arrivées ✈️'}</b>.",
    unsafe_allow_html=True
)

# Métriques de la ville
col_badge1, col_badge2, col_badge3 = st.columns(3)
with col_badge1:
    st.markdown(
        f'<div class="metric-card teal"><h4>Recommandation de Référence</h4>'
        f'<p class="val" style="color:#0d9488;font-size:18px;">{defaults["rec"]}</p></div>',
        unsafe_allow_html=True
    )
with col_badge2:
    card_cls = "purple" if use_nights else ""
    st.markdown(
        f'<div class="metric-card {card_cls}"><h4>Part des Nuitées Nationales</h4>'
        f'<p class="val" style="color:#d97706;font-size:18px;">{defaults["part"]*100:.1f} %</p></div>',
        unsafe_allow_html=True
    )
with col_badge3:
    st.markdown(
        f'<div class="metric-card teal"><h4>Capitaux d\'Investissement Estimés</h4>'
        f'<p class="val" style="color:#0f172a;font-size:18px;">{capex:.1f} Millions USD</p></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Identification automatique du Top 3 (par cible)
# ---------------------------------------------------------------------------
model_name_mapping = {
    'SARIMA': 'SARIMA', 'SARIMAX': 'SARIMA', 'Prophet': 'Prophet',
    'Ridge': 'Ridge', 'Random Forest': 'Random Forest',
    'Extra Trees': 'Extra Trees', 'Gradient Boosting': 'Gradient Boosting',
    'AdaBoost': 'AdaBoost', 'XGBoost': 'XGBoost',
    'LightGBM': 'LightGBM', 'CatBoost': 'CatBoost', 'SVR': 'SVR',
    'LSTM': 'LSTM', 'Best DL (LSTM)': 'LSTM', 'SimpleRNN': 'SimpleRNN'
}

if use_nights:
    metrics_path = os.path.join("data", "model_performance_metrics_nuitees.csv")
    fallback_list = ['Ridge', 'XGBoost', 'LSTM', 'SARIMA', 'Prophet']
else:
    metrics_path = os.path.join("data", "model_performance_metrics.csv")
    fallback_list = ['Ridge', 'XGBoost', 'LSTM', 'SARIMA', 'Prophet']

top_3_models = []
if os.path.exists(metrics_path):
    try:
        metrics_df = pd.read_csv(metrics_path)
        metrics_df['Mapped_Model'] = metrics_df['Model'].map(model_name_mapping)
        valid_df = metrics_df.dropna(subset=['Mapped_Model'])
        valid_df = valid_df.sort_values(by='R2', ascending=False)
        top_3_models = valid_df['Mapped_Model'].unique().tolist()
    except Exception as e:
        st.warning(f"Impossible de lire le fichier de métriques : {e}")

for fallback in fallback_list:
    if len(top_3_models) >= 3:
        break
    if fallback not in top_3_models:
        top_3_models.append(fallback)
top_3_models = top_3_models[:3]

target_label = "Nuitées 🌙" if use_nights else "Arrivées ✈️"
file_note = (f" | 📂 `{os.path.basename(metrics_path)}`"
             if os.path.exists(metrics_path)
             else " | ⚠️ Métriques non trouvées — fallback utilisé")
st.info(
    f"💡 **Top 3 modèles retenus pour : {target_label}** — "
    + ", ".join([f"`{m}`" for m in top_3_models])
    + file_note
)

# ---------------------------------------------------------------------------
# Bouton de simulation
# ---------------------------------------------------------------------------
sim_btn = st.button("🚀 Lancer la Simulation Comparée")

if sim_btn:
    if not selected_features:
        st.error("Veuillez sélectionner au moins une feature dans la barre latérale.")
    else:
        with st.spinner(
            f"Génération des prévisions 2026–2035 ({target_label}) et calcul des cash flows..."
        ):
            # Données de référence 2025
            df_2025 = df_clean[pd.to_datetime(df_clean['Date']).dt.year == 2025]
            arrivals_2025 = df_2025[TARGET_COL].sum()
            nights_2025 = (df_2025[NIGHTS_COL].sum()
                           if NIGHTS_COL in df_2025.columns else None)

            future_dates = pd.date_range(start='2026-05-01', end='2035-12-01', freq='MS')

            df_featured = feat.build_features(df_clean)
            df_ml = df_featured.dropna(subset=[active_target_col]).copy()

            valid_sel = [f for f in selected_features if f in df_ml.columns]
            X_train = df_ml[valid_sel].fillna(df_ml[valid_sel].median())
            y_train = df_ml[active_target_col]

            df_2026_real = df_clean[pd.to_datetime(df_clean['Date']).dt.year == 2026]
            real_arrivals_2026 = df_2026_real[TARGET_COL].sum()
            real_nights_2026 = (df_2026_real[NIGHTS_COL].sum()
                                if NIGHTS_COL in df_2026_real.columns else 0)

            sim = HotelROISimulator(
                rooms=rooms,
                investment_usd=capex * 1e6,
                opex_margin=opex_margin,
                discount_rate=discount_rate,
                base_occupancy=base_occupancy,
                wc_adr_boost_pct=wc_adr_boost,
                inflation_rate=inflation_rate
            )

            model_sim_dfs = {}
            model_metrics = {}

            for model_name in top_3_models:
                preds = generate_projections(
                    model_name=model_name,
                    target_col=active_target_col,
                    df_ml=df_ml,
                    future_dates=future_dates,
                    selected_features=valid_sel,
                    X_train=X_train,
                    y_train=y_train,
                    dl_epochs=dl_epochs
                )
                if preds is None:
                    continue

                df_temp = pd.DataFrame({'Date': future_dates, 'Value': preds})

                if use_nights:
                    # Mode Nuitées : occupation directe
                    annual_nights = {}
                    for y in range(2026, 2036):
                        if y == 2026:
                            pred_yr = df_temp[df_temp['Date'].dt.year == 2026]['Value'].sum()
                            annual_nights[2026] = real_nights_2026 + pred_yr
                        else:
                            annual_nights[y] = df_temp[df_temp['Date'].dt.year == y]['Value'].sum()

                    df_sim = sim.simulate_with_nuitees_forecast(
                        annual_nights_dict=annual_nights,
                        start_year=2026,
                        wc_boost_2030=enable_wc_boost
                    )
                else:
                    # Mode Arrivées : croissance relative
                    annual_arrivals = {}
                    for y in range(2026, 2036):
                        if y == 2026:
                            pred_yr = df_temp[df_temp['Date'].dt.year == 2026]['Value'].sum()
                            annual_arrivals[2026] = real_arrivals_2026 + pred_yr
                        else:
                            annual_arrivals[y] = df_temp[df_temp['Date'].dt.year == y]['Value'].sum()

                    df_sim = sim.simulate_with_forecast(
                        annual_arrivals_dict=annual_arrivals,
                        baseline_arrivals_2025=arrivals_2025,
                        start_year=2026,
                        wc_boost_2030=enable_wc_boost
                    )

                metrics = sim.calculate_metrics_for_gop(df_sim['GOP_USD'])
                model_sim_dfs[model_name] = df_sim
                model_metrics[model_name] = metrics

            st.session_state['sim_model_sim_dfs'] = model_sim_dfs
            st.session_state['sim_model_metrics']  = model_metrics
            st.session_state['sim_selected_city']  = selected_city
            st.session_state['sim_capex']          = capex
            st.session_state['sim_adr']            = base_adr
            st.session_state['sim_rooms']          = rooms
            st.session_state['sim_use_nights']     = use_nights
            st.session_state['sim_discount_rate']  = discount_rate
            st.session_state['sim_top3']           = top_3_models

# ---------------------------------------------------------------------------
# Affichage des résultats
# ---------------------------------------------------------------------------
if 'sim_model_metrics' in st.session_state and st.session_state['sim_model_metrics']:
    roi_metrics  = st.session_state['sim_model_metrics']
    roi_sim_dfs  = st.session_state['sim_model_sim_dfs']
    city_name    = st.session_state['sim_selected_city']
    cap          = st.session_state['sim_capex']
    rms          = st.session_state['sim_rooms']
    nights_mode  = st.session_state.get('sim_use_nights', False)
    d_rate       = st.session_state.get('sim_discount_rate', 0.08)

    mode_label = "Nuitées 🌙" if nights_mode else "Arrivées ✈️"
    st.subheader(f"📊 Bilan ROI Hôtelier — {city_name} | Mode : {mode_label}")
    st.markdown(f"Hôtel de `{rms}` chambres — Investissement : `{cap}` M$ USD")

    # ---- Tableau comparatif ----
    summary_records = []
    for name, m in roi_metrics.items():
        df_s = roi_sim_dfs[name]
        revpar_avg = (
            f"{df_s['RevPAR_USD'].mean():.1f} $"
            if 'RevPAR_USD' in df_s.columns else "—"
        )
        summary_records.append({
            'Modèle': name,
            'NPV (M$)': f"{m['NPV_MUSD']:.2f}",
            'IRR (%)': f"{m['IRR_Pct']:.2f}" if not np.isnan(m['IRR_Pct']) else "N/A",
            'Payback (ans)': f"{m['Payback_Years']}" if not np.isnan(m['Payback_Years']) else ">10",
            'ROI 10 ans (%)': f"{m['ROI_Pct']:.2f}",
            'RevPAR Moy. ($/nuit)': revpar_avg
        })

    st.dataframe(pd.DataFrame(summary_records), use_container_width=True)

    # ---- Graphique courbes de profit cumulé ----
    st.subheader("📈 Courbes Comparatives de Profit Cumulé (2025–2035)")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    years = [2025] + list(range(2026, 2036))
    colors_list  = ['#0d9488', '#d97706', '#2563eb']
    markers_list = ['o', 's', '^']

    for idx, (name, df_sim) in enumerate(roi_sim_dfs.items()):
        cf     = [-cap] + (df_sim['GOP_USD'] / 1e6).tolist()
        cum_cf = np.cumsum(cf)
        lbl = (f"{name} (NPV={roi_metrics[name]['NPV_MUSD']:.1f}M$, "
               f"IRR={roi_metrics[name]['IRR_Pct']:.1f}%)")
        ax.plot(years, cum_cf, marker=markers_list[idx % 3],
                color=colors_list[idx % 3], label=lbl, linewidth=2.5)

    ax.axhline(0, color='black', linestyle=':', alpha=0.7)
    mode_str = "Nuitées" if nights_mode else "Arrivées"
    ax.set_title(
        f"Évolution Comparée des Profits Cumulés — {city_name} | Prédiction {mode_str}",
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel("Année de projet")
    ax.set_ylabel("Profit Cumulé (Millions USD)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig)

    # ---- RevPAR annuel (mode Nuitées uniquement) ----
    if nights_mode and any('RevPAR_USD' in df for df in roi_sim_dfs.values()):
        st.subheader("🏨 RevPAR Annuel (Revenue Per Available Room) — Mode Nuitées")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        for idx, (name, df_sim) in enumerate(roi_sim_dfs.items()):
            if 'RevPAR_USD' in df_sim.columns:
                ax2.plot(df_sim['Year'], df_sim['RevPAR_USD'],
                         marker=markers_list[idx % 3],
                         color=colors_list[idx % 3], label=name, linewidth=2)
        ax2.set_title("Évolution du RevPAR (USD/nuit) 2026–2035", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Année")
        ax2.set_ylabel("RevPAR (USD/nuit)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig2)

    # ---- Recommandation d'expert ----
    best_model_roi = max(roi_metrics.keys(), key=lambda x: roi_metrics[x]['ROI_Pct'])
    max_roi    = roi_metrics[best_model_roi]['ROI_Pct']
    irr_val    = roi_metrics[best_model_roi]['IRR_Pct']
    payback_val = roi_metrics[best_model_roi]['Payback_Years']

    st.subheader("📝 Note d'Analyse et Conseils Stratégiques")

    decision = ("FAVORABLE" if max_roi >= 80.0
                else "À ÉTUDIER" if max_roi >= 40.0
                else "DEFAVORABLE")
    decision_color = (
        "#059669" if decision == "FAVORABLE"
        else "#d97706" if decision == "À ÉTUDIER"
        else "#dc2626"
    )

    revpar_note = ""
    if nights_mode:
        best_df = roi_sim_dfs[best_model_roi]
        if 'RevPAR_USD' in best_df.columns:
            avg_revpar = best_df['RevPAR_USD'].mean()
            revpar_note = (
                f"<li>Un <b>RevPAR moyen de {avg_revpar:.1f} $/nuit</b> sur la "
                f"période 2026–2035 (calculé directement via les nuitées prédites).</li>"
            )

    st.markdown(f"""
    <div style="background-color:#f8fafc; padding:22px; border-radius:10px;
                border-left:6px solid {decision_color}; margin-top:15px;">
        <h4 style="color:{decision_color}; margin-top:0; font-size:18px;">
            Décision Financière Préconisée : {decision}
        </h4>
        <p>En couplant les volumes de <b>{mode_label}</b> modélisés par
        <b>{best_model_roi}</b> avec les spécificités de la ville de
        <b>{city_name}</b>, le projet présente :</p>
        <ul>
            <li>Un <b>IRR de {irr_val:.2f}%</b> (WACC = {d_rate*100:.1f}%).</li>
            <li>Une NPV estimée à
                <b>{roi_metrics[best_model_roi]['NPV_MUSD']:.2f} M$ USD</b>.</li>
            <li>Un délai de récupération de <b>{payback_val} ans</b>.</li>
            {revpar_note}
        </ul>
        <p style="margin-bottom:0;"><i>Note sectorielle :</i> {city_name} capte environ
        <b>{defaults['part']*100:.0f}%</b> du volume de nuitées nationales.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info(
        "Veuillez configurer vos paramètres dans la barre latérale gauche, "
        "puis cliquez sur 'Lancer la Simulation Comparée' ci-dessus."
    )
