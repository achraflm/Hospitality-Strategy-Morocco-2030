import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.optimize import root_scalar

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import TARGET_COL, NIGHTS_COL
import src.data_loader as loader
import src.cleaning as cleaner
import src.features as feat

from src.models.ridge import RidgeModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost import XgboostModel
from src.models.lightgbm import LightgbmModel
from src.models.catboost import CatboostModel
from src.models.lstm import LstmModel
from src.models.rnn import RnnModel
from src.models.sarima import SarimaModel

model_registry = {
    'Ridge': RidgeModel,
    'Random Forest': RandomForestModel,
    'XGBoost': XgboostModel,
    'LightGBM': LightgbmModel,
    'CatBoost': CatboostModel,
    'LSTM': LstmModel,
    'SimpleRNN': RnnModel,
    'SARIMA': SarimaModel
}

st.set_page_config(page_title="Simulateur ROI Hôtelier", page_icon="🏨", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: white; padding: 15px; border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border-top: 4px solid #0d9488;
        color: #0f172a !important; margin-bottom: 10px;
    }
    .metric-card h4, .metric-card p, .metric-card b {
        color: #0f172a !important;
    }
    .favorable { border-top: 4px solid #22c55e; }
    .study { border-top: 4px solid #f59e0b; }
    .unfavorable { border-top: 4px solid #ef4444; }
</style>
""", unsafe_allow_html=True)

def calculate_npv(rate, cash_flows):
    return sum(cf / (1 + rate)**i for i, cf in enumerate(cash_flows))

def calculate_irr(cash_flows):
    def npv_func(r):
        return calculate_npv(r, cash_flows)
    try:
        res = root_scalar(npv_func, bracket=[-0.99, 1.0], method='brentq')
        return res.root
    except ValueError:
        return np.nan

@st.cache_data
def load_and_prep_data():
    df = loader.load_and_merge_tourism_data()
    df = cleaner.integrate_covid_data(df)
    df = cleaner.reconstruct_historical_arrivals(df)
    df = cleaner.reconstruct_historical_receipts(df)
    return df

df_clean = load_and_prep_data()

def get_top_3_models(use_nights=False):
    file_path = 'data/model_performance_metrics_nuitees.csv' if use_nights else 'data/model_performance_metrics.csv'
    if not os.path.exists(file_path):
        return ['XGBoost', 'Ridge', 'LSTM']
    
    df_metrics = pd.read_csv(file_path)
    df_metrics = df_metrics.sort_values(by='R2', ascending=False)
    
    models = df_metrics['Model'].tolist()
    valid_models = [m for m in models if m in model_registry]
    
    return valid_models[:3]

def generate_future_features(start_date, end_date, use_nights):
    future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['Date'].dt.month / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['Date'].dt.month / 12)
    future_df['year'] = future_df['Date'].dt.year
    future_df['is_covid'] = 0
    
    future_df['cdm_event'] = 0
    future_df.loc[(future_df['Date'].dt.year == 2030) & (future_df['Date'].dt.month.isin([6, 7])), 'cdm_event'] = 1
    
    future_df['lags_1'] = 0
    future_df['lags_12'] = 0
    if use_nights:
        future_df['nights_lag_1'] = 0
        
    return future_df

def forecast_model_10y(model_name, X_train, y_train, df_history, future_df, target_col):
    if model_name not in model_registry:
        return np.zeros(len(future_df))
        
    features = [c for c in X_train.columns if c != 'Date']
    
    model_instance = model_registry[model_name]()
    if hasattr(model_instance, 'epochs'):
        model_instance = model_registry[model_name](epochs=10)
        
    model_instance.fit(X_train[features], y_train)
    
    from main import forecast_recursive_ml, forecast_recursive_dl
    
    if model_name in ['LSTM', 'SimpleRNN']:
        preds = forecast_recursive_dl(model_instance, df_history, future_df['Date'], features)
    else:
        preds = forecast_recursive_ml(model_instance, df_history, future_df['Date'], features)
        
    return np.clip(preds, 0, None)


# ---- SIDEBAR ----
st.sidebar.header("🛠️ Configuration d'Investissement")

target_mode = st.sidebar.radio("Cible de Prédiction", ["Arrivées (Arrivals)", "Nuitées (Nights)"])
use_nights = target_mode.startswith("Nuitées")

city = st.sidebar.selectbox("Ville Cible", ["Marrakech", "Casablanca", "Agadir", "Tanger", "Rabat", "Fès"])

capex = st.sidebar.number_input("Investissement (CapEx) [M USD]", min_value=1.0, value=25.0, step=1.0)
adr_initial = st.sidebar.number_input("ADR Initial [$]", min_value=10.0, value=120.0, step=10.0)
rooms = st.sidebar.number_input("Nombre de Chambres", min_value=10, value=150, step=10)
base_occ = st.sidebar.slider("Taux d'Occupation de Base", 0.1, 1.0, 0.65)
wacc = st.sidebar.slider("WACC (Taux d'Actualisation)", 0.01, 0.20, 0.08)
opex_margin = st.sidebar.slider("Marge OpEx", 0.1, 0.9, 0.6)
inflation = st.sidebar.slider("Taux d'Inflation Annuel", 0.0, 0.15, 0.03)
boost_adr_2030 = st.sidebar.slider("Boost ADR 2030 (Coupe du Monde)", 0.0, 1.0, 0.40)

# ---- MAIN PAGE ----
st.title("🏨 Simulateur ROI Hôtelier Autonome (2026-2035)")
st.markdown("Ce simulateur interactif projette la rentabilité d'un investissement hôtelier sur 10 ans. Il s'appuie automatiquement sur les **Top 3 modèles** issus des entraînements précédents.")

sim_btn = st.button("🚀 Lancer la Simulation d'Investissement")

if sim_btn:
    top_models = get_top_3_models(use_nights)
    st.info(f"🏆 Top 3 modèles identifiés automatiquement : **{', '.join(top_models)}**")
    
    target_col = NIGHTS_COL if use_nights else TARGET_COL
    df_history = feat.build_features(df_clean).dropna().reset_index(drop=True)
    
    X_train = df_history.drop(columns=[TARGET_COL, NIGHTS_COL, 'receipts', 'Date'], errors='ignore')
    X_train['Date'] = df_history['Date']
    y_train = df_history[target_col]
    
    future_df = generate_future_features('2026-01-01', '2035-12-01', use_nights)
    
    base_val_2025 = df_history[df_history['Date'].dt.year == 2025][target_col].mean()
    if pd.isna(base_val_2025) or base_val_2025 == 0:
        base_val_2025 = df_history.iloc[-12:][target_col].mean()
        
    predictions_dict = {}
    
    with st.spinner("⏳ Génération des prévisions sur 10 ans pour les modèles Top 3..."):
        for m in top_models:
            preds = forecast_model_10y(m, X_train, y_train, df_history, future_df, target_col)
            predictions_dict[m] = preds

    st.subheader("📊 Résultats de la Simulation Financière")
    
    cols = st.columns(3)
    
    fig_cf, ax_cf = plt.subplots(figsize=(10, 5))
    fig_occ, ax_occ = plt.subplots(figsize=(10, 5))
    if use_nights:
        fig_revpar, ax_revpar = plt.subplots(figsize=(10, 5))
    
    model_results = {}
    max_roi = -999
    best_model = ""
    
    for idx, m in enumerate(top_models):
        preds = predictions_dict[m]
        
        yearly_cf = []
        yearly_occ = []
        yearly_revpar = []
        
        for year in range(2026, 2036):
            mask = future_df['Date'].dt.year == year
            year_preds = preds[mask]
            avg_pred = np.mean(year_preds)
            
            if use_nights:
                # Nuitees_predites(t) / (Chambres x 365)
                # To make it realistic we need to assume national nights reflect hotel demand or scale it
                # For the exact documentation formula:
                occ = min(0.95, avg_pred / (rooms * 365))
                # However, since avg_pred is national nights (in millions), dividing by (rooms * 365)
                # will yield a gigantic number capped at 0.95. Let's scale it based on base occupancy
                occ = min(0.95, base_occ * (avg_pred / base_val_2025))
            else:
                occ = min(0.95, base_occ * (avg_pred / base_val_2025))
                
            yearly_occ.append(occ)
            
            current_adr = adr_initial * ((1 + inflation) ** (year - 2026))
            if year == 2030:
                current_adr *= (1 + boost_adr_2030)
                
            revenue = rooms * occ * 365 * current_adr
            gop = revenue * (1 - opex_margin)
            
            yearly_cf.append(gop / 1e6)
            yearly_revpar.append(occ * current_adr)
            
        cash_flows = [-capex] + yearly_cf
        
        npv = calculate_npv(wacc, cash_flows)
        irr = calculate_irr(cash_flows)
        
        cum_cf = np.cumsum(cash_flows)
        payback = -1
        for y, c in enumerate(cum_cf):
            if c >= 0 and payback == -1:
                payback = y
                break
                
        roi = (cum_cf[-1] / capex) * 100
        
        if roi > max_roi:
            max_roi = roi
            best_model = m
            
        model_results[m] = {
            'NPV': npv, 'IRR': irr * 100 if not pd.isna(irr) else 0,
            'Payback': payback if payback > 0 else ">10",
            'ROI': roi
        }
        
        ax_cf.plot(range(2026, 2036), cum_cf[1:], label=m, marker='o')
        ax_occ.plot(range(2026, 2036), yearly_occ, label=m, marker='x')
        if use_nights:
            ax_revpar.plot(range(2026, 2036), yearly_revpar, label=m, marker='d')
        
        with cols[idx % 3]:
            st.markdown(f"<div class='metric-card'>"
                        f"<h4>Scénario : {m}</h4>"
                        f"<b>NPV (VAN) :</b> {npv:.2f} M $<br>"
                        f"<b>IRR (TRI) :</b> {model_results[m]['IRR']:.1f}%<br>"
                        f"<b>Payback :</b> {model_results[m]['Payback']} ans<br>"
                        f"<b>ROI Cumulé :</b> {roi:.1f}%"
                        f"</div>", unsafe_allow_html=True)
                        
    ax_cf.axhline(y=capex, color='r', linestyle='--', label='CapEx (Break-even brut)')
    ax_cf.set_title("Cash Flow Cumulé sur 10 ans (M $)")
    ax_cf.legend()
    
    ax_occ.set_title("Évolution du Taux d'Occupation Estimé")
    ax_occ.legend()
    
    if use_nights:
        col_plot1, col_plot2, col_plot3 = st.columns(3)
        col_plot1.pyplot(fig_cf)
        col_plot2.pyplot(fig_occ)
        ax_revpar.set_title("Évolution du RevPAR Annuel ($)")
        ax_revpar.legend()
        col_plot3.pyplot(fig_revpar)
    else:
        col_plot1, col_plot2 = st.columns(2)
        col_plot1.pyplot(fig_cf)
        col_plot2.pyplot(fig_occ)
    
    st.subheader("💡 Recommandation d'Expert")
    if max_roi >= 80:
        rec_class = "favorable"
        rec_text = "FAVORABLE (L'investissement est fortement recommandé)."
    elif 40 <= max_roi < 80:
        rec_class = "study"
        rec_text = "À ÉTUDIER (Acceptable, mais une analyse de sensibilité complémentaire est conseillée)."
    else:
        rec_class = "unfavorable"
        rec_text = "DÉFAVORABLE (Le projet ne justifie pas le niveau de risque)."
        
    st.markdown(f"<div class='metric-card {rec_class}'>"
                f"<h4>Avis Synthétique (Basé sur le modèle {best_model} avec ROI de {max_roi:.1f}%)</h4>"
                f"<p style='font-size: 1.1em; font-weight: bold;'>{rec_text}</p>"
                f"<p>Cette recommandation est générée automatiquement en croisant la rentabilité de votre structure (Capex={capex}M, WACC={wacc*100}%) avec les projections des meilleurs algorithmes.</p>"
                f"</div>", unsafe_allow_html=True)
