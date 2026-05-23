"""
Serveur API Flask - Stratégie Hôtelière Maroc 2030
=================================================

Ce script déploie un serveur Flask RESTful servant de backend API et distribuant
une interface web interactive personnalisée de haute qualité.

Auteurs: Achraf Lemrani & Hamza El Faghloumi
Filière: IATD-SI --- ENSAM Meknès
"""

import os
import sys
import logging
from flask import Flask, render_template, jsonify, request, send_from_directory

# Importation des répertoires depuis src.config
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.config import DATA_DIR, FIGURES_DIR, TARGET_COL

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route('/figures/<path:filename>')
def serve_figure(filename):
    """Sert les images du dossier figures/."""
    return send_from_directory(FIGURES_DIR, filename)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FlaskBackend")

# Données ROI par défaut de l'étude (Tableau 6.3 du rapport)
DEFAULT_ROI_DATA = {
    'Marrakech': {'adr': 207, 'pct': 35, 'revenue': 24.5, 'cost': 150.0, 'roi': 12.5, 'status': 'Investir'},
    'Casablanca': {'adr': 192, 'pct': 20, 'revenue': 25.2, 'cost': 180.0, 'roi': 10.8, 'status': 'Investir'},
    'Agadir': {'adr': 145, 'pct': 18, 'revenue': 22.1, 'cost': 130.0, 'roi': 7.5, 'status': 'A etudier'},
    'Tanger': {'adr': 137, 'pct': 10, 'revenue': 23.5, 'cost': 145.0, 'roi': 6.2, 'status': 'Attendre'},
    'Rabat': {'adr': 153, 'pct': 9, 'revenue': 24.8, 'cost': 165.0, 'roi': 5.8, 'status': 'Attendre'},
    'Fes': {'adr': 121, 'pct': 8, 'revenue': 21.5, 'cost': 120.0, 'roi': 3.2, 'status': 'Eviter'}
}

@app.route('/')
def index():
    """Sert la page web principale."""
    return render_template('index.html')

@app.route('/api/metrics')
def get_metrics():
    """Retourne les métriques de performance des modèles pré-calculés."""
    path = os.path.join(DATA_DIR, 'model_performance_metrics.csv')
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return jsonify(df.to_dict(orient='records'))
        except Exception as e:
            logger.error(f"Erreur de lecture des métriques: {e}")
            
    # Fallback/Mock si le pipeline n'a pas été lancé
    mock_metrics = [
        {'Model': 'Best DL (LSTM)', 'R2': 0.9412, 'RMSE': 108.40, 'MAE': 75.30, 'MAPE': 5.80},
        {'Model': 'Transformer', 'R2': 0.9125, 'RMSE': 125.10, 'MAE': 88.40, 'MAPE': 6.80},
        {'Model': 'Hybrid Model', 'R2': 0.8805, 'RMSE': 145.20, 'MAE': 98.10, 'MAPE': 7.20},
        {'Model': 'Ridge', 'R2': 0.7826, 'RMSE': 182.18, 'MAE': 127.94, 'MAPE': 8.17},
        {'Model': 'SARIMAX', 'R2': 0.7095, 'RMSE': 208.37, 'MAE': 143.97, 'MAPE': 8.88},
        {'Model': 'XGBoost', 'R2': 0.0106, 'RMSE': 388.62, 'MAE': 291.59, 'MAPE': 18.07}
    ]
    return jsonify(mock_metrics)

@app.route('/api/simulate-roi', methods=['POST'])
def simulate_roi():
    """Calcule dynamiquement le ROI hôtelier sur 10 ans selon la saisie utilisateur."""
    try:
        data = request.json or {}
        
        # Récupération des paramètres avec valeurs par défaut de l'étude
        city = data.get('city', 'Marrakech')
        chambres = int(data.get('chambres', 200))
        occ_base = float(data.get('occ_base', 65.0))
        occ_cdm = float(data.get('occ_cdm', 85.0))
        adr_2024 = float(data.get('adr_2024', 207.0))
        inflation = float(data.get('inflation', 1.2))
        cost_room = float(data.get('cost_room', 120000.0))
        
        # 1. Calcul de l'ADR en 2030 (après 6 ans d'inflation)
        adr_2030 = adr_2024 * ((1 + (inflation / 100)) ** 6)
        
        # 2. Capital Expenditure (investissement initial)
        capex = chambres * cost_room
        
        # 3. Calcul des revenus sur 10 ans (Année 6 = Coupe du Monde avec boost 15% ADR et occ_cdm)
        total_revenues = 0
        for annee in range(1, 11):
            if annee == 6:
                occ = occ_cdm / 100.0
                adr_year = adr_2030 * 1.15
            else:
                occ = occ_base / 100.0
                adr_year = adr_2024 * ((1 + (inflation / 100)) ** annee)
                
            revenue_year = chambres * 365 * occ * adr_year
            total_revenues += revenue_year
            
        # 4. Calcul du ROI en %
        roi_percent = ((total_revenues - capex) / capex) * 100
        
        # Détermination du statut de recommandation
        if roi_percent > 12.0:
            status = 'Investir'
            rec_text = "FORTEMENT RECOMMANDÉ : Rentabilité élevée par rapport aux risques du marché."
        elif roi_percent > 6.0:
            status = 'A etudier'
            rec_text = "À ÉTUDIER : Rentabilité modérée, nécessite une étude d'optimisation opérationnelle."
        elif roi_percent > 3.0:
            status = 'Attendre'
            rec_text = "ATTENDRE : Rendements trop faibles face aux risques fonciers actuels."
        else:
            status = 'Eviter'
            rec_text = "ÉVITER : Risque de perte en capital ou de rendement inférieur au taux sans risque."
            
        # Préparation du tableau comparatif pour le graphique frontend
        comparative_data = []
        for c, val in DEFAULT_ROI_DATA.items():
            if c == city:
                comparative_data.append({'city': f"{c} (Simulé)", 'roi': round(roi_percent, 2)})
            else:
                comparative_data.append({'city': c, 'roi': val['roi']})
                
        return jsonify({
            'city': city,
            'capex_musd': round(capex / 1e6, 2),
            'adr_2030_usd': round(adr_2030, 2),
            'revenues_10y_musd': round(total_revenues / 1e6, 2),
            'roi_percent': round(roi_percent, 2),
            'status': status,
            'recommendation_text': rec_text,
            'comparative': comparative_data
        })
        
    except Exception as e:
        logger.error(f"Erreur de calcul du ROI: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Flask s'exécute sur le port 5000 par défaut
    logger.info("Lancement de l'application web d'économie touristique...")
    app.run(host='0.0.0.0', port=5000, debug=True)
