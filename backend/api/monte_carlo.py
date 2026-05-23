import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.roi_simulator import HotelROISimulator

router = APIRouter()

class MonteCarloRequest(BaseModel):
    rooms: int = 200
    capex_m: float = 150.0
    base_adr: float = 250.0
    base_occupancy: float = 0.65
    discount_rate: float = 0.08
    opex_margin: float = 0.65
    wc_adr_boost: float = 0.40
    inflation_rate: float = 0.025
    simulations_count: int = 500
    enable_wc: bool = True

@router.post("/simulate")
def run_monte_carlo(req: MonteCarloRequest):
    try:
        npvs = []
        irrs = []
        rois = []
        
        # Générer des distributions de variables aléatoires
        # Utiliser un générateur aléatoire fixe pour la reproductibilité (seed)
        rng = np.random.default_rng(seed=42)
        
        # Tirages aléatoires des variables
        sampled_inflation = rng.normal(loc=req.inflation_rate, scale=0.005, size=req.simulations_count)
        sampled_occupancy = rng.normal(loc=req.base_occupancy, scale=0.03, size=req.simulations_count)
        sampled_opex = rng.normal(loc=req.opex_margin, scale=0.02, size=req.simulations_count)
        
        if req.enable_wc:
            sampled_wc_adr = rng.triangular(
                left=max(0.0, req.wc_adr_boost - 0.15), 
                mode=req.wc_adr_boost, 
                right=req.wc_adr_boost + 0.20, 
                size=req.simulations_count
            )
        else:
            sampled_wc_adr = np.zeros(req.simulations_count)
            
        # Lancer les simulations
        for i in range(req.simulations_count):
            # Paramètres de cette simulation
            inf = max(0.0, float(sampled_inflation[i]))
            occ = min(0.95, max(0.20, float(sampled_occupancy[i])))
            opex = min(0.90, max(0.30, float(sampled_opex[i])))
            adr_boost = max(0.0, float(sampled_wc_adr[i]))
            
            sim = HotelROISimulator(
                rooms=req.rooms,
                investment_usd=req.capex_m * 1e6,
                opex_margin=opex,
                discount_rate=req.discount_rate,
                base_occupancy=occ,
                wc_adr_boost_pct=adr_boost,
                inflation_rate=inf
            )
            
            df_sim = sim.simulate_10years(start_year=2026)
            
            if not req.enable_wc:
                df_sim['ADR_WC_USD'] = df_sim['ADR_Base_USD']
                df_sim['Occ_WC'] = df_sim['Occ_Base']
                df_sim['Revenue_WC_USD'] = df_sim['Revenue_Base_USD']
                df_sim['GOP_WC_USD'] = df_sim['GOP_Base_USD']
                
            metrics = sim.calculate_metrics(df_sim)
            
            # Récupérer les métriques Coupe du monde (ou base si désactivé)
            if req.enable_wc:
                npvs.append(metrics['NPV_WC_MUSD'])
                irrs.append(metrics['IRR_WC_Pct'])
                rois.append(metrics['ROI_WC_Pct'])
            else:
                npvs.append(metrics['NPV_Base_MUSD'])
                irrs.append(metrics['IRR_Base_Pct'])
                rois.append(metrics['ROI_Base_Pct'])
                
        # Nettoyer les irrs (supprimer les NaN)
        valid_irrs = [x for x in irrs if not np.isnan(x)]
        
        # 3. Statistiques descriptives
        mean_npv = float(np.mean(npvs))
        std_npv = float(np.std(npvs))
        mean_irr = float(np.mean(valid_irrs)) if valid_irrs else 0.0
        mean_roi = float(np.mean(rois))
        std_roi = float(np.std(rois))
        
        # Probabilité de perte
        prob_loss = float(np.sum(np.array(npvs) < 0) / len(npvs) * 100)
        
        # Value at Risk (VaR 95%): 5ème percentile de la NPV
        var_95_npv = float(np.percentile(npvs, 5))
        
        # Intervalles de confiance (5% et 95%)
        ci_npv_low = float(np.percentile(npvs, 5))
        ci_npv_high = float(np.percentile(npvs, 95))
        ci_roi_low = float(np.percentile(rois, 5))
        ci_roi_high = float(np.percentile(rois, 95))
        
        # Préparer les données pour l'histogramme Recharts (ROI %)
        roi_array = np.array(rois)
        counts, bins = np.histogram(roi_array, bins=25)
        
        histogram_data = []
        for idx in range(len(counts)):
            bin_label = f"{bins[idx]:.1f}% - {bins[idx+1]:.1f}%"
            histogram_data.append({
                "BinRange": bin_label,
                "MinVal": float(bins[idx]),
                "MaxVal": float(bins[idx+1]),
                "Count": int(counts[idx])
            })
            
        return {
            "summary": {
                "SimulationsCount": req.simulations_count,
                "Expected_NPV_MUSD": mean_npv,
                "Std_NPV_MUSD": std_npv,
                "Expected_IRR_Pct": mean_irr,
                "Expected_ROI_Pct": mean_roi,
                "Std_ROI_Pct": std_roi,
                "ProbabilityOfLoss_Pct": prob_loss,
                "ValueAtRisk_95_MUSD": var_95_npv,
                "CI_NPV_Low": ci_npv_low,
                "CI_NPV_High": ci_npv_high,
                "CI_ROI_Low": ci_roi_low,
                "CI_ROI_High": ci_roi_high
            },
            "histogram": histogram_data,
            "rawRois": [float(x) for x in rois[:100]]  # Échantillon des 100 premiers tirages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
