import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from src.roi_simulator import HotelROISimulator

router = APIRouter()

# Schémas
class RoiRequest(BaseModel):
    city: str = "Marrakech"
    rooms: int = 200
    capex_m: float = 150.0
    base_adr: float = 250.0
    base_occupancy: float = 0.65
    discount_rate: float = 0.08
    opex_margin: float = 0.65
    wc_adr_boost: float = 0.40
    wc_occ_boost: float = 0.15
    inflation_rate: float = 0.025
    wc_opex_inflation: float = 0.05
    enable_wc: bool = True

@router.get("/cities")
def get_cities_baseline():
    return {
        'Marrakech': {'capex': 150.0, 'adr': 250.0, 'part': 0.35, 'rec': 'Investir (Prioritaire, forte demande)'},
        'Casablanca': {'capex': 180.0, 'adr': 230.0, 'part': 0.20, 'rec': "Investir (Tourisme d'affaires)"},
        'Agadir': {'capex': 130.0, 'adr': 165.0, 'part': 0.18, 'rec': "À étudier (Saisonnier balnéaire)"},
        'Tanger': {'capex': 145.0, 'adr': 155.0, 'part': 0.10, 'rec': "Attendre (En développement rapide)"},
        'Rabat': {'capex': 165.0, 'adr': 175.0, 'part': 0.09, 'rec': "Attendre (Administratif haut de gamme)"},
        'Fès': {'capex': 120.0, 'adr': 135.0, 'part': 0.08, 'rec': "Éviter (Besoin d'infrastructures)"}
    }

@router.post("/simulate")
def simulate_roi(req: RoiRequest):
    try:
        # Instancier le simulateur
        sim = HotelROISimulator(
            rooms=req.rooms,
            investment_usd=req.capex_m * 1e6,
            opex_margin=req.opex_margin,
            discount_rate=req.discount_rate,
            base_occupancy=req.base_occupancy,
            wc_occupancy_2030=min(0.95, req.base_occupancy * (1 + req.wc_occ_boost)),
            base_adr=req.base_adr,
            wc_adr_boost_pct=req.wc_adr_boost,
            inflation_rate=req.inflation_rate,
            wc_opex_inflation=req.wc_opex_inflation
        )
        
        # Simuler 10 ans
        df_sim = sim.simulate_10years(start_year=2026)
        
        # Adapter les colonnes en fonction du paramètre enable_wc
        # Si Coupe du Monde désactivée, on remplace le scénario WC par le scénario base
        if not req.enable_wc:
            df_sim['ADR_WC_USD'] = df_sim['ADR_Base_USD']
            df_sim['Occ_WC'] = df_sim['Occ_Base']
            df_sim['Revenue_WC_USD'] = df_sim['Revenue_Base_USD']
            df_sim['GOP_WC_USD'] = df_sim['GOP_Base_USD']
            
        metrics = sim.calculate_metrics(df_sim)
        
        # Formater la table de cash flows
        records = []
        for idx, row in df_sim.iterrows():
            records.append({
                "Year": int(row['Year']),
                "Year_Index": int(row['Year_Index']),
                "ADR_Base": float(row['ADR_Base_USD']),
                "Occ_Base": float(row['Occ_Base'] * 100),
                "Revenue_Base": float(row['Revenue_Base_USD'] / 1e6), # MUSD
                "GOP_Base": float(row['GOP_Base_USD'] / 1e6),         # MUSD
                "ADR_WC": float(row['ADR_WC_USD']),
                "Occ_WC": float(row['Occ_WC'] * 100),
                "Revenue_WC": float(row['Revenue_WC_USD'] / 1e6),     # MUSD
                "GOP_WC": float(row['GOP_WC_USD'] / 1e6)              # MUSD
            })
            
        return {
            "metrics": {
                "NPV_Base": float(metrics["NPV_Base_MUSD"]),
                "NPV_WC": float(metrics["NPV_WC_MUSD"]),
                "IRR_Base": float(metrics["IRR_Base_Pct"]) if not np.isnan(metrics["IRR_Base_Pct"]) else None,
                "IRR_WC": float(metrics["IRR_WC_Pct"]) if not np.isnan(metrics["IRR_WC_Pct"]) else None,
                "Payback_Base": int(metrics["Payback_Base_Years"]) if not np.isnan(metrics["Payback_Base_Years"]) else None,
                "Payback_WC": int(metrics["Payback_WC_Years"]) if not np.isnan(metrics["Payback_WC_Years"]) else None,
                "ROI_Base": float(metrics["ROI_Base_Pct"]),
                "ROI_WC": float(metrics["ROI_WC_Pct"])
            },
            "cashFlows": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
