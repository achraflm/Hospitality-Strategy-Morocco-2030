"""
Module de Simulation ROI Hôtelier pour la Coupe du Monde 2030
============================================================

Ce module contient la classe HotelROISimulator, qui modélise et simule la performance
financière d'un investissement hôtelier sur 10 ans au Maroc en comparant un scénario de base
avec un scénario bénéficiant de l'impact de la Coupe du Monde FIFA 2030.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class HotelROISimulator:
    def __init__(self, rooms=200, investment_usd=40000000.0, opex_margin=0.65, 
                 discount_rate=0.08, base_occupancy=0.68, wc_occupancy_2030=0.75,
                 base_adr=250.0, wc_adr_boost_pct=0.40, inflation_rate=0.025):
        """
        Initialise le simulateur avec les paramètres du projet.
        
        Args:
            rooms (int): Nombre de chambres de l'hôtel.
            investment_usd (float): Coût d'acquisition / construction initial en USD.
            opex_margin (float): Ratio des coûts opérationnels (ex: 0.65 pour 65% opex, 35% GOP).
            discount_rate (float): Taux d'actualisation (WACC) pour le calcul de la NPV (ex: 0.08).
            base_occupancy (float): Taux d'occupation annuel moyen hors Coupe du Monde (ex: 0.68).
            wc_occupancy_2030 (float): Taux d'occupation en 2030 avec Coupe du Monde (ex: 0.75).
            base_adr (float): Tarif journalier moyen initial en USD (ADR).
            wc_adr_boost_pct (float): Boost sur l'ADR en 2030 dû à la Coupe du Monde (ex: 0.40 pour +40%).
            inflation_rate (float): Taux d'inflation annuel moyen appliqué à l'ADR (ex: 0.025).
        """
        self.rooms = rooms
        self.investment_usd = investment_usd
        self.opex_margin = opex_margin
        self.discount_rate = discount_rate
        self.base_occupancy = base_occupancy
        self.wc_occupancy_2030 = wc_occupancy_2030
        self.base_adr = base_adr
        self.wc_adr_boost_pct = wc_adr_boost_pct
        self.inflation_rate = inflation_rate

    def simulate_10years(self, start_year=2026):
        """
        Génère les projections annuelles de revenus, dépenses et GOP sur 10 ans.
        
        Returns:
            pd.DataFrame: Table contenant les cash flows détaillés pour chaque année.
        """
        years = list(range(start_year, start_year + 10))
        records = []
        
        for idx, year in enumerate(years):
            t = year - start_year
            
            # 1. Inflation cumulée de l'ADR
            adr_base = self.base_adr * ((1 + self.inflation_rate) ** t)
            occ_base = self.base_occupancy
            
            # Scénario A (Base)
            rev_base = self.rooms * occ_base * 365 * adr_base
            gop_base = rev_base * (1 - self.opex_margin)
            
            # Scénario B (World Cup 2030 Boost)
            if year == 2030:
                occ_wc = self.wc_occupancy_2030
                adr_wc = adr_base * (1 + self.wc_adr_boost_pct)
            else:
                occ_wc = self.base_occupancy
                adr_wc = adr_base
                
            rev_wc = self.rooms * occ_wc * 365 * adr_wc
            gop_wc = rev_wc * (1 - self.opex_margin)
            
            records.append({
                'Year': year,
                'Year_Index': idx + 1,
                'ADR_Base_USD': adr_base,
                'Occ_Base': occ_base,
                'Revenue_Base_USD': rev_base,
                'GOP_Base_USD': gop_base,
                'ADR_WC_USD': adr_wc,
                'Occ_WC': occ_wc,
                'Revenue_WC_USD': rev_wc,
                'GOP_WC_USD': gop_wc
            })
            
        return pd.DataFrame(records)

    def calculate_metrics(self, df_roi):
        """
        Calcule les indicateurs financiers : NPV, IRR, Payback Period et ROI cumulé.
        
        Args:
            df_roi (pd.DataFrame): Table générée par simulate_10years.
            
        Returns:
            dict: Métriques financières calculées pour les deux scénarios.
        """
        # Construction des vecteurs de Cash Flow
        cf_base = [-self.investment_usd] + df_roi['GOP_Base_USD'].tolist()
        cf_wc = [-self.investment_usd] + df_roi['GOP_WC_USD'].tolist()
        
        # 1. Net Present Value (NPV)
        npv_base = sum(val / ((1 + self.discount_rate) ** idx) for idx, val in enumerate(cf_base))
        npv_wc = sum(val / ((1 + self.discount_rate) ** idx) for idx, val in enumerate(cf_wc))
        
        # 2. Internal Rate of Return (IRR)
        def calculate_irr(cash_flows):
            for r in np.linspace(-0.2, 0.6, 25000):
                npv = sum(val / ((1 + r) ** idx) for idx, val in enumerate(cash_flows))
                if abs(npv) < 100.0:  # Précision tolérée
                    return r * 100
            return np.nan
            
        irr_base = calculate_irr(cf_base)
        irr_wc = calculate_irr(cf_wc)
        
        # 3. Payback Period (Année de retour sur investissement)
        def get_payback(cash_flows):
            cum_sum = np.cumsum(cash_flows)
            # Trouver à quel index la somme cumulée franchit 0
            for idx, val in enumerate(cum_sum):
                if val >= 0:
                    return idx  # correspond à l'année du projet
            return np.nan
            
        payback_base = get_payback(cf_base)
        payback_wc = get_payback(cf_wc)
        
        # 4. ROI Cumulé sur 10 ans (%)
        roi_base = (df_roi['GOP_Base_USD'].sum() - self.investment_usd) / self.investment_usd * 100
        roi_wc = (df_roi['GOP_WC_USD'].sum() - self.investment_usd) / self.investment_usd * 100
        
        return {
            'NPV_Base_MUSD': npv_base / 1e6,
            'NPV_WC_MUSD': npv_wc / 1e6,
            'IRR_Base_Pct': irr_base,
            'IRR_WC_Pct': irr_wc,
            'Payback_Base_Years': payback_base,
            'Payback_WC_Years': payback_wc,
            'ROI_Base_Pct': roi_base,
            'ROI_WC_Pct': roi_wc
        }

    def plot_comparison(self, df_roi, metrics, save_path=None):
        """
        Génère un graphique comparant le Cash Flow cumulé sous les deux scénarios.
        """
        cf_base = [-self.investment_usd] + df_roi['GOP_Base_USD'].tolist()
        cf_wc = [-self.investment_usd] + df_roi['GOP_WC_USD'].tolist()
        
        cum_base = np.cumsum(cf_base) / 1e6 # en millions
        cum_wc = np.cumsum(cf_wc) / 1e6
        
        years = [df_roi['Year'].iloc[0] - 1] + df_roi['Year'].tolist()
        
        plt.figure(figsize=(10, 6))
        plt.plot(years, cum_base, marker='o', label=f"Base (NPV={metrics['NPV_Base_MUSD']:.1f}M$, IRR={metrics['IRR_Base_Pct']:.1f}%)", color='gray', linestyle='--')
        plt.plot(years, cum_wc, marker='s', label=f"Coupe du Monde (NPV={metrics['NPV_WC_MUSD']:.1f}M$, IRR={metrics['IRR_WC_Pct']:.1f}%)", color='teal', linewidth=2)
        
        plt.axhline(0, color='black', linestyle=':', alpha=0.7)
        plt.title("Évolution des Cash Flows Cumulés - Hôtel 5★ Marrakech", fontsize=14, fontweight='bold')
        plt.xlabel("Année")
        plt.ylabel("Profit Cumulé (Millions USD)")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"[ROI Plot] Graphique sauvegardé -> {save_path}")
        plt.close()

    def simulate_with_forecast(self, annual_arrivals_dict, baseline_arrivals_2025, start_year=2026, wc_boost_2030=True, duration_years=10):
        """
        Simule la performance financière sur 10 ans où le taux d'occupation de l'hôtel
        suit la croissance relative des arrivées touristiques nationales prédites par un modèle.
        """
        years = list(range(start_year, start_year + duration_years))
        records = []
        
        for idx, year in enumerate(years):
            t = year - start_year
            
            # Tarif journalier (ADR) indexé sur l'inflation
            adr = self.base_adr * ((1 + self.inflation_rate) ** t)
            
            # Croissance relative par rapport à l'année de référence (2025)
            projected_arrivals = annual_arrivals_dict.get(year, baseline_arrivals_2025)
            g_t = projected_arrivals / baseline_arrivals_2025
            
            # Ajustement du taux d'occupation par rapport au coefficient de croissance
            occ = min(0.95, self.base_occupancy * g_t)
            
            # Application du boost supplémentaire en 2030 s'il est activé
            if year == 2030 and wc_boost_2030:
                occ = min(0.95, occ * 1.15)  # +15% de boost d'occupation relatif
                adr = adr * (1 + self.wc_adr_boost_pct)  # boost ADR de la Coupe du Monde
                
            rev = self.rooms * occ * 365 * adr
            gop = rev * (1 - self.opex_margin)
            
            records.append({
                'Year': year,
                'Year_Index': idx + 1,
                'ADR_USD': adr,
                'Occ': occ,
                'Revenue_USD': rev,
                'GOP_USD': gop
            })
            
        return pd.DataFrame(records)

    def simulate_with_nuitees_forecast(self, annual_nights_dict, start_year=2026, wc_boost_2030=True, duration_years=10):
        """
        Simule la performance financière sur 10 ans en utilisant directement les
        **nuitées annuelles prédites** (overnight stays) pour déduire le taux d'occupation.

        Contrairement à ``simulate_with_forecast`` (basée sur les arrivées + ratio de croissance),
        cette méthode calcule l'occupation de façon directe et plus précise :

        .. math::

            \\text{Occ}_t = \\min\\left(0.95,\\ \\frac{\\hat{N}_t}{\\text{Chambres} \\times 365}\\right)

        où :math:`\\hat{N}_t` est le nombre de nuitées annuelles prédites.

        Args:
            annual_nights_dict (dict): Dictionnaire {année: nuitées_annuelles_prédites}.
            start_year (int): Première année de la simulation (défaut : 2026).
            wc_boost_2030 (bool): Activer le boost Coupe du Monde FIFA 2030 (défaut : True).
            duration_years (int): Durée de la simulation en années (défaut : 10).

        Returns:
            pd.DataFrame: Table des cash flows annuels (ADR_USD, Occ, Revenue_USD, GOP_USD,
                          Nights_Predicted, RevPAR_USD).
        """
        years = list(range(start_year, start_year + duration_years))
        total_room_nights = self.rooms * 365  # capacité annuelle totale de l'hôtel
        records = []

        for idx, year in enumerate(years):
            t = year - start_year

            # ADR indexé sur l'inflation annuelle
            adr = self.base_adr * ((1 + self.inflation_rate) ** t)

            # Taux d'occupation déduit directement des nuitées prédites
            predicted_nights = annual_nights_dict.get(year, total_room_nights * self.base_occupancy)
            occ = min(0.95, predicted_nights / total_room_nights)

            # Boost Coupe du Monde 2030 (ADR uniquement — l'occupation est déjà modélisée)
            if year == 2030 and wc_boost_2030:
                adr = adr * (1 + self.wc_adr_boost_pct)

            rev = self.rooms * occ * 365 * adr
            gop = rev * (1 - self.opex_margin)

            # RevPAR = Revenu par chambre disponible par nuit
            revpar = occ * adr

            records.append({
                'Year': year,
                'Year_Index': idx + 1,
                'ADR_USD': adr,
                'Occ': occ,
                'Nights_Predicted': predicted_nights,
                'RevPAR_USD': revpar,
                'Revenue_USD': rev,
                'GOP_USD': gop
            })

        return pd.DataFrame(records)


    def calculate_metrics_for_gop(self, gop_list):
        """
        Calcule les indicateurs financiers sur la base d'une série de cash flows GOP.
        """
        cf = [-self.investment_usd] + list(gop_list)
        
        # 1. Net Present Value (NPV)
        npv = sum(val / ((1 + self.discount_rate) ** idx) for idx, val in enumerate(cf))
        
        # 2. Internal Rate of Return (IRR)
        def calculate_irr(cash_flows):
            for r in np.linspace(-0.2, 0.6, 25000):
                npv_r = sum(val / ((1 + r) ** idx) for idx, val in enumerate(cash_flows))
                if abs(npv_r) < 100.0:
                    return r * 100
            return np.nan
            
        irr = calculate_irr(cf)
        
        # 3. Payback Period
        def get_payback(cash_flows):
            cum_sum = np.cumsum(cash_flows)
            for idx, val in enumerate(cum_sum):
                if val >= 0:
                    return idx
            return np.nan
            
        payback = get_payback(cf)
        
        # 4. ROI Cumulé (%)
        roi = (sum(gop_list) - self.investment_usd) / self.investment_usd * 100
        
        return {
            'NPV_MUSD': npv / 1e6,
            'IRR_Pct': irr,
            'Payback_Years': payback,
            'ROI_Pct': roi
        }
