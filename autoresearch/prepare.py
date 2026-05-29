"""
prepare.py - Autoresearch Data Preparation
------------------------------------------
Ce script est exécuté une seule fois pour préparer les données d'entraînement.
Il fige les variables, nettoie le dataset et exporte des matrices CSV légères 
afin que le script `train.py` puisse s'exécuter en quelques millisecondes 
lors des itérations de l'agent IA.

NE DOIT PAS ÊTRE MODIFIÉ PAR L'AGENT.
"""

import os
import sys
import pandas as pd

# Ajouter le root path pour importer les modules du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.data_loader as loader
import src.cleaning as cleaner
import src.features as feat
from src.config import TARGET_COL

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def prepare_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Loading and cleaning data...")
    df = loader.load_and_merge_tourism_data()
    df = cleaner.integrate_covid_data(df)
    df = cleaner.reconstruct_historical_arrivals(df)
    
    print("Building features...")
    df_featured = feat.build_features(df)
    df_ml = df_featured.dropna(subset=[TARGET_COL]).copy()
    
    # Train/Val Split temporel (ex: Train jusqu'en 2022, Val 2023-2025)
    df_train = df_ml[df_ml['Date'].dt.year <= 2022].copy()
    df_val = df_ml[(df_ml['Date'].dt.year > 2022) & (df_ml['Date'].dt.year <= 2025)].copy()
    
    # Sélection des features
    valid_features = [f for f in feat.get_feature_list() if f in df_ml.columns]
    
    X_train = df_train[valid_features].fillna(df_train[valid_features].median())
    y_train = df_train[TARGET_COL]
    
    X_val = df_val[valid_features].fillna(df_train[valid_features].median())
    y_val = df_val[TARGET_COL]
    
    print(f"Exporting matrices to {DATA_DIR}...")
    X_train.to_csv(os.path.join(DATA_DIR, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), index=False)
    X_val.to_csv(os.path.join(DATA_DIR, 'X_val.csv'), index=False)
    y_val.to_csv(os.path.join(DATA_DIR, 'y_val.csv'), index=False)
    
    print("Data preparation complete!")
    print(f"Train size: {X_train.shape}, Val size: {X_val.shape}")

if __name__ == "__main__":
    prepare_data()
