"""
train.py - Autoresearch Training Script
---------------------------------------
C'est LE SEUL fichier que l'agent IA (vous) est autorisé à modifier.
Il contient le chargement des données (préparées par prepare.py), 
l'architecture du modèle, les hyperparamètres, et la boucle d'entraînement.

Votre objectif est de modifier le code de ce fichier pour minimiser `val_mape`.
"""

import os
import time
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()
    X_val = pd.read_csv(os.path.join(DATA_DIR, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(DATA_DIR, 'y_val.csv')).squeeze()
    return X_train, y_train, X_val, y_val

def main():
    t0 = time.time()
    
    # 1. Chargement des données
    X_train, y_train, X_val, y_val = load_data()
    
    # 2. Définition du Modèle & Hyperparamètres
    # TOUT EST MODIFIABLE ICI (Hyperparamètres, type de modèle, etc.)
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    # 3. Entraînement
    print(f"Training {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    
    # 4. Évaluation
    preds = model.predict(X_val)
    val_mape = mean_absolute_percentage_error(y_val, preds) * 100
    val_r2 = r2_score(y_val, preds)
    
    t1 = time.time()
    
    print("-" * 40)
    print(f"Time taken: {t1 - t0:.2f} seconds")
    print(f"val_r2:   {val_r2:.4f}")
    print(f"val_mape: {val_mape:.4f}%")
    print("-" * 40)

if __name__ == "__main__":
    # Vérification que prepare.py a bien été exécuté
    if not os.path.exists(os.path.join(DATA_DIR, 'X_train.csv')):
        print("Erreur : Les matrices de données sont introuvables. Exécutez 'python prepare.py' d'abord.")
        exit(1)
        
    main()
