import pandas as pd
import numpy as np

def clean_morocco_data(input_file, output_file):
    print(f"Chargement de {input_file}...")
    
    # Chargement du CSV
    # On suppose que la première colonne est la date
    df = pd.read_csv(input_file)
    
    # Renommer la première colonne en 'Date' si ce n'est pas déjà fait
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    
    # Conversion en datetime et indexation
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Mapping des colonnes (Nom cible : Nom dans le CSV)
    mapping = {
        'InterTourismeReceipts': 'InterTourismeReceipts(usd)',
        'REER': 'REER(2010 = 100)',
        'Oil_price': 'brent_oil_prices(USD/barrel)',
        'FDI': 'IDE(USD)',
        'Unemployment_rate': 'Chômage, total (% de la population)',
        'Poverty_rate': 'Pauvrete'
    }
    
    # Extraction et renommage
    available_cols = {}
    for target, source in mapping.items():
        if source in df.columns:
            available_cols[source] = target
        else:
            # Recherche floue si le nom exact n'est pas trouvé
            matches = [c for c in df.columns if target.lower() in c.lower() or (source and source.lower() in c.lower())]
            if matches:
                available_cols[matches[0]] = target
                print(f"Trouvé '{matches[0]}' pour '{target}'")
    
    df_clean = df[list(available_cols.keys())].rename(columns=available_cols)
    
    # Gestion spécifique de GDP_Construction (souvent absent des exports simplifiés)
    construction_matches = [c for c in df.columns if 'construct' in c.lower() or 'pib' in c.lower()]
    if 'GDP_Construction' not in df_clean.columns:
        if construction_matches:
            df_clean['GDP_Construction'] = df[construction_matches[0]]
            print(f"Utilisation de '{construction_matches[0]}' pour GDP_Construction")
        else:
            df_clean['GDP_Construction'] = np.nan
            print("Attention : GDP_Construction non trouvée, colonne créée avec des NaN.")

    # Nettoyage des données
    # Dans votre fichier, beaucoup de 0.0 semblent être des valeurs manquantes
    df_clean = df_clean.replace(0.0, np.nan)
    
    # Interpolation pour combler les mois vides (données annuelles -> mensuelles)
    df_clean = df_clean.interpolate(method='time').ffill().bfill()
    
    # Exportation
    df_clean.to_csv(output_file)
    print(f"Fichier nettoyé sauvegardé sous : {output_file}")
    print("\nAperçu des données :")
    print(df_clean.head())

if __name__ == "__main__":
    clean_morocco_data('Morocco.csv', 'Morocco_cleaned.csv')
