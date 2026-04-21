import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================
# 0. CONFIGURATION ET STYLES
# =================================================================
# Forcer Pandas à afficher toutes les colonnes
pd.set_option('display.max_columns', None)
sns.set_theme(style="whitegrid") # Style propre pour les graphiques
plt.rcParams['figure.figsize'] = (15, 8)

# Profil de saisonnalité (part mensuelle des indicateurs annuels)
# Basé sur les poids définis dans le notebook original
anualiate = [0.06, 0.06, 0.08, 0.09, 0.09, 0.10, 0.12, 0.12, 0.09, 0.08, 0.06, 0.05]
profil_saisonnalite = dict(zip(range(1, 13), anualiate))

# =================================================================
# 1. FONCTIONS DE TRANSFORMATION
# =================================================================

def mensualiser_ligne_complete(ligne, profil_dict, colonnes):
    """
    Désagrège une ligne annuelle en 12 lignes mensuelles selon un profil.
    Utilisé pour les données de la Banque Mondiale.
    """
    annee = int(ligne['year'])
    lignes_mensuelles = []
    
    for mois in range(1, 13):
        poids = profil_dict[mois]
        
        # Structure de base de la ligne mensuelle
        nouvelle_ligne = {
            'Year': annee,
            'Month': mois
        }
        
        # Application du pourcentage à chaque indicateur
        for col in colonnes:
            val_annuelle = ligne[col]
            if pd.notna(val_annuelle):
                nouvelle_ligne[col] = val_annuelle * poids
            else:
                nouvelle_ligne[col] = np.nan
                
        lignes_mensuelles.append(nouvelle_ligne)
        
    return lignes_mensuelles

def mensualiser_recettes_mdh(df_annuel, col_annee, col_recettes, profil_dict):
    """
    Désagrège les recettes annuelles en recettes mensuelles (MDH).
    """
    lignes_mensuelles = []
    
    for _, ligne in df_annuel.iterrows():
        annee = int(ligne[col_annee])
        recette_annuelle = ligne[col_recettes]
        
        for mois in range(1, 13):
            recette_du_mois = recette_annuelle * profil_dict[mois]
            
            lignes_mensuelles.append({
                'Year': annee,
                'Month': mois,
                'Recettes_Mensuelles_MDH': round(recette_du_mois, 2) 
            })
            
    return pd.DataFrame(lignes_mensuelles)

# =================================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =================================================================

print("--- 1/5 : Chargement des fichiers sources ---")

# 2.1 Données mensuelles (Postes frontières / EHTC)
try:
    df_monthly = pd.read_csv("tourism_maroc_monthly_data.csv")
except FileNotFoundError:
    print("Attention: tourism_maroc_monthly_data.csv non trouvé.")
    df_monthly = pd.DataFrame(columns=['Year', 'Month', 'Arrivals'])

# 2.2 Données World Bank (Historique long)
try:
    df_wb = pd.read_csv("morocco_tourism_worldbank.csv")
    cols_wb = ['arrivals', 'receipts_usd', 'expenditure_usd']
    # Nettoyage des années vides
    df_wb_clean = df_wb.dropna(subset=cols_wb, how='all')

    # Mensualisation de la World Bank
    toutes_les_lignes_wb = []
    for _, ligne in df_wb_clean.iterrows():
        toutes_les_lignes_wb.extend(mensualiser_ligne_complete(ligne, profil_saisonnalite, cols_wb))
    df_wb_mensuel = pd.DataFrame(toutes_les_lignes_wb)
    
    # Conversion en MDH (Million Dirhams) approximatif via le facteur du notebook
    df_wb_mensuel["receipts_MHD"] = df_wb_mensuel["receipts_usd"] * 1e-5
    df_wb_mensuel["expenditure_MHD"] = df_wb_mensuel["expenditure_usd"] * 1e-5
except FileNotFoundError:
    print("Attention: morocco_tourism_worldbank.csv non trouvé.")
    df_wb_mensuel = pd.DataFrame(columns=['Year', 'Month', 'arrivals'])

# 2.3 Données de Recettes Historiques (MDH)
data_historique_recettes = {
    'Annee': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2023, 2024, 2025],
    'Recettes_Annuelles_MDH': [
        56421.5, 58904.0, 57835.0, 57614.0, 62034.0, 
        61150.0, 64226.0, 72127.0, 73039.0, 78747.0, 
        36364.0, 107562.0, 114475.0, 138101.0
    ]
}
df_recettes_annuelles = pd.DataFrame(data_historique_recettes)
df_recettes_mensuelles_mdh = mensualiser_recettes_mdh(df_recettes_annuelles, 'Annee', 'Recettes_Annuelles_MDH', profil_saisonnalite)

# =================================================================
# 3. HARMONISATION ET FILTRAGE (EXCLUSION 2020)
# =================================================================

print("--- 2/5 : Harmonisation et exclusion de 2020 ---")

# Renommage des arrivées pour identifier les sources
df_monthly = df_monthly.rename(columns={'Arrivals': 'arr_1'})
df_wb_mensuel = df_wb_mensuel.rename(columns={'arrivals': 'arr_2'})

# Action demandée : Exclusion de l'année 2020 pour éviter les biais COVID
df_monthly = df_monthly[df_monthly['Year'] != 2020]
df_wb_mensuel = df_wb_mensuel[df_wb_mensuel['Year'] != 2020]
df_recettes_mensuelles_mdh = df_recettes_mensuelles_mdh[df_recettes_mensuelles_mdh['Year'] != 2020]

# Suppression des colonnes Date disparates pour fusionner proprement
for d in [df_monthly, df_wb_mensuel, df_recettes_mensuelles_mdh]:
    if 'Date' in d.columns:
        d.drop(columns=['Date'], inplace=True)

# =================================================================
# 4. FUSION ET CRÉATION DU CALENDRIER CONTINU
# =================================================================

print("--- 3/5 : Fusion et création du calendrier continu ---")

# Fusion des sources disponibles
df_tmp = pd.merge(df_monthly, df_wb_mensuel, on=['Year', 'Month'], how='outer')
df_combined = pd.merge(df_tmp, df_recettes_mensuelles_mdh, on=['Year', 'Month'], how='outer')

# Création de la colonne unique 'Arrivals' avec priorité aux données locales (EHTC)
df_combined['Arrivals'] = df_combined['arr_1'].fillna(df_combined['arr_2'])
df_combined.rename(columns={'arr_1': 'Arrivals_EHTC', 'arr_2': 'Arrivals_WB'}, inplace=True)

# Génération d'un calendrier continu (Maître) pour boucher les trous (ex: 2021)
y_min = int(df_combined['Year'].min())
y_max = int(df_combined['Year'].max())
dates = pd.date_range(start=f"{y_min}-01-01", end=f"{y_max}-12-01", freq='MS')

df_maitre = pd.DataFrame({
    'Date': dates,
    'Year': dates.year,
    'Month': dates.month
})

# Fusion finale sur le calendrier : le 'Left join' maintient tous les mois même vides
df_final = pd.merge(df_maitre, df_combined, on=['Year', 'Month'], how='left')

# Nettoyage des colonnes techniques et réorganisation
cols_to_drop = ['receipts_usd', 'expenditure_usd', 'expenditure_MHD']
df_final.drop(columns=cols_to_drop, inplace=True, errors='ignore')

cols_priority = ['Date', 'Year', 'Month', 'Arrivals', 'Arrivals_EHTC', 'Arrivals_WB']
cols_others = [c for c in df_final.columns if c not in cols_priority]
df_final = df_final[cols_priority + cols_others]

# Tri final
df_final = df_final.sort_values('Date').reset_index(drop=True)

# =================================================================
# 5. EXPORTS ET VALIDATION
# =================================================================

print("--- 4/5 : Sauvegarde du dataset nettoyé ---")
nom_csv = 'maroc_tourism_final_clean.csv'
df_final.to_csv(nom_csv, index=False, encoding='utf-8-sig')
print(f"✅ Dataset enregistré sous : {nom_csv}")

print("--- 5/5 : Génération du graphique de comparaison ---")
fig, ax1 = plt.subplots()

# Ligne principale fusionnée
ax1.plot(df_final['Date'], df_final['Arrivals'], 
         color='#008080', linewidth=3, label='Arrivals (Fusionnée)')

# Sources individuelles pour vérification
ax1.plot(df_final['Date'], df_final['Arrivals_EHTC'], 
         color='#FF8C00', linewidth=1.5, linestyle=':', alpha=0.8, label='Source EHTC (Local)')

ax1.plot(df_final['Date'], df_final['Arrivals_WB'], 
         color='#00008B', linewidth=1, linestyle='--', alpha=0.7, label='Source WB (World Bank)')

# Mise en évidence de la zone COVID exclue/NaN (2020-2021)
plt.axvspan('2020-01-01', '2021-12-31', color='gray', alpha=0.15, label='Période COVID (Exclue/NaN)')

plt.title('Arrivées Touristiques au Maroc : Synthèse Multi-Sources 2010-2025', fontsize=16, pad=15)
plt.xlabel('Chronologie', fontsize=12)
plt.ylabel('Nombre d\'Arrivées', fontsize=12)
plt.legend(loc='best')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
fig.tight_layout()

nom_image = "comparaison_sources_arrivées.png"
plt.savefig(nom_image, dpi=300, bbox_inches='tight')
print(f"✅ Graphique enregistré sous : {nom_image}")

# Affichage du résultat final
print("\n--- Aperçu des dernières lignes (2025) ---")
print(df_final.tail(5))
