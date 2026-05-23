import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Import the configured figures directory
from src.config import FIGURES_DIR

def set_style():
    """Sets standard styling for visualisations."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'figure.titlesize': 16,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

def clean_filename(title):
    """Converts a title to an ASCII-safe filename (no accents, no special chars, no spaces)."""
    t = title.lower()
    replacements = {
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'â': 'a', 'ä': 'a',
        'î': 'i', 'ï': 'i',
        'ô': 'o', 'ö': 'o',
        'û': 'u', 'ü': 'u',
        'ç': 'c',
        '&': 'and',
        "'": '',
        '(': '',
        ')': ''
    }
    for char, rep in replacements.items():
        t = t.replace(char, rep)
    t = t.replace(' ', '_')
    while '__' in t:
        t = t.replace('__', '_')
    return t.strip('_')

def plot_arrivals_evolution(df, title='Evolution of Arrivals over Time', save_path=None):
    """Plots arrivals over time, highlighting the COVID-19 period."""
    set_style()
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Arrivals'], color='teal', label='Arrivals', linewidth=1.5)
    
    # Highlight COVID
    covid_period = df[df['is_covid'] == 1]
    if len(covid_period) > 0:
        plt.scatter(covid_period['Date'], covid_period['Arrivals'], color='red', s=15, label='COVID Period (Override)')
        
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Number of Arrivals')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Auto-save default path
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, '01_arrivals_evolution.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_seasonal_decomposition(decomposition, title='Seasonal Decomposition', save_path=None):
    """Plots seasonal decomposition of a series (Observed, Trend, Seasonal, Residuals)."""
    set_style()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    ax1.plot(decomposition.observed, color='blue')
    ax1.set_ylabel('Observed')
    ax1.set_title(title)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(decomposition.trend, color='red')
    ax2.set_ylabel('Trend')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    ax3.plot(decomposition.seasonal, color='green')
    ax3.set_ylabel('Seasonal')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    ax4.plot(decomposition.resid.index, decomposition.resid, color='purple', alpha=0.7)
    ax4.axhline(0, color='black', linestyle='--', linewidth=1)
    ax4.set_ylabel('Residuals')
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Auto-save default path
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, '02_seasonal_decomposition.png')
        
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(df, columns, title='Correlation Matrix', save_path=None):
    """Plots a correlation heatmap for specified columns in a dataframe."""
    set_style()
    corr_columns = [c for c in columns if c in df.columns]
    corr_matrix = df[corr_columns].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    
    # Auto-save default path
    if save_path is None:
        clean_title = clean_filename(title)
        save_path = os.path.join(FIGURES_DIR, f'correlation_matrix_{clean_title}.png')
        
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_predictions_comparison(y_true, predictions_dict, dates, title='Model Predictions vs Actual Data', save_path=None):
    """Plots comparison of predicted values from multiple models against true test data."""
    set_style()
    plt.figure(figsize=(16, 8))
    
    # Plot true values
    plt.plot(dates, y_true, label='Actual Data', color='black', linewidth=2.0, alpha=0.7)
    
    # Plot each model prediction
    colors = ['green', 'blue', 'orange', 'purple', 'red', 'cyan', 'magenta', 'brown', 'gray']
    styles = ['--', '-.', ':', '--', '-', ':', '-.', '--', '-.']
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        plt.plot(dates, y_pred.flatten(), label=f'{model_name} Prediction', color=color, linestyle=style, linewidth=2.0)
        
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Arrivals')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Auto-save default path
    if save_path is None:
        clean_title = clean_filename(title)
        save_path = os.path.join(FIGURES_DIR, f'predictions_{clean_title}.png')
        
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_feature_importance(importance_series, title='Feature Importance', save_path=None):
    """
    Plots a bar chart showing feature importances.
    """
    set_style()
    plt.figure(figsize=(12, 8))
    top_n = importance_series.head(15)
    
    sns.barplot(x=top_n.values, y=top_n.index, palette='viridis')
    plt.title(title, fontweight='bold')
    plt.xlabel('Importance / Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Auto-save default path
    if save_path is None:
        clean_title = clean_filename(title)
        save_path = os.path.join(FIGURES_DIR, f'feature_importance_{clean_title}.png')
        
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_acf_pacf(series, lags=24, title='ACF and PACF Plots', save_path=None):
    """Plots Autocorrelation (ACF) and Partial Autocorrelation (PACF) functions."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # ACF
    plot_acf(series.dropna(), lags=lags, ax=ax1, color='teal')
    ax1.set_title(f"{title} - Autocorrelation (ACF)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # PACF
    plot_pacf(series.dropna(), lags=lags, ax=ax2, color='teal', method='ywm')
    ax2.set_title(f"{title} - Partial Autocorrelation (PACF)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Auto-save default path
    if save_path is None:
        clean_title = clean_filename(title)
        save_path = os.path.join(FIGURES_DIR, f'acf_pacf_{clean_title}.png')
        
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
