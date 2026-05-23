import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Computes regression evaluation metrics: RMSE, MAE, MAPE (%), and R2.
    """
    y_true_arr = np.array(y_true).flatten()
    y_pred_arr = np.array(y_pred).flatten()
    
    # Root Mean Squared Error (RMSE)
    mse = mean_squared_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    
    # R-squared (R2)
    r2 = r2_score(y_true_arr, y_pred_arr)
    
    # Mean Absolute Percentage Error (MAPE)
    # Avoids division by zero by replacing 0 with small epsilon
    eps = np.finfo(float).eps
    mape = np.mean(np.abs((y_true_arr - y_pred_arr) / np.maximum(np.abs(y_true_arr), eps))) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def print_metrics_summary(y_true, y_pred, model_name="Model"):
    """Prints a clean text summary of model metrics."""
    metrics = calculate_metrics(y_true, y_pred)
    print(f"\n==================================================")
    print(f" Performance Metrics: {model_name}")
    print(f"==================================================")
    print(f"R-squared (R2) : {metrics['R2']:.4f}")
    print(f"RMSE           : {metrics['RMSE']:.2f}")
    print(f"MAE            : {metrics['MAE']:.2f}")
    print(f"MAPE           : {metrics['MAPE']:.2f}%")
    print(f"==================================================")
    return metrics

def compare_models(models_preds_dict, y_true):
    """
    Takes a dictionary of {model_name: predictions} and true values,
    computes metrics for all, and returns a sorted pandas DataFrame.
    """
    results = []
    for name, pred in models_preds_dict.items():
        metrics = calculate_metrics(y_true, pred)
        metrics['Model'] = name
        results.append(metrics)
        
    results_df = pd.DataFrame(results)
    # Reorder columns to put Model first
    cols = ['Model', 'R2', 'RMSE', 'MAE', 'MAPE']
    results_df = results_df[cols].sort_values(by='R2', ascending=False)
    return results_df

def save_metrics(metrics_df, output_path):
    """
    Saves the comparative metrics DataFrame to a CSV file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    print(f"[Metrics] Saved metrics report to {output_path}")
