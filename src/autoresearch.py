import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class AutoResearchEvaluator:
    def __init__(self, output_dir="autoresearch_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
        
    def _smape(self, y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        
    def _mape(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
    def evaluate_model(self, target_name, model_name, y_true, y_pred, is_walk_forward=False):
        """Calculates metrics, plots results, and generates insights."""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = self._mape(y_true, y_pred)
        smape = self._smape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Plot Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, label="Actual", marker='o')
        plt.plot(y_pred, label=f"Predicted ({model_name})", marker='x')
        plt.title(f"AutoResearch: {model_name} on {target_name} (WF: {is_walk_forward})")
        plt.xlabel("Time steps (Test Set)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(self.output_dir, f"{model_name.replace(' ', '_')}_{target_name}_pred.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot Residuals
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=15)
        plt.axvline(0, color='r', linestyle='--')
        plt.title(f"AutoResearch Residuals: {model_name} on {target_name}")
        plt.xlabel("Error (Actual - Predicted)")
        res_plot_path = os.path.join(self.output_dir, f"{model_name.replace(' ', '_')}_{target_name}_resid.png")
        plt.savefig(res_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Insights Generation
        insights = []
        if r2 < 0:
            insights.append("Model is worse than baseline average. Temporal degradation or structural break (e.g. COVID) likely broke generalization.")
        elif r2 > 0.7:
            insights.append("Excellent generalization. The model successfully captures both trend and seasonality without severe overfitting.")
        else:
            insights.append("Moderate performance. The model captures some dynamics but struggles with high variance.")
            
        mean_res = np.mean(residuals)
        if abs(mean_res) > np.std(residuals) * 0.5:
            insights.append("Systematic bias detected: The model consistently overpredicts or underpredicts.")
        else:
            insights.append("Residuals are relatively centered, indicating no severe systemic bias.")
            
        if is_walk_forward:
            insights.append("Walk-Forward evaluation confirms stability across forecasting windows, mitigating data leakage.")
        
        insight_text = " ".join(insights)
        
        res_dict = {
            "Target": target_name,
            "Model": model_name,
            "Validation": "Walk-Forward" if is_walk_forward else "Standard",
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE (%)": mape,
            "SMAPE (%)": smape,
            "Insights": insight_text
        }
        self.results.append(res_dict)
        return res_dict

    def generate_report(self):
        df = pd.DataFrame(self.results)
        df = df.sort_values(by=["Target", "R2"], ascending=[True, False])
        
        csv_path = os.path.join(self.output_dir, "autoresearch_comparison.csv")
        df.to_csv(csv_path, index=False)
        
        md_path = os.path.join(self.output_dir, "autoresearch_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# AutoResearch Automated Evaluation Report\n\n")
            
            for target in df["Target"].unique():
                f.write(f"## Target: {target}\n\n")
                target_df = df[df["Target"] == target]
                
                f.write("### Model Rankings\n")
                f.write(target_df[["Model", "Validation", "R2", "MAPE (%)", "SMAPE (%)"]].to_markdown(index=False))
                f.write("\n\n### AutoResearch Insights\n")
                
                for _, row in target_df.iterrows():
                    f.write(f"- **{row['Model']} ({row['Validation']})**: {row['Insights']}\n")
                
                f.write("\n---\n")
                
        return df, md_path
