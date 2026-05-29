import nbformat
import re

def inject_autoresearch(filepath):
    try:
        nb = nbformat.read(filepath, as_version=4)
        
        # Check if already injected
        if "AutoResearchEvaluator" in nb.cells[0].source:
            print(f"Already injected in {filepath}")
            return
            
        import_cell = nbformat.v4.new_code_cell(
            "import sys\n"
            "sys.path.append('../src')\n"
            "from autoresearch import AutoResearchEvaluator\n"
            "auto_eval = AutoResearchEvaluator(output_dir='results/autoresearch_output')\n"
        )
        nb.cells.insert(0, import_cell)

        # Replace in standard ML loop
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if 'for name, model in ml_models.items():' in cell.source:
                    new_source = re.sub(
                        r"# Metrics\n\s*r2 = r2_score.*?(?=all_results\.append)",
                        "target_str = 'Arrivals' if 'Arrivals' in df_y_test.columns[0] else 'Nights' if 'Nights' in df_y_test.columns[0] else df_y_test.columns[0]\n    res_dict = auto_eval.evaluate_model(target_name=target_str, model_name=name, y_true=y_test, y_pred=preds, is_walk_forward=False)\n    r2 = res_dict['R2']\n    rmse = res_dict['RMSE']\n    mae = res_dict['MAE']\n    mape = res_dict['MAPE (%)']\n    \n    ",
                        cell.source, flags=re.DOTALL
                    )
                    cell.source = new_source

                if 'def walk_forward_evaluate' in cell.source:
                    new_source = re.sub(
                        r"r2 = r2_score.*?(?=all_results\.append)",
                        "target_str = 'Arrivals' if 'Arrivals' in df_y_test.columns[0] else 'Nights' if 'Nights' in df_y_test.columns[0] else df_y_test.columns[0]\n    res_dict = auto_eval.evaluate_model(target_name=target_str, model_name=model_name, y_true=y_true_aligned, y_pred=y_pred_finales, is_walk_forward=True)\n    r2 = res_dict['R2']\n    rmse = res_dict['RMSE']\n    mae = res_dict['MAE']\n    mape = res_dict['MAPE (%)']\n    \n    ",
                        cell.source, flags=re.DOTALL
                    )
                    cell.source = new_source

        report_cell = nbformat.v4.new_code_cell(
            "auto_eval.generate_report()\n"
            "print('AutoResearch report generated successfully.')\n"
        )
        nb.cells.append(report_cell)
        
        nbformat.write(nb, filepath)
        print(f"Successfully injected AutoResearch in {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

inject_autoresearch('notebooks/04_machine_learning.ipynb')
if __import__('os').path.exists('notebooks/05_deep_learning.ipynb'):
    inject_autoresearch('notebooks/05_deep_learning.ipynb')
