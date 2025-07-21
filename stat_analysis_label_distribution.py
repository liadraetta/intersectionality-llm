import pathlib
import pandas as pd
from collections import Counter
from utils.label_distribution_analysis import read_all_model_datasets, filter_out_missing_rows, compute_intersectionality_cohen_kappa, visualize_heatmap

def main():
    model_names = ['Qwen2', 'deepseek', 'Llama', 'gemma', 'Ministral']
    cot_names = ['noCoT']
    socio_demographic_variables = ['gender', 'race', 'political']
    # Input paths
    results_base_path = pathlib.Path('predictions/cleaned')
    
    for model_name in model_names:
        results_socdem_base_path = pathlib.Path(f"predictions_dem/{model_name}/cleaned")
        for cot_name in cot_names:
            output_plot_path = pathlib.Path(f'statistical_analysis/ablation_label_distribution/{model_name}_{cot_name}.png')
            print(f"Model: {model_name}, CoT: {cot_name}")
            all_model_datasets = read_all_model_datasets(model_name, cot_name, results_base_path, results_socdem_base_path)
            all_model_datasets_filtered = filter_out_missing_rows(all_model_datasets)
            all_model_results_cohen = compute_intersectionality_cohen_kappa(all_model_datasets_filtered, socio_demographic_variables)
            visualize_heatmap(all_model_results_cohen, save_path=output_plot_path, model_name=model_name)



if __name__ == "__main__":
    main()