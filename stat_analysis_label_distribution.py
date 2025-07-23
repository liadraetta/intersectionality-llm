import pathlib
import pandas as pd
from collections import Counter
from utils.label_distribution_analysis import read_all_model_datasets, filter_out_missing_rows, compute_intersectionality_cohen_kappa
from utils.label_distribution_analysis import visualize_heatmap, visualize_combined_heatmap

def main():
    model_names = ['Llama', 'deepseek', 'Qwen2', 'gemma', 'Ministral']
    cot_names = ['noCoT']
    socio_demographic_variables = ['gender', 'race', 'political']
    # Input paths
    results_base_path = pathlib.Path('predictions/cleaned')
    all_models_data = {}
    for model_name in model_names:
        results_socdem_base_path = pathlib.Path(f"predictions_dem/{model_name}/cleaned")
        for cot_name in cot_names:
            output_plot_path_cohen = pathlib.Path(f'statistical_analysis/ablation_label_distribution/cohen/{model_name}_{cot_name}.png')
            output_plot_path_match_perc = pathlib.Path(f'statistical_analysis/ablation_label_distribution/match_perc/{model_name}_{cot_name}.png')
            print(f"Model: {model_name}, CoT: {cot_name}")
            all_model_datasets = read_all_model_datasets(model_name, cot_name, results_base_path, results_socdem_base_path)
            all_model_datasets_filtered = filter_out_missing_rows(all_model_datasets)
            all_model_results_cohen = compute_intersectionality_cohen_kappa(all_model_datasets_filtered, socio_demographic_variables)
            # Store data for combined plot
            if model_name != 'gemma':
                all_models_data[model_name] = all_model_results_cohen
            visualize_heatmap(all_model_results_cohen, save_path=output_plot_path_cohen, model_name=model_name, plot_var='kappa')
            visualize_heatmap(all_model_results_cohen, save_path=output_plot_path_match_perc, model_name=model_name, plot_var='exact_match_perc', vmin=80, vmax=100)

    # Create combined plot with all models
    combined_output_path_cohen = pathlib.Path('statistical_analysis/ablation_label_distribution/cohen/all_models_combined_noCoT.png')
    combined_output_path_match_perc = pathlib.Path('statistical_analysis/ablation_label_distribution/match_perc/all_models_combined_noCoT.png')
    visualize_combined_heatmap(all_models_data, save_path=combined_output_path_cohen, plot_var='kappa')
    visualize_combined_heatmap(all_models_data, save_path=combined_output_path_match_perc, plot_var='exact_match_perc', vmin=80, vmax=100)


if __name__ == "__main__":
    main()