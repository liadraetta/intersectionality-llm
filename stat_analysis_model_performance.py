import pathlib
import pandas as pd
from utils.performance_analysis import extract_correct, get_mcnemar_results, initialize_file, run_mcnemar_tests
from utils.label_distribution_analysis import read_all_model_datasets

"""
Assesses whether the model performance when including the socio-demographic traits specified
 in socio_demographic_variables is significantly better than the baseline using McNemar's test.
 The results are stored in statistical_analysis/performance_analysis/mcnemar_results.txt

Additionally, run the same test separately for the positive and negative labels.
 This is useful to understand if the model is better at predicting the positive or negative labels when using intersectional traits.
 The results are stored in statistical_analysis/performance_analysis/mcnemar_results_positive_negative.txt
"""

# Restructure the code to have statistical_analysis/performance_analysis/{model_name}/[mcnemar_results.txt, mcnemar_results_positive.txt, mcnemar_results_negative.txt]

def main():
    model_names = ['Qwen2', 'gemma', 'Llama', 'Ministral', 'deepseek']
    socio_demographic_variables = ['gender', 'race', 'political']
    cot_names = ['noCoT']
    results_base_path = pathlib.Path('predictions/cleaned') 
    for model_name in model_names:
        results_socdem_base_path = pathlib.Path(f"predictions_dem/{model_name}/cleaned")
        for cot_name in cot_names:
            output_path = pathlib.Path(f"statistical_analysis/performance_analysis/{model_name}/mcnemar_results.txt")
            output_path_positive = pathlib.Path(f"statistical_analysis/performance_analysis/{model_name}/mcnemar_results_positive.txt")
            output_path_negative = pathlib.Path(f"statistical_analysis/performance_analysis/{model_name}/mcnemar_results_negative.txt")
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path_positive.parent.mkdir(parents=True, exist_ok=True)
            output_path_negative.parent.mkdir(parents=True, exist_ok=True)
            # Create empty file if it doesn't exist
            initialize_file(output_path, "McNemar's Test Results", model_name=model_name, cot_name=cot_name)
            initialize_file(output_path_positive, "McNemar's Test Results for Positive Labels", model_name=model_name, cot_name=cot_name)
            initialize_file(output_path_negative, "McNemar's Test Results for Negative Labels", model_name=model_name, cot_name=cot_name)

            print(f"Running McNemar's test for model: {model_name}, {cot_name}")
            # Read all model datasets
            all_model_datasets = read_all_model_datasets(model_name, cot_name, results_base_path, results_socdem_base_path)
            run_mcnemar_tests(all_model_datasets, output_path, output_path_positive, output_path_negative)

if __name__ == '__main__':
    main()