import pathlib
import pandas as pd
from utils.performance_analysis import extract_correct, get_mcnemar_results, initialize_file

"""
Assesses whether the model performance when including the socio-demographic traits specified
 in socio_demographic_variables is significantly better than the baseline using McNemar's test.
 The results are stored in statistical_analysis/performance_analysis/mcnemar_results.txt

Additionally, run the same test separately for the positive and negative labels.
 This is useful to understand if the model is better at predicting the positive or negative labels when using intersectional traits.
 The results are stored in statistical_analysis/performance_analysis/mcnemar_results_positive_negative.txt
"""


def main():
    output_path = pathlib.Path("statistical_analysis/performance_analysis/mcnemar_results.txt")
    output_path_positive = pathlib.Path("statistical_analysis/performance_analysis/mcnemar_results_positive.txt")
    output_path_negative = pathlib.Path("statistical_analysis/performance_analysis/mcnemar_results_negative.txt")
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path_positive.parent.mkdir(parents=True, exist_ok=True)
    output_path_negative.parent.mkdir(parents=True, exist_ok=True)
    # Create empty file if it doesn't exist
    initialize_file(output_path, "McNemar's Test Results")
    initialize_file(output_path_positive, "McNemar's Test Results for Positive Labels")
    initialize_file(output_path_negative, "McNemar's Test Results for Negative Labels")
    
    socio_demographic_variables = 'gender_race_political'
    model_names = ['Qwen2', 'gemma', 'Llama', 'Ministral', 'deepseek']
    cot_names = ['noCoT']

    for model_name in model_names:
        for cot_name in cot_names:
            print(f"Running McNemar's test for model: {model_name}, {cot_name}")
            with open(output_path, 'a') as f:
                f.write(f"Model: {model_name}, {cot_name}\n")
            intersection_df_path = pathlib.Path(f"predictions_dem/{model_name}/cleaned/cleaned_predictions_{model_name}_{cot_name}_{socio_demographic_variables}.csv")
            baseline_df_path = pathlib.Path(f"predictions/cleaned/cleaned_predictions_{model_name}_{cot_name}_baseline.csv")

            intersection_df = pd.read_csv(intersection_df_path)
            baseline_df = pd.read_csv(baseline_df_path)
            # Merge on postID, annID, offensiveYN
            merged_df = intersection_df.merge(baseline_df, on=['postId', 'annId', 'offensiveYN'], suffixes=('', '_baseline'))
 
            correct_intersectional, correct_baseline = extract_correct(merged_df)
            get_mcnemar_results(correct_intersectional, correct_baseline, option='greater', file_path=output_path)
            with open(output_path, 'a') as f:
                f.write(f"{'-'*50}\n\n")

            # Run McNemar's test for positive and negative labels only
            merged_df_positive = merged_df[merged_df['offensiveYN'] == 1]
            merged_df_negative = merged_df[merged_df['offensiveYN'] == 0]
            correct_intersectional_positive, correct_baseline_positive = extract_correct(merged_df_positive)
            correct_intersectional_negative, correct_baseline_negative = extract_correct(merged_df_negative)
            with open(output_path_positive, 'a') as f:
                f.write(f"Model: {model_name}, {cot_name}\n")
            get_mcnemar_results(correct_intersectional_positive, correct_baseline_positive, option='greater', file_path=output_path_positive)
            with open(output_path_positive, 'a') as f:
                f.write(f"{'-'*50}\n\n")
            
            with open(output_path_negative, 'a') as f:
                f.write(f"Model: {model_name}, {cot_name}\n")
            get_mcnemar_results(correct_intersectional_negative, correct_baseline_negative, option='greater', file_path=output_path_negative)
            with open(output_path_negative, 'a') as f:
                f.write(f"{'-'*50}\n\n")

    

if __name__ == '__main__':
    main()