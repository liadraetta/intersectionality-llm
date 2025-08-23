import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
def extract_correct(df, correct_column='offensiveYN', prediction_column='prediction', baseline_prediction_column='prediction_baseline'):
    """
    Extracts the correct predictions from the DataFrame as a list of 1s and 0s with 1 indicating a correct prediction.
    -1 are treated as incorrect predictions.
    """
    correct = [1 if row[prediction_column] == row[correct_column] else 0 for _, row in df.iterrows()]
    correct_baseline = [1 if row[baseline_prediction_column] == row[correct_column] else 0 for _, row in df.iterrows()]
    return correct, correct_baseline

def get_mcnemar_results(correct_intersectional, correct_baseline, option='greater', file_path=None):
    contingency_table = pd.crosstab(correct_intersectional, correct_baseline, rownames=['Intersectional'], colnames=['Baseline'])
    result = mcnemar(contingency_table, exact=True)
    with open(file_path, 'a') if file_path else None as f:
        if f:
            f.write("Contingency Table:\n")
            f.write(str(contingency_table) + "\n")
            f.write("McNemar's Test Result:\n")
            if option == 'two-sided':
                f.write(f"Statistic: {result.statistic}, p-value: {result.pvalue}\n")
            elif option == 'greater':
                # To use one-sided test, we need to check the contingency table
                if contingency_table[0][1] < contingency_table[1][0]:
                    f.write("The contingency table does not meet the requirements for a one-sided test.\nThe results with intersectional data are worse than the baseline.\n")
                    return
                f.write(f"Statistic: {result.statistic}, p-value (greater): {result.pvalue / 2}\n")
        else:
            print(contingency_table)
            print("McNemar's Test Result:")
            if option == 'two-sided':
                print(f"Statistic: {result.statistic}, p-value: {result.pvalue}")
            elif option == 'greater':
                # To use one-sided test, we need to check the contingency table
                if contingency_table[0][1] < contingency_table[1][0]:
                    print("The contingency table does not meet the requirements for a one-sided test.\nThe results with intersectional data are worse than the baseline.")
                    return
                print(f"Statistic: {result.statistic}, p-value (greater): {result.pvalue / 2}")

def initialize_file(file_path, header, model_name=None, cot_name=None):
    """
    Initializes the file by creating it if it doesn't exist and writing the header.
    """
    with open(file_path, 'w') as f:
        f.write(header)
        f.write(f"Model: {model_name}, CoT: {cot_name}\n")
        f.write("="*50 + "\n\n")


def run_mcnemar_tests(all_model_datasets, 
                      output_path, 
                      output_path_positive, 
                      output_path_negative,):
    # extract baseline data
    baseline_df = all_model_datasets['baseline']

    # Iterate through each reduced set of socio-demographic variables.
    # Sort the keys to ensure consistent order: baseline, gender, race, political, gender_race, gender_political, race_political, gender_race_political
    sorted_keys_order = ['baseline', 'gender', 'race', 'political', 'gender_race', 'gender_political', 'race_political', 'gender_race_political']
    for key in sorted_keys_order:
        df = all_model_datasets[key]
        if key == 'baseline':
            continue
        # Merge on postID, annID, offensiveYN
        merged_df = df.merge(baseline_df, on=['postId', 'annId', 'offensiveYN'], suffixes=('', '_baseline'))
        correct_intersectional, correct_baseline = extract_correct(merged_df)
        with open(output_path, 'a') as f:
            f.write(f"Trait: {key}\n")
        get_mcnemar_results(correct_intersectional, correct_baseline, option='greater', file_path=output_path)
        with open(output_path, 'a') as f:
            f.write(f"{'-'*50}\n\n")

        # Run McNemar's test for positive and negative labels only
        merged_df_positive = merged_df[merged_df['offensiveYN'] == 1]
        merged_df_negative = merged_df[merged_df['offensiveYN'] == 0]
        correct_intersectional_positive, correct_baseline_positive = extract_correct(merged_df_positive)
        correct_intersectional_negative, correct_baseline_negative = extract_correct(merged_df_negative)
        with open(output_path_positive, 'a') as f:
            f.write(f"Trait: {key}\n")
        get_mcnemar_results(correct_intersectional_positive, correct_baseline_positive, option='greater', file_path=output_path_positive)
        with open(output_path_positive, 'a') as f:
            f.write(f"{'-'*50}\n\n")
        
        with open(output_path_negative, 'a') as f:
            f.write(f"Trait: {key}\n")
        get_mcnemar_results(correct_intersectional_negative, correct_baseline_negative, option='greater', file_path=output_path_negative)
        with open(output_path_negative, 'a') as f:
            f.write(f"{'-'*50}\n\n")


