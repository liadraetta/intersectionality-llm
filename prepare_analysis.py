import pandas as pd
import os

# Paths
input_data_path = "dataset/AnnAttDataset.csv"
predictions_folder = "predictions_dem/Llama/cleaned"
output_folder = "error_analysis"
os.makedirs(output_folder, exist_ok=True)

input_df = pd.read_csv(input_data_path)

input_df = input_df[['postId', 'tweet']].drop_duplicates(subset='postId')

for pred_file in os.listdir(predictions_folder):
    if pred_file.endswith(".csv"):
        pred_path = os.path.join(predictions_folder, pred_file)

        pred_df = pd.read_csv(pred_path, sep=",")
        pred_df.columns = pred_df.columns.str.strip()

        errors_df = pred_df[pred_df['offensiveYN'] != pred_df['prediction']]

        if 'output' in errors_df.columns:
            errors_df = errors_df.drop(columns=['output'])

        # Merge with one-row-per-postId text data
        merged_df = errors_df.merge(input_df, on='postId', how='left')


        base_name = os.path.splitext(pred_file)[0]
        output_path = os.path.join(output_folder, f"errorAnalysis_{base_name}.csv")
        merged_df.to_csv(output_path, index=False)