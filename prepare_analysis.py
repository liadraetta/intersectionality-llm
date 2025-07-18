import pandas as pd
import os

input_data = "dataset/AnnAttDataset.csv"

predictions_folder = "predictions_dem/Llama/cleaned"

output_folder = "error_analysis"
os.makedirs(output_folder, exist_ok=True)

input_df = pd.read_csv(input_data)

for pred_file in os.listdir(predictions_folder):
    if pred_file.endswith(".csv"):
        pred_path = os.path.join(predictions_folder, pred_file)

        pred_df = pd.read_csv(pred_path, sep=",")

        errors_df = pred_df[pred_df['offensiveYN'] != pred_df['prediction']]

        if 'output' in errors_df.columns:
            errors_df = errors_df.drop(columns=['output'])

        merged_df = errors_df.merge(input_df, on='postId', how='left')
        merged_df = merged_df.drop(columns=['annId_y'])
        base_name = os.path.splitext(pred_file)[0]
        output_path = os.path.join(output_folder, f"errorAnalysis_{base_name}.csv")
        merged_df.to_csv(output_path, index=False)

        print(f"Saved: {output_path}")