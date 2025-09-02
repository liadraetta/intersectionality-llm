import pandas as pd
import os

input_data_path = "dataset/AnnAttDataset.csv"
#predictions_folder = "predictions_dem/Llama/cleaned"
predictions_path = "results/predictions_dem/Llama/cleaned/cleaned_predictions_Llama_noCoT_race_political.csv"
output_folder = "qualitative_analysis"
os.makedirs(output_folder, exist_ok=True)

input_df = pd.read_csv(input_data_path)
input_df = input_df[['postId', 'tweet', 'isAAE', 'vulgar', 'targetsBlackPeople']].drop_duplicates(subset='postId')

df = pd.read_csv(predictions_path, sep=",")
df.columns = df.columns.str.strip()

errors_df = df[df['offensiveYN'] != df['prediction']]

if 'output' in errors_df.columns:
    errors_df = errors_df.drop(columns=['output'])

false_positive_df = errors_df[errors_df['offensiveYN'] == 0]

merged_df = false_positive_df.merge(input_df, on='postId', how='left')

#merged_df_ministral = merged_df.merge(df_intersections, on='postId', how='left')
#merged_df_ministral.drop_duplicates(subset='annId')
#merged_df_ministral = merged_df_ministral[merged_df_ministral['prediction_ministral_intersection'] == 1]

output_path = os.path.join(output_folder, f"false_negative_llama_race_politics.csv")
merged_df.to_csv(output_path, index=False)