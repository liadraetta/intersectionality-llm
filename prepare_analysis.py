import pandas as pd
import os

input_data_path = "dataset/AnnAttDataset.csv"
#predictions_folder = "predictions_dem/Llama/cleaned"
predictions_path = "predictions/cleaned/cleaned_predictions_Ministral_noCoT_baseline.csv"
output_folder = "qualitative_analysis"
path_ministral_intersection = "predictions_dem/Ministral/cleaned/cleaned_predictions_Ministral_noCoT_gender_race_political.csv"
os.makedirs(output_folder, exist_ok=True)

input_df = pd.read_csv(input_data_path)

input_df = input_df[['postId', 'tweet', 'isAAE', 'vulgar', 'targetsBlackPeople']].drop_duplicates(subset='postId')

df_intersections = pd.read_csv(path_ministral_intersection).drop_duplicates(subset='postId')
df_intersections = df_intersections[['prediction', 'postId', 'demographics']]
df_intersections = df_intersections.rename(columns={'prediction': 'prediction_ministral_intersection'})

df = pd.read_csv(predictions_path, sep=",")
df.columns = df.columns.str.strip()

errors_df = df[df['offensiveYN'] != df['prediction']]

if 'output' in errors_df.columns:
    errors_df = errors_df.drop(columns=['output'])

false_positive_df = errors_df[errors_df['offensiveYN'] == 1]

merged_df = false_positive_df.merge(input_df, on='postId', how='left')

merged_df_ministral = merged_df.merge(df_intersections, on='postId', how='left')
#merged_df_ministral.drop_duplicates(subset='annId')
merged_df_ministral = merged_df_ministral[merged_df_ministral['prediction_ministral_intersection'] == 1]

output_path = os.path.join(output_folder, f"merged_false_negative_Ministral.csv")
merged_df_ministral.to_csv(output_path, index=False)