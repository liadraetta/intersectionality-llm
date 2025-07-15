from utils.evaluator import generate_classification_reports
from utils.clean_output import *
import pandas as pd 
import os 
from glob import glob

dir_processed_dataset = ".processed_dataset"
dir_predictions_original = "./predictions/original"
dir_predictions_cleaned = "./predictions/cleaned"

results_dir="./results/"
pattern="predictions_*_*_*.csv"
pattern_cleaned = "cleaned_predictions_*_*_*.csv"


for file in glob(os.path.join(dir_predictions_original, pattern)):
    print(file)
    filename = file.split("/")[-1]
    df = pd.read_csv(file)
    df = extract_output(df=df, output_col="output")
    df.to_csv(f"{dir_predictions_cleaned}/cleaned_{filename}", index=False)

# evaluate
generate_classification_reports(dir_predictions_cleaned, results_dir, pattern_cleaned)
