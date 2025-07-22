from utils.evaluator import *
from utils.clean_output import *
import pandas as pd 
import os 
from glob import glob

demographics = True
list_models = ["deepseek", "gemma", "Llama", "Ministral", "Qwen2"]
ids_to_remove = [3768, 60]


if not demographics:
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
        df = df[~df['postId'].isin(ids_to_remove)]
        df.to_csv(f"{dir_predictions_cleaned}/cleaned_{filename}", index=False)

    generate_classification_reports(dir_predictions_cleaned, results_dir, pattern_cleaned)

else:
    for model_name in list_models:

        dir_predictions_original = f"./predictions_dem/{model_name}/original"
        dir_predictions_cleaned = f"./predictions_dem/{model_name}/cleaned"

        Path(dir_predictions_cleaned).mkdir(parents=True, exist_ok=True)
        results_dir=f"./results_dem/{model_name}/"
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        pattern = f"predictions_{model_name}*.csv"
        pattern_cleaned = f"cleaned_predictions_{model_name}*.csv"

        for file in glob(os.path.join(dir_predictions_original, pattern)):
            print(file)
            filename = file.split("/")[-1]
            df = pd.read_csv(file)
            df = extract_output(df=df, output_col="output")
            df = df[~df['postId'].isin(ids_to_remove)]
            df.to_csv(f"{dir_predictions_cleaned}/cleaned_{filename}", index=False)

    
        generate_classification_reports(dir_predictions_cleaned, results_dir, pattern_cleaned)



"""
for file in glob(os.path.join(dir_predictions_original, pattern)):
    print(file)
    filename = file.split("/")[-1]
    df = pd.read_csv(file)
    df = extract_output(df=df, output_col="output")
    df.to_csv(f"{dir_predictions_cleaned}/cleaned_{filename}", index=False)"""
