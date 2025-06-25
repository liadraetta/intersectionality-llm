import pandas as pd 
import csv
import transformers
import torch
from tqdm import tqdm
from pathlib import Path
from utils.prompts import Prompts
from utils.clean_output import *

Path("/home/marem/VScProjects/intersectionality-llm/predictions/original").mkdir(parents=True, exist_ok=True)
Path("/home/marem/VScProjects/intersectionality-llm/predictions/cleaned").mkdir(parents=True, exist_ok=True)
Path("/home/marem/VScProjects/intersectionality-llm/processed_dataset").mkdir(exist_ok=True)

dir_predictions_original = "/home/marem/VScProjects/intersectionality-llm/predictions/original"
dir_predictions_cleaned = "/home/marem/VScProjects/intersectionality-llm/predictions/cleaned"
dir_processed_dataset = "/home/marem/VScProjects/intersectionality-llm/processed_dataset"

prompts = Prompts()

"""
Usage examples:

# Basic prompt (no demographics)
df["prompt"] = df.apply(prompts.get_prompt, axis=1)

# Prompt with CoT
df["prompt"] = df.apply(lambda row: prompts.get_prompt(row, demographic_traits= None, CoT=True), axis=1)

# Prompt with single demographic
df["prompt"] = df.apply(lambda row: prompts.get_prompt(row, demographic_traits='gender', CoT=False), axis=1)

# Prompt with multiple demographics
df["prompt"] = df.apply(lambda row: prompts.get_prompt(row, demographic_traits=['gender', 'race'], CoT=False), axis=1)

# Prompt with all demographics and CoT
df["prompt"] = df.apply(lambda row: prompts.get_prompt(row, demographic_traits=['gender', 'race', 'generation'], CoT=True), axis=1)
"""


# obtain subset
df_subset = pd.read_csv("/home/marem/VScProjects/intersectionality-llm/dataset/subset_50_marem.csv")


#  process subset
df = df_subset.copy()
demographic_traits=None
CoT=False

df["prompt"] = df.apply(lambda row: prompts.get_prompt(row, demographic_traits=demographic_traits, CoT=CoT), axis=1)


# obtain variables for the file name and the processed dataset
model_id = "mistralai/Ministral-8B-Instruct-2410"
model_name = model_id.split("/")[1].split("-")[0]

demogr_str = "_".join(demographic_traits) if demographic_traits else "baseline"
cot_str = "CoT" if CoT else "noCoT"

df.to_csv(f'{dir_processed_dataset}/processed_subset_{model_name}_{cot_str}_{demogr_str}.csv',index=False)
prediction_filename = f'predictions_{model_name}_{cot_str}_{demogr_str}.csv'


# create the file 
print(f"Creating file: {prediction_filename}")

file = open(f"{dir_predictions_original}/{prediction_filename}", mode='w')
writer = csv.DictWriter(file,fieldnames=["offensiveYN","HITId", "WorkerId","output"])
writer.writeheader()


# classify
transformers.set_seed(42)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.bfloat16},
    token = "my_token"
)

# Print device information
print(f"Available GPUs: {torch.cuda.device_count()}")
if hasattr(pipeline.model, 'hf_device_map'):
    print(f"Device map: {pipeline.model.hf_device_map}")
else:
    first_param_device = next(pipeline.model.parameters()).device
    print(f"Model device: {first_param_device}")


# extract the output and write it
for _,item in tqdm(df.iterrows(),total=len(df)):
  output = pipeline(
    item.prompt,
    max_new_tokens=40,
  )

  writer.writerow({'offensiveYN':item.offensiveYN,'HITId':item.HITId, 'WorkerId': item.WorkerId, 'output':output})


# clean the output
df_pred = pd.read_csv(f"{dir_predictions_original}/{prediction_filename}")
df_pred['parsed_output'] = df_pred['output'].apply(parse_output)
df_pred['prediction'] = df_pred['parsed_output'].apply(extract_prediction)


# save the final file 
df_pred.to_csv(f"{dir_predictions_cleaned}/{prediction_filename}", index=False)
