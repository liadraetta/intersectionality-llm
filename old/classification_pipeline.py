import pandas as pd 
import csv
import transformers
import torch
from tqdm import tqdm
from pathlib import Path
from utils.prompts import Prompts

dir_processed_dataset = ".processed_dataset"
dir_predictions_original = "./predictions/original"

Path(dir_processed_dataset).mkdir(exist_ok=True)
Path(dir_predictions_original).mkdir(parents=True, exist_ok=True)



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
df_subset = pd.read_csv("./dataset/subset_50_marem.csv")


#  process subset
df = df_subset.copy()
demographic_traits=None
CoT=True

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