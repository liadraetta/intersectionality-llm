import pandas as pd 
import csv
import transformers
import torch
from tqdm import tqdm
from pathlib import Path
from utils.prompts import Prompts
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.clean_output import extract_demographics

dir_processed_dataset = "./processed_dataset"
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
df_subset = pd.read_csv("./dataset/subset_100.csv")


#  process subset
df = df_subset.copy()
demographic_traits=None
CoT=True

df["prompt"] = df.apply(lambda row: prompts.get_prompt(row, demographic_traits=demographic_traits, CoT=CoT), axis=1)


# obtain variables for the file name and the processed dataset
model_id = "meta-llama/Llama-3.1-8B"
model_name = model_id.split("/")[1].split("-")[0]

demogr_str = "_".join(demographic_traits) if demographic_traits else "baseline"
cot_str = "CoT" if CoT else "noCoT"

df.to_csv(f'{dir_processed_dataset}/processed_subset_{model_name}_{cot_str}_{demogr_str}.csv',index=False)
prediction_filename = f'predictions_{model_name}_{cot_str}_{demogr_str}.csv'


# create the file 
print(f"Creating file: {prediction_filename}")

file = open(f"{dir_predictions_original}/{prediction_filename}", mode='w')
writer = csv.DictWriter(file,fieldnames=["offensiveYN","HITId", "WorkerId","demographics","output"])
writer.writeheader()


# classify
transformers.set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  token="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",
  torch_dtype=torch.bfloat16,
  device_map="auto"
  )

# extract the output and write it
device=next(model.parameters()).device

for _,item in tqdm(df.iterrows(),total=len(df)):
  encoded = tokenizer(item.prompt, return_tensors="pt")
  input_ids = encoded.input_ids.to(device)
  attention_mask = encoded.attention_mask.to(device)

  with torch.no_grad():
    outputs = model.generate(
      input_ids,
      attention_mask = attention_mask,
      do_sample=False,
      max_new_tokens=30,
      pad_token_id=tokenizer.eos_token_id
    )

  # gen_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
  input_length = input_ids.shape[1]
  new_tokens = outputs[0][input_length:]
  gen_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
  
  demographics = extract_demographics(item.prompt)

  writer.writerow({
    'offensiveYN':item.offensiveYN,
    'HITId':item.HITId, 
    'WorkerId': item.WorkerId, 
    'demographics':demographics, 
    'output':gen_output
    })


