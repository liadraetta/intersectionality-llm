import pandas as pd 
import csv
import transformers
import torch
from tqdm import tqdm
from pathlib import Path
from utils.prompts import Prompts
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.clean_output import extract_demographics

dir_processed_dataset = "intersectionality-llm/processed_dataset"
dir_predictions_original = "intersectionality-llm/predictions/original"

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
df["prompt"] = df.apply(lambda row: prompts.get_prompt(row, demographic_traits=['gender', 'race', 'political leaning'], CoT=True), axis=1)
"""


# obtain subset
df = pd.read_csv("intersectionality-llm/dataset/AnnAttDataset.csv")


#  process subset
df = df.copy()
demographic_traits=None
CoT=True
batch_size=16

df["prompt"] = df.apply(lambda row: prompts.get_prompt(row, demographic_traits=demographic_traits, CoT=CoT), axis=1)

# obtain variables for the file name and the processed dataset
model_id = "mistralai/Ministral-8B-Instruct-2410"
model_name = model_id.split("/")[1].split("-")[0]

demogr_str = "_".join(demographic_traits) if demographic_traits else "baseline"
cot_str = "CoT" if CoT else "noCoT"

df.to_csv(f'{dir_processed_dataset}/processed_{model_name}_{cot_str}_{demogr_str}.csv',index=False)
prediction_filename = f'predictions_{model_name}_{cot_str}_{demogr_str}.csv'


# create the file 
print(f"Creating file: {prediction_filename}")

file = open(f"{dir_predictions_original}/{prediction_filename}", mode='w')
writer = csv.DictWriter(file,fieldnames=["offensiveYN","postId", "annId","demographics","output"])
writer.writeheader()


# classify
transformers.set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(model_id,
    token="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",
    padding_side='left')
# Set pad token if not already set
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  token="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",
  torch_dtype=torch.bfloat16,
  device_map="auto"
  )

device=next(model.parameters()).device
num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)

for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
  start_idx = batch_idx * batch_size
  end_idx = min((batch_idx + 1) * batch_size, len(df))
  df_batch = df.iloc[start_idx:end_idx]
  
  encoded = tokenizer(
    df_batch['prompt'].tolist(),
    return_tensors="pt",
    padding = True,
    truncation=True
  )
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
  
  for i, (_, item) in enumerate(df_batch.iterrows()):
    input_length = input_ids[i].shape[0]
    new_tokens = outputs[i][input_length:]
    gen_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Extract demographics and write row
    demographics = extract_demographics(item.prompt)
    writer.writerow({
        'offensiveYN': item.offensiveYN,
        'postId': item.postId,
        'annId': item.annId,
        'demographics': demographics,
        'output': gen_output
    })

"""
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
  
  for i, (_, item) in enumerate(df_batch.iterrows()):
    input_length = input_ids[i].shape[0]
    new_tokens = outputs[i][input_length:]
    gen_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Extract demographics and write row
    demographics = extract_demographics(item.prompt)
    writer.writerow({
        'offensiveYN': item.offensiveYN,
        'HITId': item.HITId,
        'WorkerId': item.WorkerId,
        'demographics': demographics,
        'output': gen_output
    })
"""