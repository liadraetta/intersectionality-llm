import pandas as pd 
import csv
import argparse
from itertools import combinations
import transformers
import torch
from tqdm import tqdm
from pathlib import Path
from utils.prompts import Prompts
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.clean_output import extract_demographics


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
df["prompt"] = df.apply(lambda row: prompts.get_prompt(row, demographic_traits=['gender', 'race', 'political'], CoT=True), axis=1)
"""
def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cot', action='store_true', help='Use Chain of Thought (CoT) prompting')
    return parser.parse_args()

args = parse_command_line_args()
df = pd.read_csv("intersectionality-llm/dataset/AnnAttDataset.csv")

model_id = args.model_id
model_name = model_id.split("/")[1].split("-")[0]
CoT = args.cot
batch_size = args.batch_size

dir_predictions_dem_model = f"intersectionality-llm/predictions_dem/{model_name}/original"
dir_processed_dem_model = f"intersectionality-llm/processed_dataset_dem/{model_name}"
Path(dir_predictions_dem_model).mkdir(parents=True, exist_ok=True)
Path(dir_processed_dem_model).mkdir(parents=True, exist_ok=True)

transformers.set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                          token="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",
                                          padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    token="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",
    torch_dtype=torch.bfloat16,
    device_map="auto"
   )

device=next(model.parameters()).device

list_traits = ["gender", "race", "political"]

for r in range(1,len(list_traits)+1):
    print("subset lenght: ", r)
    trait_combination = list(combinations(list_traits, r))
    for i in trait_combination:
        list_dem = list(i)
        print(list_dem)

        #  process subset
        df = df.copy()
        
        demographic_traits=list_dem

        df["prompt"] = df.apply(lambda row: prompts.get_prompt(row, demographic_traits=demographic_traits, CoT=CoT), axis=1)
        print(df["prompt"][0])
        print(f"\n\n\n")
        continue
        #raise ValueError("STOP HERE")  # Debugging point

        # obtain variables for the file name and the processed dataset
        demogr_str = "_".join(demographic_traits) if demographic_traits else "baseline"
        cot_str = "CoT" if CoT else "noCoT"

        df.to_csv(f'{dir_processed_dem_model}/processed_{model_name}_{cot_str}_{demogr_str}.csv',index=False)
        prediction_filename = f'predictions_{model_name}_{cot_str}_{demogr_str}.csv'


        # prediction_filenamecreate the file 
        print(f"Creating file: {prediction_filename}")

        file = open(f"{dir_predictions_dem_model}/{prediction_filename}", mode='w')
        writer = csv.DictWriter(file,fieldnames=["offensiveYN","postId", "annId","demographics","output"])
        writer.writeheader()

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
                max_new_tokens=80,
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