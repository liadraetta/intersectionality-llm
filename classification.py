import pandas as pd 
import csv
import transformers
import argparse
import torch
import os
from tqdm import tqdm
from pathlib import Path
from utils.prompts import Prompts
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.clean_output import extract_demographics
from itertools import combinations
# Load hf key using dotenv
from dotenv import load_dotenv

def prepare_tokenizer_model_device(model_id, hf_token):
      transformers.set_seed(42)

      tokenizer = AutoTokenizer.from_pretrained(model_id,
          token=hf_token,
          padding_side='left')
      # Set pad token if not already set
      if tokenizer.pad_token is None:
              tokenizer.pad_token = tokenizer.eos_token

      model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto"
        )

      device=next(model.parameters()).device
      return tokenizer, model, device

dir_processed_dataset = "processed_dataset"
dir_predictions_original = "predictions/original"

Path(dir_processed_dataset).mkdir(exist_ok=True)
Path(dir_predictions_original).mkdir(parents=True, exist_ok=True)

prompts = Prompts()

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cot', action='store_true', help='Use Chain of Thought (CoT) prompting')
    parser.add_argument('--sociodemographic_traits', action='store_true', help='Use sociodemographic prompting')
    return parser.parse_args()

def main():
  args = parse_command_line_args()

  # Read hf token from .env file
  load_dotenv()
  hf_token = os.getenv("HF_TOKEN")
  if not hf_token:
      raise ValueError("HF_TOKEN is not set, create a .env file and add it.")

  # Load dataset and annotator demographics dataset
  df = pd.read_csv("dataset/AnnAttDataset.csv")
  df = df.copy()
  demographics_df = pd.read_csv("dataset/AnnAttDemographics.csv")
  # Discretize if <0 'liberal' if 0 'neutral' if > 0 'conservative'
  demographics_df['annotatorPoliticsDiscrete'] = demographics_df['annotatorPolitics'].apply(lambda x: 'liberal' if x < 0 else ('neutral' if x == 0 else 'conservative'))

  model_name = args.model_id.split("/")[1].split("-")[0]
  cot_str = "CoT" if args.cot else "noCoT"
  batch_size = args.batch_size
  if args.sociodemographic_traits:
    list_traits=["gender", "race", "political"]
  else:
    list_traits=[None]

  dir_predictions_base = f"predictions/{model_name}/original"
  dir_predictions_dem_model = f"predictions_dem/{model_name}/original"
  dir_predictions = dir_predictions_dem_model if args.sociodemographic_traits else dir_predictions_base

  tokenizer, model, device = prepare_tokenizer_model_device(args.model_id, hf_token)

  for r in range(1,len(list_traits)+1):
    trait_combinations = list(combinations(list_traits, r))
    for i in trait_combinations:
        list_dem = list(i)
        demographic_traits = list_dem if list_dem[0] else None
        print(list_dem)
        demogr_str = "_".join(list_dem) if list_dem[0] else "baseline"
        prediction_filename = f'predictions_{model_name}_{cot_str}_{demogr_str}.csv'

        df = df.copy()
        df["prompt"] = df.apply(lambda row: prompts.get_prompt(row, demographic_traits=demographic_traits, CoT=args.cot), axis=1)
        # Extract demographics as a dictionary.
        df["demographics"] = extract_demographics(df, demographics_df, list_dem)

        # Create the file:
        file = open(f"{dir_predictions}/{prediction_filename}", mode='w')
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
                demographics = item['demographics']

                # Extract demographics and write row
                # demographics = extract_demographics(item.prompt)
                writer.writerow({
                    'offensiveYN': item.offensiveYN,
                    'postId': item.postId,
                    'annId': item.annId,
                    'demographics': demographics,
                    'output': gen_output
                })

if __name__ == "__main__":
    main()