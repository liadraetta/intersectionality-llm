import pandas as pd 
from utils.prompts import prompt_baseline, prompt_demographics

df = pd.read_csv("./output/FilteredTestSet.csv")
df_subset = df.sample(n=100, random_state=42) 


df_prompts = df_subset[["offensiveYN","HITId","WorkerId","post"]]
print(f"Subset of {len(df_prompts)} rows")

df_prompts = df_prompts.copy()
df_prompts["prompt"] = df_prompts.apply(prompt_baseline, axis=1)
df_prompts["prompt_demographics"] = df_prompts.apply(lambda row: prompt_demographics(row, demographic_traits=None), axis=1)
df_prompts.to_csv('output/processed_subset_50.csv',index=False)
