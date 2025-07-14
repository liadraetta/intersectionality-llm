import pandas as pd
import numpy as np
from utils.dataset_info import *

df = pd.read_csv("data/annWithAttitudes/largeScale.csv")
"""original columns
['Unnamed: 0', 'postId', 'tweet', 'ogId', 'ogLabel', 'source',
       'ogLabelToxic', 'aae', 'hispanic', 'other', 'white', 'dialAm', 'noi',
       'oi', 'oni', 'vulgar', 'targetsBlackPeople', 'isAAE', 'postCategory',
       'altruism', 'annotatorAge', 'annotatorGender', 'annotatorMinority',
       'annotatorPolitics', 'annotatorRace', 'dontUnderstand', 'empathy',
       'freeSpeech', 'harmHateSpeech', 'intent', 'lingPurism', 'racism',
       'racist', 'toany', 'toyou', 'traditionalism', 'off_avg', 'annId']
"""

df = df[[
    'annId', 
    'postId', 'tweet', 
    'dialAm',                                                                                       # which dialect has the highest score
    'vulgar', 'targetsBlackPeople', 'isAAE',                                                        # wheter a post belongs to a certain category
    'annotatorAge', 'annotatorGender', 'annotatorMinority','annotatorPolitics', 'annotatorRace',    # demographics 
    'intent', 'racist', 'toany', 'toyou', 'off_avg'                                                 # toxicity ratings columns
       ]]

print(df["aae"].value_counts())
# Obtain dataframe with demographic info only
df_demographics = df[["annId", "annotatorGender", "annotatorPolitics", "annotatorRace", "annotatorAge", "annotatorMinority"]]
df_demographics = df_demographics.drop_duplicates(subset="annId")

df_demographics.to_csv("./dataset/AnnAttDemographics.csv", index=False)



# Add aggregated columns
df["OffensiveYN"] = df.apply(binary_offensiveness, axis=1)
df["annotatorPoliticsBinary"] = df.apply(binary_politics, axis=1)

# Remove unwanted annotators and neutral labels 
df_cleaned = clean_dataset(df)
print("original dataframe: ", df.shape)
print("cleaned dataframe: ", df_cleaned.shape)
print()

# Dataset information and statistics
print_info(df_cleaned)


df_cleaned.to_csv("./dataset/AnnAttDataset.csv", index=False)


subset_100 = df_cleaned.sample(100, random_state=42)
subset_100.to_csv("./dataset/subset_100.csv", index=False)