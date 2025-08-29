import pandas as pd
import numpy as np
from utils.dataset_info import *
import matplotlib.pyplot as plt
import seaborn as sns

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


# Obtain dataframe with demographic info only
df_demographics = df[["annId", "annotatorGender", "annotatorPolitics", "annotatorRace", "annotatorAge", "annotatorMinority"]]
df_demographics = df_demographics.drop_duplicates(subset="annId")

# df_demographics.to_csv("./dataset/AnnAttDemographics.csv", index=False)



# Add aggregated columns
df["offensiveYN"] = df.apply(binary_offensiveness, axis=1)
df["annotatorPoliticsBinary"] = df.apply(binary_politics, axis=1)


# Remove unwanted annotators and neutral labels 
df_cleaned = clean_dataset(df)



# df_cleaned.to_csv("./dataset/AnnAttDataset.csv", index=False)


# subset_100 = df_cleaned.sample(100, random_state=42)
# subset_100.to_csv("./dataset/subset_100.csv", index=False)



# Print statistics
print("original dataframe: ", df.shape)
print("cleaned dataframe: ", df_cleaned.shape)
print("-"*100)

print("original dataframe statistics: ")
print("*"*25)
print_info(df)

print("cleaned dataframe statistics: ")
print("*"*25)
print_info(df_cleaned)

print("-"*100)




# Prepare data
counts = df["off_avg"].value_counts().sort_index()
plot_data = counts.reset_index()
plot_data.columns = ["off_avg", "count"]

plt.figure(figsize=(8, 5))
sns.barplot(data=plot_data, x="off_avg", y="count", palette="Blues")
plt.xlabel("Offensiveness")
plt.ylabel("Frequency")

for i in range(len(plot_data)):
    plt.text(i, plot_data["count"][i] + 10, plot_data["count"][i], ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("./dataset/plot_offensiveness.png", dpi=300)
plt.close()

