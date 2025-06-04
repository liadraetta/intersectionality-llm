import pandas as pd
import numpy as np
from utils.dataset_info import *

train = pd.read_csv("data/SBIC.v2/SBIC.v2.trn.csv")
train.insert(1, "set", "train")
dev = pd.read_csv("data/SBIC.v2/SBIC.v2.dev.csv")
dev.insert(1, "set", "dev")
test = pd.read_csv("data/SBIC.v2/SBIC.v2.tst.csv")
test.insert(1, "set", "test")


print(train.shape, dev.shape, test.shape)
df = pd.concat([train,dev,test], axis=0)

# Obtain dataframe with demographic info only
df_demographics = df[["set","WorkerId", "annotatorGender", "annotatorMinority", 
                      "annotatorPolitics", "annotatorRace", "annotatorAge"]]
df_demographics = df_demographics.drop_duplicates(subset="WorkerId")
df_demographics["annotatorGeneration"] = df_demographics.apply(age_range, axis=1)
df_demographics["IntersectionMinority"] = df_demographics.apply(white_male, axis=1)

df_demographics.to_csv("./output/SBICdemographics.csv")


"""We will work on the test set only"""
# Add column ["annotatorGeneration"] to main df, excluding the unknown values
df_test = test.copy()
df_test = df_test.merge(df_demographics[["WorkerId", "annotatorGeneration", "IntersectionMinority"]], on="WorkerId", how="left")

# Remove unwanted annotators 
df_test = df_test[df_test["annotatorGeneration"] != "Unknown"] 
df_test = df_test[df_test["IntersectionMinority"] != "NoMinority"] 


# Dataset information and statistics
print("Filtered dataset shape: ", df_test.shape)
print_info(df_test)

df_test.to_csv("./output/FilteredTestSet.csv", index=False)



# traits = ["annotatorGender", "annotatorRace", "annotatorGeneration"]


