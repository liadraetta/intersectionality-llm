import pandas as pd
from collections import defaultdict
import pickle

filtered_dataset = pd.read_csv("./output/FilteredTestSet.csv")

demographics = filtered_dataset[["WorkerId","annotatorGender","annotatorRace","annotatorGeneration"]]
demographics = demographics.drop_duplicates(subset="WorkerId")

print(demographics.shape)

dict_user = defaultdict(dict)
for index,row in demographics.iterrows():
    id = row["WorkerId"]
    gender = row["annotatorGender"]
    race = row["annotatorRace"]
    gen = row["annotatorGeneration"]
    dict_user[id]["gender"] = gender
    dict_user[id]["race"] = race
    dict_user[id]["generation"] = gen

with open('./output/dict_user_demographics.pickle', 'wb') as handle:
    pickle.dump(dict_user, handle, protocol=pickle.HIGHEST_PROTOCOL)