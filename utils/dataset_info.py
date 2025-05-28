import pandas as pd 
import numpy as np 

def statistics_annotations(df, column_name):
  df2 = df.pivot_table(index = [column_name], aggfunc ='size')

  d = df2.to_dict()

  l_n_annotations = []
  for k,v in d.items():
    l_n_annotations.append(v)

  print("minimum of annotations: ",min(l_n_annotations))
  print("maximum of annotations: ",max(l_n_annotations))
  print("mean of annotations: ",np.average(l_n_annotations))
  print("median of annotations: ",np.median(l_n_annotations))

  return d



def print_info(df):
    print("number of texts: ", len(df["HITId"].unique()))
    print("number of annotators: ", len(df["WorkerId"].unique()))
    print()

    print("Label distribution")
    print(df["offensiveYN"].value_counts())
    print()

    print("Annotator level")
    d_annotator = statistics_annotations(df, "WorkerId")
    print()

    print("Annotation level")
    d_annotation = statistics_annotations(df, "HITId")
    print()



def age_range(row):
    """https://www.pewresearch.org/short-reads/2019/01/17/where-millennials-end-and-generation-z-begins/"""

    age = row["annotatorAge"]
    
    if 56 <= age <= 74:
        return "Boomer"
    elif 40 <= age <= 55:
        return "GenX"
    elif 24 <= age <= 39:
        return "GenY"
    elif 8 <= age <= 23:
        return "GenZ"
    else:
        return "Unknown"  # unknown or out of range
    


def white_male(row):
    race = row["annotatorRace"]
    gender = row["annotatorGender"]

    if race == "white" and gender == "man":
        return "NoMinority"
    else:
        return "YesMinority"
