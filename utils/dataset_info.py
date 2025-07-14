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

def binary_offensiveness(row):
    off_avg = row["off_avg"]

    if off_avg > 3.0:
        return 1
    elif off_avg < 3.0: 
        return 0
    else:
        return "neutral"
    
def binary_politics(row):
    politics = row["annotatorPolitics"]

    if politics > 0:
        return "conservative"
    elif politics < 0:
        return "liberal"
    else:
        return "neutral"
    
def clean_dataset(df): 
    df = df[df["OffensiveYN"] != "neutral"] #remove off_avg = 3.0
    races_to_remove = ["middleEastern", "na", "other", "hisp", "native"]

    df = df[~df["annotatorRace"].isin(races_to_remove)]

    return df 


def print_info(df):
    print("number of texts: ", len(df["postId"].unique()))
    print("number of annotators: ", len(df["annId"].unique()))
    print()

    print("Label distribution")
    print(df["OffensiveYN"].value_counts())
    print()

    print("Annotator level")
    d_annotator = statistics_annotations(df, "annId")
    print()

    print("Annotation level")
    d_annotation = statistics_annotations(df, "postId")
    print()







#######################################   Functions for the SBIC dataset   ############################################
"""

def age_range(row): 
    # https://www.pewresearch.org/short-reads/2019/01/17/where-millennials-end-and-generation-z-begins/

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
"""
