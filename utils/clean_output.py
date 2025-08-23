import re 
import ast 

def extract_demographics(df, demographics_df, list_dem):
   demographics = []
   for _, row in df.iterrows():
       row_annId = row['annId']
       annotator_row = demographics_df[demographics_df['annId'] == row_annId]
       annotator_full_demographics = {'gender': annotator_row['annotatorGender'].values[0], 'race': annotator_row['annotatorRace'].values[0], 'political': annotator_row['annotatorPoliticsDiscrete'].values[0]}
       demo = {trait: annotator_full_demographics[trait] for trait in list_dem} if list_dem[0] else None
       demographics.append(demo)
   return demographics

def extract_prediction(output):
    text = output.lower().strip().splitlines()
    for i in text:
        if 'the sentence is offensive' in i:
            return 1, i
        elif 'the sentence is not offensive' in i:
            return 0, i
    return -1, None

def extract_output(df, output_col):
    list_preds = []
    list_parsed_outputs = []
    for idx,row in df.iterrows():
        pred, parsed_output = extract_prediction(row[output_col])
        list_preds.append(pred)
        list_parsed_outputs.append(parsed_output)

    df['prediction'] = list_preds
    df['parsed_output'] = list_parsed_outputs

    return df 
