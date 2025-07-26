import re 
import ast 

def extract_demographics(prompt):
    """
    TODO: TO FIX TO WORK WITH NEW PROMPT ---
    Extract demographics dictionary from prompt text.
    Returns the dictionary if found, otherwise returns "None".
    """
    # Pattern to match "You are characterized by the following demographics:" followed by a dictionary
    pattern = r"You are characterized by the following demographics:\s*(\{[^}]*\})"
    match = re.search(pattern, prompt)
    
    if match:
        try:
            # Extract the dictionary string
            dict_str = match.group(1)
            # Safely evaluate the dictionary string
            demographics = ast.literal_eval(dict_str)
            return demographics
        except (ValueError, SyntaxError):
            # If the dictionary string is malformed, return "None"
            return "None"
    else:
        return "None"


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

"""
def extract_prediction(output):
    text = re.sub(r'\s+', ' ', output.strip()).lower()
    
    # Check if repeats the prompt
    if "output" in text:
        pattern = r"(?<=output:\s).*?(?=\n|$)"  # Fixed regex
        match = re.search(pattern, text)
        
        if match:  # Check if match exists
            match_text = match.group()
            if "the sentence is offensive" in match_text:
                return 1
            elif "the sentence is not offensive" in match_text:
                return 0
            elif '[the sentence is offensive]' in text:
                return 1
            elif '[the sentence is not offensive]' in text:
                return 0
            else:
                return -1
        elif '[the sentence is offensive]' in text: #c'è "output" ma non fa match con la regex
            return 1
        elif '[the sentence is not offensive]' in text:
            return 0
        else: return -1
    else:
        if '[the sentence is offensive]' in text:
            return 1
        elif '[the sentence is not offensive]' in text:
            return 0  # Added missing return
        else:
            return -1



def extract_bracket_content(output):
    if not output:
        return ""
    
    text = re.sub(r'\s+', ' ', output.strip()).lower()
    
    # Check if it contains "output" and extract from that pattern
    if "output" in text:
        pattern = r"output:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text)
        if match:
            match_text = match.group(1)
            brackets = re.findall(r'\[([^\]]+)\]', match_text)
            if brackets:
                return ' '.join(brackets)
    
    # Fallback: find lines with multiple bracket pairs
    lines = output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('[') and line.count('[') >= 2:
            brackets = re.findall(r'\[([^\]]+)\]', line)
            return ' '.join(brackets)
    
    return ""


def extract_output(df, output_col):
    df['prediction'] = df[output_col].apply(extract_prediction)
    
    texts = df[output_col].tolist()
    results = []
    for text in texts: 
        result = extract_bracket_content(text)
        results.append(result)
    df['parsed_output'] = results

    return df 
"""