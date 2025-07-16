import re 
import ast 

def extract_demographics(prompt):
    """
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
    text = output.lower()
    
    # Check if repeats the prompt
    if "output" in text:
        pattern = r"(?<=output:\s).*?(?=\n|$)"  # Fixed regex
        match = re.search(pattern, text)
        
        if match:  # Check if match exists
            match_text = match.group()
            if "the sentence is offensive" in match_text:
                print(match_text)
                print(1)
                return 1
            elif "the sentence is not offensive" in match_text:
                print(match_text)
                print(0)
                return 0
            else:
                print(match_text)
                print(-1)
                return -1
        else:
            print("No match found")
            print(-1)
            return -1
    else:
        if '[the sentence is offensive]' in text:
            print(1)
            return 1
        elif '[the sentence is not offensive]' in text:
            print(0)
            return 0  # Added missing return
        else:
            print(-1)
            return -1



def extract_bracket_content(text):
    # Split text into lines
    lines = text.strip().split('\n')
    
    # Find the line that starts with brackets and contains multiple bracket pairs
    for line in lines:
        line = line.strip()
        # Check if line starts with [ and has multiple bracket pairs
        if line.startswith('[') and line.count('[') >= 2:
            # Pattern to match text within square brackets
            pattern = r'\[([^\]]+)\]'
            # Find all matches in this specific line
            matches = re.findall(pattern, line)
            return ' '.join(matches)
    
    return []


def extract_output(df, output_col):
    df['prediction'] = df[output_col].apply(extract_prediction)
    
    texts = df[output_col].tolist()
    results = []
    for text in texts: 
        result = extract_bracket_content(text)
        results.append(result)
    df['parsed_output'] = results

    return df 