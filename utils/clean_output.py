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



def extract_prediction(parsed_output):
    text = parsed_output.lower()
    
    # Check if repeats the prompt
    if '[the sentence is offensive]' in text: 
        return 1
    elif '[the sentence is not offensive]' in text:
        return 0
    else:
        return -1

