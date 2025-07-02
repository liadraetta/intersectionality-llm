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


def parse_output(text):
    output_match = re.search(r'Output:\s*(.*?)(?=\n\n|$)', text, re.DOTALL)
    if output_match:
        return output_match.group(1).strip()
    return -1


def extract_prediction(parsed_output):
    text = parsed_output.lower()
    
    # Check if repeats the prompt
    if 'the sentence is offensive/is not offensive' in text:
        return -1
    
    offensive_count = text.count('offensive')
    not_offensive_count = text.count('not offensive')
    
    # Actual offensive count (subtract "not offensive" instances)
    actual_offensive_count = offensive_count - not_offensive_count
    
    if not_offensive_count > 0 and actual_offensive_count == 0:
        return 0  # Only "not offensive" found
    elif actual_offensive_count > 0 and not_offensive_count == 0:
        return 1  # Only "offensive" found
    else:
        return -1  # Both found, neither found, or ambiguous