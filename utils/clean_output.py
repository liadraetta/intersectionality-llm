import re 

def parse_output(text):
    output_match = re.search(r'Output:\s*(.*?)(?=\n\nInput:|$)', text, re.DOTALL)
    if output_match:
        return output_match.group(1).strip()
    return -1

def extract_prediction(parsed_output):
    if parsed_output.lower().startswith('not offensive'):
        return 0
    elif parsed_output.lower().startswith('offensive'):
        return 1
    else:
        # Try to find the label anywhere in the text
        if 'not offensive' in parsed_output.lower():
            return 0
        elif 'offensive' in parsed_output.lower():
            return 1
    return -1