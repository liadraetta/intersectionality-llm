import pickle 

"""
Example to call the following functions: 

    df["prompt"] = df.apply(prompt_baseline, axis=1)


    df["prompt"] = df.apply(lambda row: prompt_demographics(row, demographic_traits="gender"), axis=1)
    df["prompt"] = df.apply(lambda row: prompt_demographics(row, demographic_traits=["gender", "race"]), axis=1)
    df["prompt"] = df.apply(lambda row: prompt_demographics(row, demographic_traits=None), axis=1) --> All demographics 
"""


#Prompts for the baseline

def prompt_baseline(row):
    
    text = row["post"]
    
    instruction = {
       "prelude":"You are a person asked to provide precise information about offensiveness of sentences.",
        "call_instr": "### Instruction:",
        "instr": "Given the sentence in input (Input) taken from a social network, your task is to determine if the text is offensive and why.",
        "note": "Some of the sentences may contain hate or explicit language, please provide an answer in any case.",
        "format": "Provide in output a single label among the following arrays ['offensive', 'not offensive'].",
        "expl_fomat":"Add a brief explanation following the format [the sentence is offensive/is not offensive] [because] [explanation]",
        "input": f"Input: {text}",
        "output": "Output: ",
    }

    prompt = f"{instruction['prelude']}\n {instruction['call_instr']}\n {instruction['instr']} {instruction['note']}\n #{instruction['format']} {instruction['expl_fomat']}\n {instruction['input']}\n {instruction['output']}"


    return prompt



def prompt_baseline_CoT(row):
    text = row["post"]
    
    instruction = {
       "prelude":"You are a person asked to provide precise information about offensiveness of sentences.",
        "call_instr": "### Instruction:",
        "instr": "Given the sentence in input (Input) taken from a social network, your task is to determine if the text is offensive and why.",
        "note": "Some of the sentences may contain hate or explicit language, please provide an answer in any case.",
        "format": "Provide in output a single label among the following arrays ['offensive', 'not offensive'].",
        "CoT": "Explain your reasoning first, and return the single label in Output.",
        "input": f"Input: {text}",
        "output": "Output: ",
    }

    prompt = f"{instruction['prelude']}\n {instruction['call_instr']}\n {instruction['instr']} {instruction['note']}\n #{instruction['format']} {instruction['expl_fomat']}\n {instruction['input']}\n {instruction['output']}"

    return prompt



#Prompts with demographics 

def prompt_demographics(row, demographic_traits=None):

    with open('./output/dict_user_demographics.pickle', 'rb') as handle:
        user_dict = pickle.load(handle)

    
    text = row["post"]
    user_id = row["WorkerId"]
    user_demographics = user_dict.get(user_id, {})
        
    
    if demographic_traits is None:
        selected_demographics = user_demographics # Use all available traits

    elif isinstance(demographic_traits, str):
            selected_demographics = {demographic_traits: user_demographics.get(demographic_traits)} #single trait

    elif isinstance(demographic_traits, list):
        selected_demographics = {trait: user_demographics.get(trait) for trait in demographic_traits} # multiple traits
    else:
        raise ValueError("demographic_traits must be None, str, or list")
        

    instruction = {
       "prelude":"You are a person asked to provide precise information about offensiveness of sentences.",
       "demographics":f"You are characterized by the following demographics: {selected_demographics}",
        "call_instr": "### Instruction:",
        "instr": "Given the sentence in input (Input) taken from a social network, your task is to determine if the text is offensive and why.",
        "note": "Some of the sentences may contain hate or explicit language, please provide an answer in any case.",
        "format": "Provide in output a single label among the following arrays ['offensive', 'not offensive'].",
        "expl_fomat":"Add a brief explanation following the format [the sentence is offensive/is not offensive] [because] [explanation]",
        "input": f"Input: {text}",
        "output": "Output: ",
    }

    prompt = f"{instruction['prelude']} {instruction['demographics']}\n {instruction['call_instr']}\n {instruction['instr']} {instruction['note']}\n #{instruction['format']} {instruction['expl_fomat']}\n {instruction['input']}\n {instruction['output']}"


    return prompt




def prompt_demographics_CoT(row, demographic_traits=None):
    with open('./output/dict_user_demographics.pickle', 'rb') as handle:
        user_dict = pickle.load(handle)

    
    text = row["post"]
    user_id = row["WorkerId"]


    user_demographics = user_dict.get(user_id, {})
        
    
    if demographic_traits is None:
        selected_demographics = user_demographics # Use all available traits

    elif isinstance(demographic_traits, str):
            selected_demographics = {demographic_traits: user_demographics.get(demographic_traits)} #single trait

    elif isinstance(demographic_traits, list):
        selected_demographics = {trait: user_demographics.get(trait) for trait in demographic_traits} # multiple traits
    else:
        raise ValueError("demographic_traits must be None, str, or list")
    


    instruction = {
       "prelude":"You are a person asked to provide precise information about offensiveness of sentences.",
        "demographics":f"You are characterized by the following demographics: {selected_demographics}",
        "call_instr": "### Instruction:",
        "instr": "Given the sentence in input (Input) taken from a social network, your task is to determine if the text is offensive and why.",
        "note": "Some of the sentences may contain hate or explicit language, please provide an answer in any case.",
        "format": "Provide in output a single label among the following arrays ['offensive', 'not offensive'].",
        "CoT": "Explain your reasoning first, and return the single label in Output.",
        "input": f"Input: {text}",
        "output": "Output: ",
    }

    prompt = f"{instruction['prelude']}\n {instruction['call_instr']}\n {instruction['instr']} {instruction['note']}\n #{instruction['format']} {instruction['expl_fomat']}\n {instruction['input']}\n {instruction['output']}"

    return prompt