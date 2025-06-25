import pandas as pd
from collections import defaultdict
import pickle

import pandas as pd
from collections import defaultdict



import pandas as pd
from collections import defaultdict

class Prompts:
    def __init__(self):
        self.dataset = pd.read_csv("./dataset/FilteredTestSet.csv")
        self.base_instruction = {
            "prelude": "You are a person asked to provide precise information about offensiveness of sentences.",
            "call_instr": "### Instruction:",
            "instr": "Given the sentence in input (Input) taken from a social network, your task is to determine if the text is offensive and why.",
            "note": "Some of the sentences may contain hate or explicit language, please provide an answer in any case.",
            "format": "Provide in output (Output) a single label among the following arrays ['offensive', 'not offensive'].",
            "expl_format": "Add a brief explanation following the format [the sentence is offensive/is not offensive] [because] [explanation]",
            "CoT": "Explain your reasoning first, and return a single label. Follow this format [reasoning explanation] [so] [the sentence is offensive/is not offensive]",
            "output": "Output: ",
        }
        self.user_dict = self._build_user_dict()
    

    
    def _build_user_dict(self):
        """Build user demographics dictionary from dataset."""
        df_demographics = self.dataset[["WorkerId", "annotatorGender", "annotatorRace", "annotatorGeneration"]]
        df_demographics = df_demographics.drop_duplicates(subset="WorkerId")
        
        user_dict = defaultdict(dict)
        for index, row in df_demographics.iterrows():
            user_id = row["WorkerId"]
            user_dict[user_id]["gender"] = row["annotatorGender"]
            user_dict[user_id]["race"] = row["annotatorRace"]
            user_dict[user_id]["generation"] = row["annotatorGeneration"]
        
        return user_dict
    


    def get_demographics(self, user_id, demographic_traits=None):
        """Get selected demographic traits for a user."""
        user_demographics = self.user_dict.get(user_id, {})
        
        if demographic_traits is None:
            return {}
        elif isinstance(demographic_traits, str):
            return {demographic_traits: user_demographics.get(demographic_traits)}
        elif isinstance(demographic_traits, list):
            return {trait: user_demographics.get(trait) for trait in demographic_traits}
        else:
            raise ValueError("demographic_traits must be a str or a list")
    


    def build_prompt(self, text, selected_demographics=None, CoT=False):
        """Build the prompt string."""
        prelude = self.base_instruction['prelude']
        
        # Add demographics if provided and not empty 
        if selected_demographics:
            demographics_str = f" You are characterized by the following demographics: {selected_demographics}"
            prelude += demographics_str
        
        if CoT:
            format_instruction = f"{self.base_instruction['format']} {self.base_instruction['CoT']}"
        else:
            format_instruction = f"{self.base_instruction['format']} {self.base_instruction['expl_format']}"
        

        prompt = (f"{prelude}\n "
                 f"{self.base_instruction['call_instr']}\n "
                 f"{self.base_instruction['instr']} {self.base_instruction['note']}\n "
                 f"{format_instruction}\n "
                 f"Input: {text}\n "
                 f"{self.base_instruction['output']}")
        
        return prompt
    


    def get_prompt(self, row, demographic_traits=None, CoT=False):
        """Generate a prompt for the given row."""
        text = row["post"]
        user_id = row["WorkerId"]
        
        # Get demographics (will be {} if demographic_traits is None)
        selected_demographics = self.get_demographics(user_id, demographic_traits)
        
        return self.build_prompt(text, selected_demographics, CoT)











#####################################           OLD CODE        #########################################################
"""
Example to call the following functions: 

    df["prompt"] = df.apply(prompt_baseline, axis=1)


    df["prompt"] = df.apply(lambda row: prompt_demographics(row, demographic_traits="gender"), axis=1)
    df["prompt"] = df.apply(lambda row: prompt_demographics(row, demographic_traits=["gender", "race"]), axis=1)
    df["prompt"] = df.apply(lambda row: prompt_demographics(row, demographic_traits=None), axis=1) --> All demographics 
"""

"""

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
"""