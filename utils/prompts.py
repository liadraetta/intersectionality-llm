import ast
import pandas as pd
from collections import defaultdict


class Prompts:
    def __init__(self):
        self.dataset = pd.read_csv("intersectionality-llm/dataset/AnnAttDataset.csv")
        self.base_instruction = {
            # "prelude": "You are a person asked to provide precise information about offensiveness of sentences.",
            "person_info": "You are a {race} {gender} {political_leaning}",
            "prelude": " asked to provide precise information about offensiveness of a sentence.",
            "call_instr": "### Instruction:",
            "instr": "Given the sentence in input (Input) taken from a social network, your task is to determine if the text is offensive and why.",
            "note": "Some of the sentences may contain hate or explicit language, please provide an answer in any case.",
            "format": "Provide in output (Output) a single label among the following arrays ['offensive', 'not offensive'].",
            "expl_format": "Add a brief explanation following the format [the sentence is offensive/is not offensive] [because] [explanation].",
            "CoT": "Explain your reasoning first, and return a single label. Follow this format [reasoning explanation] [so] [the sentence is offensive/is not offensive].",
            "output": "Output: ",
        }
        self.user_dict = self._build_user_dict()

        self.few_shot = {
            "input_offensive1": "Input: A girl that looked like she worth the world to ya is around the corner getting fucked for a perk #Facts",
            "output_offensive1": "Output: [The sentence is offensive][because][it conveys a stereotypical message suggesting that women use their bodies to obtain benefits and contains vulgar language]",
            "output_offensive_CoT1": "Output: [The sentence conveys a stereotypical message suggesting that women use their bodies to obtain benefits and contains vulgar language][so][the sentence is offensive]",

            "input_not1": "Input: as a black person, i disagree with white people saying nigga",
            "output_not1": "Output: [The sentence is not offensive][because][the person is expressing an opinion about the misuse of the racial slur 'nigga']",
            "output_not_CoT1": "Output: [The person is expressing an opinion about the misuse of the racial slur 'nigga'][so][the sentence is not offensive]",

            "input_offensive2": "Input: to call black people niggas is just funny",
            "output_offensive2": "Output: [The sentence is offensive][because][it promotes the use of a racial slur]",
            "output_offensive_CoT2": "Output: [The sentence promotes the use of a racial slur][so][the sentence is offensive]",

            "input_not2": "Input: bitch it's fucking spring, why is this weather wildin? :triumph:",
            "output_not2": "Output: [The sentence is not offensive][because][it is just a colloquial way to comment on the weather]",
            "output_not_CoT2": "Output: [The sentence is just a colloquial way to comment on the weather][so][the sentence is not offensive]"


        }
    def build_person_info(self, user_demographics_selected):
        """Build person info string based on user demographics."""
        base_person_info = self.base_instruction['person_info']
        person_info = base_person_info.format(
            race=f" {user_demographics_selected.get('race', '')} ",
            gender=f" {user_demographics_selected.get('gender', 'person')} ",
            political_leaning=f" with {user_demographics_selected.get('political')} political leaning" if user_demographics_selected.get('political') else ""
        )
        return person_info.strip()

    
    def _build_user_dict(self):
        #Build user demographics dictionary from dataset.
        df_demographics = self.dataset[["annId", "annotatorGender", "annotatorRace", "annotatorPoliticsBinary"]]
        df_demographics = df_demographics.drop_duplicates(subset="annId")
        
        user_dict = defaultdict(dict)
        for index, row in df_demographics.iterrows():
            user_id = row["annId"]
            user_dict[user_id]["gender"] = row["annotatorGender"]
            user_dict[user_id]["race"] = row["annotatorRace"]
            # user_dict[user_id]["generation"] = row["annotatorGeneration"]
            user_dict[user_id]["political"] = row["annotatorPoliticsBinary"]

        
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
        #person_info_notformatted = self.base_instruction['person_info']
        prelude_notformatted = self.base_instruction['prelude']
        person_info_formatted = self.build_person_info(selected_demographics)  
        prelude = f"{person_info_formatted}{prelude_notformatted}"
        # Remove all extra spaces
        prelude = ' '.join(prelude.split())


        if CoT:
            format_instruction = f"{self.base_instruction['format']} {self.base_instruction['CoT']}"
            output_example_yes1 = f"{self.few_shot['output_offensive_CoT1']}"
            output_example_no1 = f"{self.few_shot['output_not_CoT1']}"
            output_example_yes2 = f"{self.few_shot['output_offensive_CoT2']}"
            output_example_no2 = f"{self.few_shot['output_not_CoT2']}"

        else:
            format_instruction = f"{self.base_instruction['format']} {self.base_instruction['expl_format']}"
            output_example_yes1 = f"{self.few_shot['output_offensive1']}"
            output_example_no1 = f"{self.few_shot['output_not1']}"
            output_example_yes2 = f"{self.few_shot['output_offensive2']}"
            output_example_no2 = f"{self.few_shot['output_not2']}"

        prompt = (f"{prelude}\n "
                 f"{self.base_instruction['call_instr']}\n "
                 f"{self.base_instruction['instr']} {self.base_instruction['note']}\n "
                 f"{format_instruction}\n\n "

                 f"Example 1:\n "
                 f"{self.few_shot['input_offensive1']}\n "
                 f"{output_example_yes1}\n "
                 f"Example 2:\n "
                 f"{self.few_shot['input_not1']}\n "
                 f"{output_example_no1}\n "
                 f"Example 3:\n "
                 f"{self.few_shot['input_offensive2']}\n "
                 f"{output_example_yes2}\n "
                 f"Example 4:\n "
                 f"{self.few_shot['input_not2']}\n "
                 f"{output_example_no2}\n\n "

                 f"Example to label:\n "
                 f"Input: {text}\n "
                 f"{self.base_instruction['output']}")
        
        return prompt
    


    def get_prompt(self, row, demographic_traits=None, CoT=False):
        """Generate a prompt for the given row."""
        text = row["tweet"]
        user_id = row["annId"]
        
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