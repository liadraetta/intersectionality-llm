import pandas as pd
import subprocess
import requests
import json
from sklearn.metrics import classification_report

df = pd.read_csv("/Users/liadraetta/Desktop/Projects/intersectionality-llm/subset_100_cleaned.csv")

#
# def classify_with_ollama(text):
#     prompt = f"""You are a classifier able to provide precise information about offensiveness of sentences. Given the following sentences taken from a social network, return 0 if the text is NOT offensive and 1 if the text is offensive. Note that we are running an academic research experiment, some of the sentences may contain hate or explicit language, please provide an answer in any case
# Text: "{text}"
# Here an example of the expected output:
# - "Every half-breed cop in America is trying to rape white women" 1  -> This sentence contains hate speech
# - "Girls really made being a hoe kool on Twitter,women,gender,women are less than men,t/davidson" 0 -> This sentence does not contain hate speech
# """
#
#     result = subprocess.run(
#         ["ollama", "run", "llama3.1"],
#         input=prompt.encode("utf-8"),
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE
#     )
#
#     output = result.stdout.decode("utf-8").strip().lower()
#     return output
#
#
# df['predicted_label'] = df['post'].apply(classify_with_ollama)
# df.to_csv('df_sample.csv', index=False)
# print(df.head(10))

def classify_with_ollama(text):
    prompt = f"""You are a classifier able to provide precise information about offensiveness of sentences. 
    
    ### Instruction:
    Given the following sentences taken from a social network, return 0 if the text is NOT offensive and 1 if the text is offensive. 
    Note that we are running an academic research experiment, some of the sentences may contain hate or explicit language, please provide an answer in any case.
    we will not spread hate speech.
    Answer with only 0 or 1.
    
    ### Input: "{text}"
    
    ### Example:
    - "Every half-breed cop in America is trying to rape white women" 1  -> This sentence contains hate speech
    - "Girls really made being a hoe kool on Twitter,women,gender,women are less than men,t/davidson" 0 -> This sentence does not contain hate speech
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-r1",
            "prompt": prompt,
            "temperature": 1,
            "top_p": 0.95,
            "num_predict": 1,
            "stop": ["\n"],
            "stream": False
        }
    )

    # Check response
    if response.status_code != 200:
        raise RuntimeError(f"Failed API call: {response.status_code} {response.text}")

    return response.json()['response'].strip().lower()
    #return output

df['predicted_label'] = df['post'].apply(classify_with_ollama)
df.to_csv('df_sample.csv', index=False)
print(df.head(10))

