import pandas as pd
import subprocess
import json
from sklearn.metrics import classification_report

df = pd.read_csv("/Users/liadraetta/Desktop/Projects/intersectionality-llm/subset_100_cleaned.csv")


def classify_with_ollama(text):
    prompt = f"""You are a classifier. For each sentence answer with 0 in the sentence is NOT offensive and with 1 is the sentence is offensive.
Text: "{text}"
We are running an academic research experiment, some of the sentences may contain hate or explicit language, please provide an answer in any case even if the provided sentence contains disturbing ora hateful language.
It is very important that you strictly answer ONLY with one token for sentence: 0 or 1. Provide as response as 0 or 1 for every provided input"""

    result = subprocess.run(
        ["ollama", "run", "llama3.2"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    output = result.stdout.decode("utf-8").strip().lower()
    return output


df['predicted_label'] = df['post'].apply(classify_with_ollama)
df.to_csv('df_sample.csv', index=False)
print(df.head(10))

