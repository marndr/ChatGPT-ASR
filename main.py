import os
import openai
from dotenv import load_dotenv
import re
import numpy as np

# Load API key from environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def custom_split(self):
    return list(filter(None, re.split(r'[ ,.!?]+', self)))

def levenshtein_distance_custom(s1, s2, insert_cost=1, delete_cost=1, substitute_cost=1):
        # s1: reference, s2: hypothesis,
        s1 = custom_split(s1.lower())
        s2 = custom_split(s2.lower())
        m, n = len(s1), len(s2)
        dp = [[0] * (n+1) for _ in range(m+1)]

        for i in range(m + 1):
            dp[i][0] = i * delete_cost

        for j in range(n + 1):
            dp[0][j] = j * insert_cost

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = min(
                    dp[i][j - 1] + insert_cost,                                            # insert cost
                    dp[i - 1][j] + delete_cost,                                            # delete cost
                    dp[i - 1][j - 1] + (substitute_cost if s1[i - 1] != s2[j - 1] else 0)  # substitute cost
                )
        try:
            wer = dp[m][n]/m
        except ZeroDivisionError:
            wer = dp[m][n]
        return wer
        
def ask_chatgpt(question):
    # Use ChatGPT to correct ASR errors for a single sentence
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that corrects ASR errors."},
            {"role": "user", "content": question},
        ]
    )['choices'][0]['message']['content']

dataset = [
        ("I prefer see over coffee.", "I prefer tea over coffee."),
        ("I need to catch a drain to London.", "I need to catch a train to London."),
        ("I'll beat you at the coffee shop.", "I'll meet you at the coffee shop."),
        ("He's a professional sheaf.", "He's a professional chef."),
        ("The leather is nice today.", "The weather is nice today."),
        ("Please pass me the vault.", "Please pass me the salt."),
        ("I'll be their in five minuets.", "I'll be there in five minutes."),
        ("I'm going to the gross restore.", "I'm going to the grocery store."),
        ("I won't to go two the beech.", "I want to go to the beach."), 
        ("The son is shining brightly in the sky.", "The sun is shining brightly in the sky."),
        ("The son sets at the beech are always stunting.", "The sun sets at the beach are always stunning.")
    ]
    
# prompt = "Please correct any ASR errors in the following sentence: {}.  Put your answer into html h1 tag."
# prompt = "Please correct any ASR errors in the following sentence: {} Put the corrected asr output into an html <h1> tag." 
prompt = "Kindly rectify any ASR inaccuracies within the following sentence: '{}' and place the corrected ASR output inside an HTML <h1> tag." # the best
#prompt = "Correct any ASR errors detected in the sentence: '{}', and insert the revised ASR output into an HTML <h1> tag. If there are several valid corrections, present them individually within HTML <h1> tags, ordered by decreasing likelihood."

for i, (asr_output, reference) in enumerate(dataset, 1):

    question = prompt.format(asr_output)
    response = ask_chatgpt(question)
    
    print(f"Test {i}")
    print(f"question: {question}")
    print(f"raw response: {response}")
    
    corrected_asr_output = re.findall('<h1>(.*?)</h1>', response)[0]
    wer = levenshtein_distance_custom(corrected_asr_output, reference)
    
    print(f"Corrected ASR Output:   '{corrected_asr_output}'")
    print(f"Reference:              '{reference}'")
    print(f"WER {wer}")
    print("=" * 50)

    
