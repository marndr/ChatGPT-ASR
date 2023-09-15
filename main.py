import os
import openai
from dotenv import load_dotenv
import re
import numpy as np
import json

# Load API key 
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a custom word split function
def custom_split(self):
    return list(filter(None, re.split(r'[ ,.!?]+', self)))

# adapted from this link:https://github.com/analyticsinmotion/werpy/blob/main/werpy/metrics.py#L98
def levenshtein_distance_custom(s1, s2, insert_cost=1, delete_cost=1, substitute_cost=1):
        # Preprocess and split the input strings
        s1 = custom_split(s1.lower())
        s2 = custom_split(s2.lower())
        
        # Initialize a dynamic programming matrix
        m, n = len(s1), len(s2)
        dp = [[0] * (n+1) for _ in range(m+1)]
        
        # Initialize the first row and column of the matrix
        for i in range(m + 1):
            dp[i][0] = i * delete_cost

        for j in range(n + 1):
            dp[0][j] = j * insert_cost
        
        # Fill in the matrix with minimum edit distances
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = min(
                    dp[i][j - 1] + insert_cost,                                            # insert cost
                    dp[i - 1][j] + delete_cost,                                            # delete cost
                    dp[i - 1][j - 1] + (substitute_cost if s1[i - 1] != s2[j - 1] else 0)  # substitute cost
                )
        # Calculate Word Error Rate (WER) and handle the case of zero division       
        try:
            wer = dp[m][n]/m
        except ZeroDivisionError:
            wer = dp[m][n]
        return wer


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


delimiter = "####"
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

f_out = open("results.md", "w")
for i, (prompt, reference) in enumerate(dataset, 1):

    messages =  [  
        {'role':'system', 'content': f"""You are a helpful assistant that corrects ASR errors. \
    You will be provided with the ASR output text delimited with {delimiter} characters. \
    Provide multiple suggestions where the errors in the ASR text is corrected. Provide your output in json format with the \
    keys: corrected_ASR_output and probability. Probability shows the likelihood of each correction. \
    Do not output any additional text that is not in JSON format. \
    Do not write any explanatory text after outputting the requested JSON.
        """},    
        {'role':'user', 'content':f'{delimiter}I meet pizza.{delimiter}'},   
        {'role':'assistant', 'content':'[{"response": "I eat pizza.", "probability": "0.99"}]'},   
        {'role':'user', 'content':f'{delimiter}{prompt}{delimiter}'}
    ]
    corrected_ASR_output = get_completion_from_messages(messages)
    corrected_ASR_output = json.loads(corrected_ASR_output)
    # sort list in-place and returns None
    list.sort(corrected_ASR_output, key=lambda x: x["probability"], reverse=True) 
    
    response = corrected_ASR_output[0]["response"]

    wer = levenshtein_distance_custom(response, reference) 
    
    print(f"Test {i}")
    print(f"ASR output:            {prompt}")
    # print(f"Suggested corrections: {json.dumps(corrected_ASR_output, indent=2)}")
    print(f"corrected ASR output:  {response}")
    print(f"Reference:             {reference}")
    print(f"WER {wer}")
    print("=" * 50)

    f_out.write(f"""
    ## Test {i}
    ASR output: {prompt}
    corrected ASR output:  {response}
    Reference:             {reference}
    WER {wer}
    ---
    """)

f_out.close()

