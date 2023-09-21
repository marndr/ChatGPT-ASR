import os
import openai
from dotenv import load_dotenv
import re
import json
from jiwer import wer as jiwer_wer, cer as jiwer_cer

# Load API key 
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
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
    messages = [
        {'role': 'system', 'content': f"""You are a helpful assistant that corrects ASR errors. \
        You will be provided with the ASR output text delimited with {delimiter} characters. \
        Provide multiple suggestions where the errors in the ASR text are corrected. Provide your output in json format with the \
        keys: corrected_ASR_output and probability. Probability shows the likelihood of each correction. \
        Do not output any additional text that is not in JSON format. \
        Do not write any explanatory text after outputting the requested JSON.
            """},
        {'role': 'user', 'content': f'{delimiter}I meet pizza.{delimiter}'},
        {'role': 'assistant', 'content': '[{"response": "I eat pizza.", "probability": "0.99"}]'},
        {'role': 'user', 'content': f'{delimiter}{prompt}{delimiter}'}
    ]

    corrected_ASR_output = get_completion_from_messages(messages)
    corrected_ASR_output = json.loads(corrected_ASR_output)

    # Sort list in-place and returns None
    corrected_ASR_output.sort(key=lambda x: x["probability"], reverse=True)

    response = corrected_ASR_output[0]["response"]

    # Calculate CER, SER, and WER using jiwer
    CER = jiwer_cer(response, reference) * 100
    WER = jiwer_wer(response, reference) * 100

    print(f"Test {i}")
    print(f"ASR output:            {prompt}")
    # print(f"Suggested corrections: {json.dumps(corrected_ASR_output, indent=2)}")
    print(f"Corrected ASR output:  {response}")
    print(f"Reference:             {reference}")
    print(f"CER: {CER:.2f}%")
    #print(f"SER: {SER:.2f}%")
    print(f"WER: {WER:.2f}%")
    print("=" * 50)

    f_out.write(f"""
    ## Test {i}
    ASR output: {prompt}
    Corrected ASR output:  {response}
    Reference:             {reference}
    CER: {CER:.2f}%
    SER: {SER:.2f}%
    WER: {WER:.2f}%
    ---
    """)

f_out.close()

