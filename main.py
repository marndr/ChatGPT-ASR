import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def check_asr_errors(asr_outputs):
    asr_results = []

    for asr_output in asr_outputs:
        user_prompt = f'''Is there any ASR error in the sentence: '{asr_output}'?
                      Correct the ASR error in the following sentence: '{asr_output}'.'''
                      

        # Use ChatGPT to check for ASR errors
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that checks for ASR errors."},
                {"role": "user", "content": user_prompt},
            ]
        )

        # Extract the model's response
        response_text = response['choices'][0]['message']['content']
        asr_results.append(response_text)

    return asr_results

if __name__ == "__main__":
    asr_outputs = [
        "I prefer see over coffee.",
        "I need to catch a drain to London.",
        "I'll beat you at the coffee shop.",
        "He's a professional sheaf.",
        "The leather is nice today.",
        "The weather is mice today.",
        "Please pass me the vault.",
        "I'll be their in five minuets.",
        "I'm going to the gross restore.",
        "I won't to go two the beech.",
        "The son is shining brightly in the sky.",
        "The son sets at the beech are always stunting.",
        "I prefer tea over coffee.",
        "I need to catch a train to London.",
        "I'll meet you at the coffee shop.",
        "He's a professional chef.",
        "The weather is nice today.",
        "Please pass me the salt.",
        "I'll be there in five minutes.",
        "I'm going to the grocery store.",
    ]

    asr_results = check_asr_errors(asr_outputs)

    for i, (original, result) in enumerate(zip(asr_outputs, asr_results), 1):
        print(f"ASR Input {i}: '{original}'")
        print(f"ASR Result {i}: '{result}'")
        print("=" * 50)

