import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def correct_asr_errors(asr_outputs):
    corrected_outputs = []

    for asr_output in asr_outputs:
        user_prompt = f"Correct the ASR error in the following sentence: '{asr_output}'"

        # Use ChatGPT to correct ASR error
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5 Turbo model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that corrects ASR errors."},
                {"role": "user", "content": user_prompt},
            ]
        )

        # Extract and store the corrected sentence
        corrected_sentence = response['choices'][0]['message']['content']
        corrected_outputs.append(corrected_sentence)

    return corrected_outputs

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
        "The son is shining brightly.",
]

    corrected_results = correct_asr_errors(asr_outputs)

    for i, (original, corrected) in enumerate(zip(asr_outputs, corrected_results)):
        print(f"ASR Input {i + 1}: '{original}'")
        print(f"Corrected Output {i + 1}: '{corrected}'")
        print("=" * 50)


