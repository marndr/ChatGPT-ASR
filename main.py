import os
import openai

from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def test_asr_correction():
    # List of ASR output sentences and their corresponding reference sentences
    asr_outputs = [
        "I prefer tea over coffee.",
        "I prefer see over coffee.",
        "I need to catch a train to London.",
        "I need to catch a drain to London.",
        "I'll meet you at the coffee shop.",
        "I'll beat you at the coffee shop.",
        "He's a professional chef.",
        "He's a professional sheaf.",
    ]

    # Prompt for ChatGPT to correct ASR errors
    prompts = [
        "Correct the ASR error in the following sentence: '{}'",
        "Is there any ASR error in the sentence: '{}'?",
    ]


    for prompt in prompts:
        for asr_output in asr_outputs:
            # Replace '{}' in the prompt with the ASR output sentence
            formatted_prompt = prompt.format(asr_output)
        
            # Call OpenAI's GPT-3 model to correct the ASR error
            response = openai.Completion.create(
            model="text-davinci-003",
            prompt=formatted_prompt,
            temperature=0.6,
            )
        
            # Print the corrected sentence from ChatGPT's response
            corrected_sentence = response.choices[0].text.strip()
            print(f"ASR Input: '{asr_output}'")
            print(f"Prompt: '{formatted_prompt}'")
            print(f"ChatGPT Response: '{corrected_sentence}'")
            print("=" * 50)

if __name__ == "__main__":
    test_asr_correction()

