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
        "I have a black cat.",
        "I have a black hat.",
        "The weather is nice today.",
        "The leather is mice today.",
        "She plays the piano beautifully.",
        "She plays the piano duty fully.",
        "Please pass me the salt.",
        "Please past me the vault.",
        "I'll be there in five minutes.",
        "I'll be their in five minuets.",
        "I'm going to the grocery store.",
        "I'm going to the gross restore.",
    ]

    # Prompt for ChatGPT to correct ASR errors
    prompts = [
        "Correct the ASR error in the following sentence: 'I prefer see over coffee.'",
        "Is there any ASR error in the sentence: 'I prefer see over coffee.'?",
    ]

    for prompt in prompts:
        for asr_output in asr_outputs:
            formatted_prompt = prompt.replace(
                "I prefer see over coffee.", asr_output
            ).replace(
                "I prefer tea over coffee.", asr_output
            )  # Replace example ASR outputs in the prompt
        
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



