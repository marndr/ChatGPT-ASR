import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def correct_outputs(dataset):
    corrected_outputs = []

    for asr_output, reference_transcription in dataset:
        user_prompt = f'''Correct the ASR error in the following sentence: '{asr_output}'
                      The reference transcription is: '{reference_transcription}'.'''
                      
        # Use ChatGPT to correct ASR errors
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that corrects ASR errors."},
                {"role": "user", "content": user_prompt},
            ]
        )

        # Extract the corrected output from the model's response
        corrected_output = response['choices'][0]['message']['content']
        corrected_outputs.append(corrected_output)

    return corrected_outputs

def evaluate_corrected_outputs(dataset, corrected_outputs):
    num_correct = sum(corrected_output == reference_transcription for corrected_output, (_, reference_transcription) in zip(corrected_outputs, dataset))
    percentage_correct = (num_correct / len(dataset)) * 100
    return percentage_correct

if __name__ == "__main__":
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
        ("The son sets at the beech are always stunting.", "The sun sets at the beach are always stunting.")
    ]

    corrected_outputs = correct_outputs(dataset)
    percentage_correct = evaluate_corrected_outputs(dataset, corrected_outputs)

    for i, (asr_output, reference_transcription) in enumerate(dataset, 1):
        corrected_output = corrected_outputs[i - 1]
        print(f"ASR Input {i}: '{asr_output}'")
        print(f"Corrected Output {i}: '{corrected_output}'")
        print(f"Reference Transcription {i}: '{reference_transcription}'")  
        print("=" * 50)

    print(f"Percentage Correct: {percentage_correct:.2f}%")


