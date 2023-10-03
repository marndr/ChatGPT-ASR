import os
import re
import whisper_timestamped as whisper
import json

def read_dummy_transcriptions():
    dataset = {
    "I prefer see over coffee.": "I prefer tea over coffee.",
    "I need to catch a drain to London.": "I need to catch a train to London.",
    "I'll beat you at the coffee shop.": "I'll meet you at the coffee shop.",
    "He's a professional sheaf.": "He's a professional chef.",
    "The leather is nice today.": "The weather is nice today.",
    "Please pass me the vault.": "Please pass me the salt.",
    "I'll be their in five minuets.": "I'll be there in five minutes.",
    "I'm going to the gross restore.": "I'm going to the grocery store.",
    "I won't to go two the beech.": "I want to go to the beach.",
    "The son is shining brightly in the sky.": "The sun is shining brightly in the sky.",
    "The son sets at the beech are always stunting.": "The sun sets at the beach are always stunning."
}
    return dataset


# Function to read audio filenames and their corresponding transcriptions
def read_librispeech_transcriptions(root_folder="."):
    data = {}

    # Regular expression to match audio filenames
    audio_file_pattern = re.compile(r'(\d+-\d+-\d+)\.flac')
    audio_file_pattern_no_ending = re.compile(r'(\d+-\d+-\d+)')

    for root, dirs, files in os.walk(root_folder):
        trans_files = [f for f in files if f.endswith('.trans.txt')]
        for trans_file in trans_files:

            trans_file_path = os.path.join(root, trans_file)
            audio_folder = os.path.dirname(trans_file_path)
            transcription_text = {}

            # Read transcription lines from the .trans.txt file
            with open(trans_file_path, 'r') as trans_file:
                for line in trans_file:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        audio_filename = parts[0]
                        transcription = parts[1]

                        # Find matching audio files using the regular expression
                        audio_match = audio_file_pattern_no_ending.match(audio_filename)
                        if audio_match:
                            audio_basename = audio_match.group(1)
                            audio_abs_path = os.path.join(audio_folder, f"{audio_basename}.flac")
                            if os.path.isfile(audio_abs_path):
                                data[audio_abs_path] = transcription.strip()
                                transcription_text[audio_filename] = transcription.strip()

    return data


def remove_punctuations(s):
    l = list(filter(None, re.split(r'[ ,.!?]+', s)))
    return " ".join(l)


def transcribe(audio_filename):
    audio = whisper.load_audio(audio_filename)
    model = whisper.load_model("tiny", device="cpu")
    result = whisper.transcribe(model, audio, language="en")
    return result["text"]


def preprocess_transcription(trans):
    d={}
    d["text"]= trans["text"]
    words=[]
    for i, seg in enumerate(trans["segments"]):
        words.extend(seg["words"])

    l=[]
    for word in words:
        l.append({"text": word["text"], "confidence": word["confidence"]})
        
    d["words"]= l
    return d

  
def chatgpt(messages, openai, model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]


def get_messages(asr_transcription, delimiter="####"):
    messages = [
        {'role': 'system', 'content': f"""You are a helpful assistant that corrects ASR errors. \
        You will be provided with the ASR output text delimited with {delimiter} characters. \
        Provide multiple suggestions with the errors in the ASR text being corrected. Provide your output in json format with the \
        keys: response and probability. response is the text with ASR errors being corrected and probability shows the likelihood of each correction. \
        Do not correct the grammar in the transcription. \
        Do not change the case, for example lower case or upper case, in the transcription. \
        Do not output any additional text that is not in JSON format. \
        Do not write any explanatory text after outputting the requested JSON. \
            """},
        {'role': 'user', 'content': f'{delimiter}Hilda watched him from her corner, trembling and scarcely breathing, dark shadows going about her eyes.{delimiter}'},
        {'role': 'assistant', 'content': '[{"response": "HILDA WATCHED HIM FROM HER CORNER TREMBLING AND SCARCELY BREATHING DARK SHADOWS GROWING ABOUT HER EYES", "probability": 0.99}]'},
        {'role': 'user', 'content': f'{delimiter}{asr_transcription}{delimiter}'}

    ]   
    return messages


def evaluate_SER(asr_transcription, corrected_asr_transcription, reference_transcription):

    asr_transcription = remove_punctuations(asr_transcription.lower())
    corrected_asr_transcription = remove_punctuations(corrected_asr_transcription.lower())
    reference_transcription = remove_punctuations(reference_transcription.lower())

    SER = 100 - (corrected_asr_transcription == reference_transcription)*100
    SER_original = 100 - (asr_transcription == reference_transcription)*100
    
    
    return SER,SER_original
    
 
