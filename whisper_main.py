import os
import re
import openai
import whisper_timestamped as whisper
from dotenv import load_dotenv
from jiwer import wer as jiwer_wer, cer as jiwer_cer
import json

delimiter = "####"

# Load API key 
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
  
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]

def remove_punctuations(s):
    l = list(filter(None, re.split(r'[ ,.!?]+', s)))
    return " ".join(l)


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

def transcribe(audio_filename):
    audio = whisper.load_audio(audio_filename)
    model = whisper.load_model("tiny", device="cpu")
    result = whisper.transcribe(model, audio, language="en")
    return result["text"]

# Function to calculate SER
def calculate_SER(ref, hyp):
    pass
    
# Read audio filenames and transcriptions
root = "/home/mnaderi/Documents/thesis/whisperii/LibriSpeech/dev-clean"
librispeech_data = read_librispeech_transcriptions(root_folder=root)

# Lists to store original and corrected ASR transcriptions
original_asr_transcriptions = []
corrected_asr_transcriptions = []

f_out = open("results_whisper.md", "w")
for i, (audio_path, reference_transcription) in enumerate(librispeech_data.items()):

    asr_transcription  = transcribe(audio_path)
    messages = [
        {'role': 'system', 'content': f"""You are a helpful assistant that corrects ASR errors. \
        You will be provided with the ASR output text delimited with {delimiter} characters. \
        Provide multiple suggestions with the errors in the ASR text being corrected. Provide your output in json format with the \
        keys: response and probability. response is the text with ASR errors being corrected and probability shows the likelihood of each correction. \
        Do not output any additional text that is not in JSON format. \
        Do not write any explanatory text after outputting the requested JSON.
            """},
        {'role': 'user', 'content': f'{delimiter}I meet pizza.{delimiter}'},
        {'role': 'assistant', 'content': '[{"response": "I eat pizza.", "probability": 0.99}]'},
        {'role': 'user', 'content': f'{delimiter}{asr_transcription}{delimiter}'}
    ]

    corrected_ASR_output = get_completion_from_messages(messages)
    try:
        corrected_ASR_output = json.loads(corrected_ASR_output)
    except json.decoder.JSONDecodeError:
        print("chatgpt seems to fail to produce output in desired format, here is its output: ", corrected_ASR_output, "original asr transcript: ", asr_transcription)
        print("skipping this iteration")
        continue 

    # Sort list in-place and returns None
    corrected_ASR_output.sort(key=lambda x: x["probability"], reverse=True)

    corrected_asr_transcription = corrected_ASR_output[0]["response"]

    # Calculate CER and WER using jiwer for both original and corrected ASR transcriptions
    asr_transcription = asr_transcription.lower()
    asr_transcription = remove_punctuations(asr_transcription )
    
    corrected_asr_transcription = corrected_asr_transcription.lower()
    corrected_asr_transcription = remove_punctuations(corrected_asr_transcription )
    
    reference_transcription = reference_transcription.lower()
    reference_transcription = remove_punctuations(reference_transcription)
    
    CER = jiwer_cer(corrected_asr_transcription , reference_transcription) * 100
    
    WER = jiwer_wer(corrected_asr_transcription , reference_transcription) * 100
    
    # SER = calculate_SER(corrected_asr_transcription , reference_transcription) * 100
    
    CER_original = jiwer_cer(asr_transcription , reference_transcription) * 100
    
    WER_original = jiwer_wer(asr_transcription , reference_transcription) * 100
    
    # SER_original = calculate_SER(asr_transcription , reference_transcription) * 100
    
    
    print("---")
    print(f"i: {i}\n")
    print(f"ASR transcription:            {asr_transcription}\n")
    # print(f"Suggested corrections: {json.dumps(corrected_ASR_output, indent=2)}")
    print(f"Corrected ASR transcription:  {corrected_asr_transcription}\n")
    print(f"Reference transcription:             {reference_transcription}\n")
    print(f"CER: {CER:.2f}%    CER(No chatgpt): {CER_original:.2f}%\n")
    #print(f"SER: {SER:.2f}%    SER(No chatgpt): {SER_original:.2f}%")
    print(f"WER: {WER:.2f}%    WER(No chatgpt): {WER_original:.2f}%\n")
    print("---")

    f_out.write(f"""
    ## Test {i}
    ASR transcription:            {asr_transcription}
    Corrected ASR transcription:  {corrected_asr_transcription}
    Reference transcription:      {reference_transcription}
    CER: {CER:.2f}%    CER(No chatgpt): {CER_original:.2f}%%
    WER: {WER:.2f}%    WER(No chatgpt): {WER_original:.2f}
    ---
    """)

f_out.close()


f_out.close()



