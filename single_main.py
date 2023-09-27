
from module import chatgpt, \
    read_librispeech_transcriptions, \
    transcribe, \
    get_messages, evaluate, read_dummy_transcriptions
from dotenv import load_dotenv
import json
import openai
import os

delimiter = "####"

# Load API key 
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
  

# Read audio filenames and transcriptions
root = "/home/mnaderi/Documents/thesis/whisperii/LibriSpeech/dev-clean"

# comment/uncomment based on dataset
librispeech_data = read_librispeech_transcriptions(root_folder=root)
#dummy_data = read_dummy_transcriptions()

# comment/uncomment based on dataset
f_out = open("results_whisper.md", "w")
#f_out = open ("results_dummy.md", "w")

WER_avg , WER_original_avg, CER_avg, CER_original_avg = 0.,0.,0.,0.
count = 0

# comment/uncomment based on dataset
for i, (audio_path, reference_transcription) in enumerate(librispeech_data.items()):
#for i, (asr_transcription, reference_transcription) in enumerate(dummy_data):

    # comment/uncomment based on dataset
    asr_transcription  = transcribe(audio_path)
    
    messages = get_messages(asr_transcription, delimiter)
    corrected_ASR_output = chatgpt(messages, openai)
    try:
        corrected_ASR_output = json.loads(corrected_ASR_output)
    except json.decoder.JSONDecodeError:
        print("chatgpt seems to fail to produce output in desired format, here is its output: ", corrected_ASR_output, "original asr transcript: ", asr_transcription)
        print("skipping this iteration")
        continue 

    # Sort list in-place and returns None
    corrected_ASR_output.sort(key=lambda x: x["probability"], reverse=True)
    corrected_asr_transcription = corrected_ASR_output[0]["response"]
    
    WER, WER_original, CER, CER_original = evaluate(asr_transcription, corrected_asr_transcription, reference_transcription)
    
    count += 1
    WER_avg += WER
    WER_original_avg += WER_original
    CER_avg += CER
    CER_original_avg += CER_original
    
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
    
    
WER_avg /= count
WER_original_avg /= count
CER_avg  /= count
CER_original_avg /= count

print(f"average WER is {WER_avg}\n")
print(f"average WER original is {WER_original_avg}\n")
print(f"average CER is {CER_avg}\n")
print(f"average CER original is  {CER_original_avg}\n")

f_out.write(f"average WER is {WER_avg}\n")
f_out.write(f"average WER original is {WER_original_avg}\n")
f_out.write(f"average CER is {CER_avg}\n")
f_out.write(f"average CER original is  {CER_original_avg}\n")

f_out.close()


