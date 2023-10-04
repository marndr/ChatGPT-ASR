from module import chatgpt, \
    read_librispeech_transcriptions, \
    read_dummy_transcriptions,\
    transcribe, remove_punctuations,\
    get_messages, ser
from dotenv import load_dotenv
import json
import openai
import os
import argparse
from jiwer import cer
from jiwer import wer

parser = argparse.ArgumentParser(description="ASR Correction")
parser.add_argument("-d", "--dataset", choices=["librispeech", "dummy"], default="librispeech", help="Select the dataset (librispeech or dummy)")
args = parser.parse_args()
    
delimiter = "####"

# Load API key 
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
  
if args.dataset=="librispeech":
    root = "/home/mnaderi/Documents/thesis/whisperii/LibriSpeech/dev-clean"
    data = read_librispeech_transcriptions(root_folder=root)
    f_out = open("results_whisper.md", "w")
    
elif args.dataset=="dummy":
    data = read_dummy_transcriptions()
    f_out = open ("results_dummy.md", "w")

ref_l = []
hyp_l= []
hyp_l_original = []

ser_corrected_chatgpt, ser_original = 0.,0.
count = 0

for i, (audio_path, reference_transcription) in enumerate(data.items()):

    if i > 30:
        break
        
    if args.dataset=="librispeech":
        asr_transcription  = transcribe(audio_path)
    elif args.dataset == "dummy":
        asr_transcription  = audio_path
    
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
    
    
    ref_l.append(remove_punctuations(reference_transcription.lower()))
    hyp_l.append(remove_punctuations(corrected_asr_transcription.lower()))
    hyp_l_original.append(remove_punctuations(asr_transcription.lower()))
    
    SER_chatgpt_, SER_original_ = ser(asr_transcription, corrected_asr_transcription, reference_transcription)
    
    count += 1
    ser_corrected_chatgpt += SER_chatgpt_
    ser_original += SER_original_

    
    print("---")
    print(f"i: {i}\n")
    print(f"ASR transcription:            {asr_transcription}\n")
    # print(f"Suggested corrections: {json.dumps(corrected_ASR_output, indent=2)}")
    print(f"Corrected ASR transcription:  {corrected_asr_transcription}\n")
    print(f"Reference transcription:      {reference_transcription}\n")
    print("---")

    f_out.write(f"""
    ## Test {i}
    ASR transcription:            {asr_transcription}
    Corrected ASR transcription:  {corrected_asr_transcription}
    Reference transcription:      {reference_transcription}
    ---
""")


wer_corrected_chatgpt= wer(hyp_l,ref_l)
wer_original= wer(hyp_l_original, ref_l)
cer_corrected_chatgpt=cer(hyp_l,ref_l)
cer_original = cer(hyp_l_original, ref_l)
ser_corrected_chatgpt /=count
ser_original /= count


print(f"WER_original is:           {wer_original:.04f}\n")
print(f"WER_corrected_chatgpt is:  {wer_corrected_chatgpt:.04f}\n")
print(f"CER_original is:           {cer_original:.04f}\n")
print(f"CER_corrected_chatgpt is:  {cer_corrected_chatgpt:.04f}\n")
print(f"SER_original is:           {ser_original:.04f}\n")
print(f"SER_corrected_chatgpt is:  {ser_corrected_chatgpt:.04f}\n")

f_out.write(f"""
    WER_original:           {wer_original:.04f}%    
    WER_corrected_chatgpt:  {wer_corrected_chatgpt:.04f}%
    CER_original:           {cer_original:.04f}%    
    CER_corrected_chatgpt:  {cer_corrected_chatgpt:.04f}%
    SER_original:           {ser_original:.04f}%    
    SER_corrected_chatgpt:  {ser_corrected_chatgpt:.04f}%
    ---
""")
f_out.close()


