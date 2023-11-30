import os
import json
from jiwer import wer, cer
from tqdm import tqdm
from dotenv import load_dotenv
from chat_gpt_asr.utils import remove_punctuations, ser
from numpy import round

load_dotenv()
Root = os.getenv("ROOT_PATH")

l = os.path.join(Root, "results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/whisper_corrected_transcriptions_Thresh=0.8.json")
output_file = os.path.join(Root, "results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/whisper_sorted_transcriptions_Thresh=0.8.json")

with open(l, "r") as f:
    json_obj = f.read()
    data = json.loads(json_obj)

results_list = []

for d in tqdm(data):
    asr_text = d["asr_transcription"]["text"]
    corrected_text = d["corrected_asr_transcription"]
    reference_text = d["reference_transcription"]

    # Calculate WER and CER
    original_wer = wer(asr_text, reference_text)*100
    corrected_wer = wer(corrected_text, reference_text)*100
    
    original_cer = cer(asr_text, reference_text)*100
    corrected_cer = cer(corrected_text, reference_text)*100

    # Compare WER and CER
    if corrected_wer < original_wer and corrected_cer > original_cer:
        result_dict = {
            "text": asr_text,
            "corrected_text": corrected_text,
            "reference_transcription": reference_text,
            "original_wer": round(original_wer,2),
            "corrected_wer": round(corrected_wer,2),
            "original_cer": round(original_cer,2),
            "corrected_cer": round(corrected_cer,2)
        }
        results_list.append(result_dict)

# Sort based on WER
sorted_list = sorted(results_list, key=lambda x: x["corrected_wer"], reverse=True)

with open(output_file, "w") as f:
        json_str = json.dumps(sorted_list, indent=2)
        f.write(json_str)
        


