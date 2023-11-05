import os
from chat_gpt_asr.utils import remove_punctuations, ser
import json
from jiwer import cer, wer
from tqdm import tqdm
import argparse

root = "/home/mnaderi/Documents/thesis/chat-gpt-asr"
l=os.path.join(root,"results/experiment_without_confidence/whisper_corrected_transcriptions.json")
OUTPUT_FILE = os.path.join(root,"results/experiment_finding_best_confidence/results_whisper.md")

with open(l , "r") as f:
    json_obj=f.read()
    data=json.loads(json_obj)
    

        
def evaluate_with_thresh(items, thresh):
    total_wer = 0
    total_cer = 0
    total_ser = 0
    total_count = 0
 
    for item in items:
        
        asr_transcription = remove_punctuations(item["asr_transcription"]["text"].lower())
        corrected_asr_transcription = remove_punctuations(item["corrected_asr_transcription"].lower())
        reference_transcription = remove_punctuations(item["reference_transcription"].lower())
        confidence = item["asr_transcription"]["confidence_score"]  
        
        if confidence < thresh:
            Wer = wer([reference_transcription], [corrected_asr_transcription])
            Cer = cer([reference_transcription], [corrected_asr_transcription])
            #Ser = ser(reference_transcription, corrected_asr_transcription)
            Ser = 0
        else:
            Wer = wer([reference_transcription],[ asr_transcription])
            Cer = cer([reference_transcription], [asr_transcription])
            #Ser = ser(reference_transcription, asr_transcription)
            Ser = 0
            
        total_wer += Wer
        total_cer += Cer
        total_ser += Ser
        total_count += 1
    
    if total_count > 0:
        avg_wer = 100 *(total_wer / total_count)
        avg_cer = 100 *(total_cer / total_count)
        avg_ser = 100 *(total_ser / total_count)
    else:
        avg_wer = 0
        avg_cer = 0
        avg_ser = 0
    
    return avg_wer, avg_cer, avg_ser


thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []

for thresh in thresholds:
    Wer, Cer, Ser = evaluate_with_thresh(data, thresh)
    results.append({
        "thresh":thresh,
        "wer": Wer,
        "cer": Cer,
        "ser": Ser   
    })

# Sort results based on WER
results.sort(key=lambda x: x["wer"])

# Print results
for result in results:
    
    print(f'Threshold: {result["thresh"], result["wer"], result["cer"], result["ser"]}')



# Write the JSON data to a file
with open(OUTPUT_FILE, 'w') as f:
    json_obj = json.dumps(results , indent = 2)
    f.write(json_obj)






