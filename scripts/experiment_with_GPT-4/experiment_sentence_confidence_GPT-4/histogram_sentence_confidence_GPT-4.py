import matplotlib.pyplot as plt
import os
import json

root = "/home/mnaderi/Documents/thesis/chat-gpt-asr"
l=os.path.join(root,"results/results_GPT-4/results_sentence_confidence_GPT-4/results_without_sentence_confidence_GPT-4/whisper_corrected_transcriptions.json")
OUTPUT_FILE = os.path.join(root,"results/results_GPT-4/results_sentence_confidence_GPT-4/results_finding_best_sentence_confidence_GPT-4/plots/histogram_sentence_confidence_plot_GPT-4.png")

with open(l , "r") as f:
    json_obj=f.read()
    items=json.loads(json_obj)

confidences = []    
for item in items:
        
        if item["corrected_asr_transcription"] is None:
            continue 
          
        confidence = item["asr_transcription"]["confidence_score"] 
        confidences.append(confidence)


plt.hist(confidences, bins=20)
plt.xlabel("confidence")
plt.ylabel("count")
plt.savefig(OUTPUT_FILE)

