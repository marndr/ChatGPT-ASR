import matplotlib.pyplot as plt
import os
import json

root = "/home/mnaderi/Documents/thesis/chat-gpt-asr"
l=os.path.join(root,"results/results_GPT-3.5-Turbo/results_sentence_confidence/results_without_sentence_confidence/whisper_corrected_transcriptions.json")
OUTPUT_FILE = os.path.join(root,"results/experiment_GPT-3.5-Turbo/results_sentence_confidence/results_finding_best_sentece_confidence/plots/histogram_sentence_confidence_plot.png")

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
plt.xlabel("sentence confidence")
plt.ylabel("count")
plt.savefig(OUTPUT_FILE)

