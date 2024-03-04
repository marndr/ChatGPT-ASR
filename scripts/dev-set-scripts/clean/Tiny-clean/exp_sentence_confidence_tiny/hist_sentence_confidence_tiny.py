import matplotlib.pyplot as plt
import os
import json
from dotenv import load_dotenv

load_dotenv()
Root = os.getenv("ROOT_PATH")

l=os.path.join(Root,"results/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json")
OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_sentence_confidence_tiny/hist_sentence_confidence_tiny.png")

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
plt.title("Histogram of sentence confidence (tiny, clean)")
plt.yticks([0,100,200,300,400,500,600,700,800,900])

plt.savefig(OUTPUT_FILE)

