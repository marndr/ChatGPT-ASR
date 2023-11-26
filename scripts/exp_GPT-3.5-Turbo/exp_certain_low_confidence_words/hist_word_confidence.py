import os
import json
import matplotlib.pyplot as plt

from chat_gpt_asr.utils import confidence_score_word_level

root = "/home/mnaderi/Documents/thesis/chat-gpt-asr"
l = os.path.join(root, "data/transcriptions/whisper_tiny_librispeech_dev-clean-full.json") 
OUTPUT_FILE = os.path.join(root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/plots/histogram_confidence_words_plot.png")

with open(l , "r") as f:
    json_obj=f.read()
    items=json.loads(json_obj)

confidences = []
for item in items:
    processed = confidence_score_word_level(item["asr_transcription"])
    for word in processed["words"]:
        confidences.append(word["confidence"])


#plt.figure(figsize=(8, 6))
plt.hist(confidences, bins=20, edgecolor='black')
plt.title('Histogram of Word Confidences')
plt.xlabel('Word confidence')
plt.ylabel('Count')
plt.grid(True)
plt.savefig(OUTPUT_FILE)

