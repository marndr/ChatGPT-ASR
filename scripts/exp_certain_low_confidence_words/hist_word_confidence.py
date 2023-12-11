import os
import json
import matplotlib.pyplot as plt

from chat_gpt_asr.utils import confidence_score_word_level
from dotenv import load_dotenv

load_dotenv()
Root = os.getenv("ROOT_PATH")

l = os.path.join(Root, "data/transcriptions/whisper_tiny_librispeech_dev-clean-full.json") 
OUTPUT_FILE = os.path.join(Root,"results/results_certain_low_confidence_words/histogram_word_confidence.png")


with open(l , "r") as f:
    json_obj=f.read()
    items=json.loads(json_obj)

confidences = []
for item in items:
    processed = confidence_score_word_level(item["asr_transcription"])
    for word in processed["words"]:
        confidences.append(word["confidence"])


plt.hist(confidences, bins=20)

plt.xlabel("Word confidence")
plt.ylabel("count")
plt.title("Histogram of word confidence")

plt.savefig(OUTPUT_FILE)

