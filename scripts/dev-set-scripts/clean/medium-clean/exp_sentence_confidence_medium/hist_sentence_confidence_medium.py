import json
import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
Root = os.getenv("ROOT_PATH")

l = os.path.join(
    Root,
    "results/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_without_sentence_confidence_medium/corrected_transcriptions_sentence_confidence_medium.json",
)
OUTPUT_FILE = os.path.join(
    Root,
    "results/results_clean/results_medium/results_sentence_confidence_medium/hist_sentence_confidence_medium.png",
)

with open(l) as f:
    json_obj = f.read()
    items = json.loads(json_obj)

confidences = []
for item in items:
    if item["corrected_asr_transcription"] is None:
        continue

    confidence = item["asr_transcription"]["confidence_score"]
    confidences.append(confidence)


plt.hist(confidences, bins=20)
plt.xlabel("sentence confidence")
plt.ylabel("count")
plt.title("Histogram of sentence confidence (medium, clean)")
plt.yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])

plt.savefig(OUTPUT_FILE)
