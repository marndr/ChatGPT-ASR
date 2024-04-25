import json
import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
Root = os.getenv("ROOT_PATH")

l = os.path.join(
    Root,
    "results/results-test-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_tiny/corrected_transcriptions_lowest_word_confidence_tiny.json",
)

OUTPUT_FILE = os.path.join(
    Root,
    "results/results-test-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/hist_lowest_word_confidence_tiny.png",
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

plt.xlabel("lowest-word confidence")
plt.ylabel("count")
plt.title("Histogram of lowest-word confidence(tiny, noisy, test-set)")
# plt.yticks([0,100,200,300,400,500,600,700,800,900])

plt.savefig(OUTPUT_FILE)
