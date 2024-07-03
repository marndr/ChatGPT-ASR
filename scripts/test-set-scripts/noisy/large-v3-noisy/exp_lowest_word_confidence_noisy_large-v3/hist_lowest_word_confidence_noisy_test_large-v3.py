"""
Module: lowest_word_confidence_histogram

This module loads a JSON file containing ASR (Automatic Speech Recognition) transcription data,
extracts confidence scores, and plots a histogram of the sentence confidence scores.
The histogram is then saved as an image file.

Example:
    python hist_lowest_word_confidence_noisy_test_large-v3.py
"""

import json
import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
Root = os.getenv("ROOT_PATH")

l = os.path.join(
    Root,
    "results/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-1106/results_without_lowest_word_confidence_noisy_large-v3/corrected_transcriptions_lowest_word_confidence_noisy_large-v3.json",
)
OUTPUT_FILE = os.path.join(
    Root,
    "results/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/hist_lowest_word_confidence_noisy_large-v3.png",
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
plt.title("Histogram of lowest-word confidence(noisy, large-v3)")
plt.yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])

plt.savefig(OUTPUT_FILE)
