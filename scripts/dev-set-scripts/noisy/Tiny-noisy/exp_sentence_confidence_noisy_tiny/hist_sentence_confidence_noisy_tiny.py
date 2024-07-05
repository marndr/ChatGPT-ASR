# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maryam Naderi  <maryam.naderi@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-chatgpt-asr

"""
Module: sentence_confidence_histogram

This module loads a JSON file containing ASR (Automatic Speech Recognition) transcription data,
extracts confidence scores, and plots a histogram of the sentence confidence scores.
The histogram is then saved as an image file.

Example:
    python hist_sentence_confidence_noisy_tiny.py
"""

import json
import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
Root = os.getenv("ROOT_PATH")

l = os.path.join(
    Root,
    "results/results-dev-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json",
)
OUTPUT_FILE = os.path.join(
    Root,
    "results/results-dev-set/results_noisy/results_tiny/results_sentence_confidence_tiny/hist_sentence_confidence_tiny.png",
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
plt.title("Histogram of sentence confidence (tiny, noisy)")
plt.yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])

plt.savefig(OUTPUT_FILE)
