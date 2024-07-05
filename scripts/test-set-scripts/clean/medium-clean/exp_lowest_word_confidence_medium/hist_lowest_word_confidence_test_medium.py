# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maryam Naderi  <maryam.naderi@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-chatgpt-asr

"""
Module: lowest_word_confidence_histogram

This module loads a JSON file containing ASR (Automatic Speech Recognition) transcription data,
extracts confidence scores, and plots a histogram of the sentence confidence scores.
The histogram is then saved as an image file.

Example:
    python hist_lowest_word_confidence_test_medium.py
"""

import json
import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
Root = os.getenv("ROOT_PATH")

l = os.path.join(
    Root,
    "results/results-test-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_medium/corrected_transcriptions_lowest_word_confidence_medium.json",
)

OUTPUT_FILE = os.path.join(
    Root,
    "results/results-test-set/results_clean/results_medium/results_lowest_word_confidence_medium/hist_lowest_word_confidence_medium.png",
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
plt.title("Histogram of lowest-word confidence (medium, clean, test-set)")
plt.yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])

plt.savefig(OUTPUT_FILE)
