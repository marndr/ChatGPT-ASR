# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maryam Naderi  <maryam.naderi@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-chatgpt-asr

"""
ASR Confidence Threshold Evaluation and Plotting Script

This script evaluates the impact of various confidence thresholds on
Word Error Rate (WER) and Character Error Rate (CER) for ASR transcriptions
corrected using GPT-4 Turbo model. It plots the results and saves
them to specified output files.


Environment Variables:
- ROOT_PATH: The root directory path for input and output files.

Example:
    python exp_find_thresh_sentence_confidence_GPT-4-Turbo_test_large-v3.py

Functions:
    - evaluate_with_thresh(items, thresh): Evaluates WER and CER at a given confidence threshold.
    - plot(results): Plots WER and CER against confidence thresholds.
"""

import json
import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from jiwer import RemovePunctuation, cer, wer

load_dotenv()
Root = os.getenv("ROOT_PATH")

l = os.path.join(
    Root,
    "results/results-test-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_large-v3/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_large-v3.json",
)

OUTPUT_FILE = os.path.join(
    Root,
    "results/results-test-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-0125-preview/results_find_thresh_sentence_confidence_GPT-4-Turbo_large-v3/results_thresh_sentence_confidence_GPT-4-Turbo_large-v3.md",
)

output_file_wer = os.path.join(
    Root,
    "results/results-test-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-0125-preview/results_find_thresh_sentence_confidence_GPT-4-Turbo_large-v3/plots_sentence_confidence_GPT-4-Turbo_large-v3/Wer_vs_sentence_confidence_GPT-4-Turbo_plot_large-v3.png",
)

output_file_cer = os.path.join(
    Root,
    "results/results-test-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-0125-preview/results_find_thresh_sentence_confidence_GPT-4-Turbo_large-v3/plots_sentence_confidence_GPT-4-Turbo_large-v3/Cer_vs_sentence_confidence_GPT-4-Turbo_plot_large-v3.png",
)

with open(l) as f:
    json_obj = f.read()
    data = json.loads(json_obj)


def evaluate_with_thresh(items, thresh):
    """
    Evaluates WER and CER for a given confidence threshold.

    Args:
        items (list): List of dictionaries containing ASR transcriptions and their confidence scores.
        thresh (float): The confidence threshold to evaluate.

    Returns:
        tuple: A tuple containing WER and CER as percentages.
    """
    hyp_l, ref_l = [], []

    for item in items:
        if item["corrected_asr_transcription"] is None:
            continue

        ref_transcription = RemovePunctuation()(item["reference_transcription"].lower())
        cor_transcription = RemovePunctuation()(
            item["corrected_asr_transcription"].lower()
        )
        asr_transcription = RemovePunctuation()(
            item["asr_transcription"]["text"].lower()
        )
        confidence = item["asr_transcription"]["confidence_score"]

        ref_l.append(ref_transcription)

        # check confidence and append the suitable value to hyp_l
        if confidence <= thresh:
            hyp_l.append(cor_transcription)

        else:
            hyp_l.append(asr_transcription)

    WER = wer(ref_l, hyp_l) * 100
    CER = cer(ref_l, hyp_l) * 100

    return WER, CER


thresholds = [0.95]
results = []


for thresh in thresholds:
    Wer, Cer = evaluate_with_thresh(data, thresh)
    results.append({"thresh": thresh, "wer": Wer, "cer": Cer})


def plot(l):
    """
    Plots WER and CER against confidence thresholds and saves the plots.

    Args:
        results (list): List of dictionaries containing thresholds, WER, and CER values.
    """
    x = [dictionary["thresh"] for dictionary in l]
    y_wer = [dictionary["wer"] for dictionary in l]

    plt.figure()
    plt.plot(x, y_wer, "-ob", label="Wer")
    plt.xlabel("sentence confidence")
    plt.ylabel("Wer")
    plt.title("Wer vs sentence confidence for GPT-4-Turbo(large-v3, clean, test-set)")
    # plt.yticks([3.5,4,4.5,5,5.5,6,6.5])
    # plt.show()
    plt.savefig(output_file_wer)

    y_cer = [dictionary["cer"] for dictionary in l]
    plt.figure()
    plt.plot(x, y_cer, "-or", label="Cer")
    plt.xlabel("sentence confidence")
    plt.ylabel("Cer")
    plt.title("Cer vs sentence confidence for GPT-4-Turbo(large-v3, clean, test-set)")
    # plt.yticks([1,1.5,2,2.5,3,3.5])
    # plt.show()
    plt.savefig(output_file_cer)


# plot(results)


# Sort results based on WER
results.sort(key=lambda x: x["wer"])

# Print results
for result in results:
    print(
        f'Threshold: {result["thresh"]:.02f}, {result["wer"]:.02f}, {result["cer"]:.02f}'
    )


# Write the JSON data to a file
with open(OUTPUT_FILE, "w") as f:
    json_obj = json.dumps(results, indent=2)
    f.write(json_obj)
