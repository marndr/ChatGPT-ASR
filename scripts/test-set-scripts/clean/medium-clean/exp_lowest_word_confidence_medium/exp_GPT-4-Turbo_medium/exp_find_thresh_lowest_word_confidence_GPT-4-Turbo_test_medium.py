"""
ASR Confidence Threshold Evaluation and Plotting Script

This script evaluates the impact of various confidence thresholds on
Word Error Rate (WER) and Character Error Rate (CER) for ASR transcriptions
corrected using GPT-4 Turbo model. It plots the results and saves
them to specified output files.


Environment Variables:
- ROOT_PATH: The root directory path for input and output files.

Example:
    python exp_find_thresh_lowest_word_confidence_GPT-4-Turbo_test_medium.py

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
    "results/results-test-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_medium/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_medium.json",
)

OUTPUT_FILE = os.path.join(
    Root,
    "results/results-test-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_find_thresh_lowest_word_confidence_GPT-4-Turbo_medium/results_thresh_lowest_word_confidence_medium.json",
)

output_file_wer = os.path.join(
    Root,
    "results/results-test-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_find_thresh_lowest_word_confidence_GPT-4-Turbo_medium/plots_lowest_word_confidence_GPT-4-Turbo_medium/Wer_vs_lowest_word_confidence_GPT-4-Turbo_plot_medium.png",
)

output_file_cer = os.path.join(
    Root,
    "results/results-test-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_find_thresh_lowest_word_confidence_GPT-4-Turbo_medium/plots_lowest_word_confidence_GPT-4-Turbo_medium/Cer_vs_lowest_word_confidence_GPT-4-Turbo_plot_medium.png",
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


thresholds = [0.7]
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
    plt.xlabel("lowest word confidence")
    plt.ylabel("Wer")
    plt.title("Wer vs lowest word confidence for GPT-4-Turbo (clean, medium, test-set)")
    # plt.yticks([3,3.5,4,4.5,5,5.5])
    # plt.show()
    plt.savefig(output_file_wer)

    y_cer = [dictionary["cer"] for dictionary in l]
    plt.figure()
    plt.plot(x, y_cer, "-or", label="Cer")
    plt.xlabel("lowest word confidence")
    plt.ylabel("Cer")
    plt.title("Cer vs lowest word confidence for GPT-4-Turbo (clean, medium, test-set)")
    # plt.yticks([1,1.5,2,2.5])
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
