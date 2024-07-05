# Copyright 2023 Idiap Research Institute <contact@idiap.ch
#
# SPDX-FileContributor: Maryam Naderi <maryam.naderi@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-chatgpt-asr


"""
ASR Confidence Threshold Evaluation and Plotting Script

This script evaluates the impact of various confidence thresholds on
Word Error Rate (WER) and Character Error Rate (CER) for ASR transcriptions
corrected using GPT-3.5 Turbo model. It plots the results and saves
them to specified output files.


Environment Variables:
- ROOT_PATH: The root directory path for input and output files.

Example:
    python exp_find_thresh_best_prompt_sentence_confidence_medium.py

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
    "results/results-dev-set/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_medium/results_find_best_prompt_medium/corrected_transcriptions_sentence_confidence_prompt_1.json",
)

OUTPUT_FILE = os.path.join(
    Root,
    "results/results-dev-set/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_medium/results_find_thresh_best_prompt_medium/results_thresh_sentence_confidence_prompt_1_medium.md",
)

output_file_wer = os.path.join(
    Root,
    "results/results-dev-set/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_medium/results_find_thresh_best_prompt_medium/plots_sentence_confidence_medium/Wer_vs_sentence_confidence_prompt_1_medium.png",
)

output_file_cer = os.path.join(
    Root,
    "results/results-dev-set/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_medium/results_find_thresh_best_prompt_medium/plots_sentence_confidence_medium/Cer_vs_sentence_confidence_prompt_1_medium.png",
)

with open(l) as f:
    json_obj = f.read()
    data = json.loads(json_obj)


def evaluate_with_thresh(items, thresh):
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


thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
results = []


for thresh in thresholds:
    Wer, Cer = evaluate_with_thresh(data, thresh)
    results.append({"thresh": thresh, "wer": Wer, "cer": Cer})


def plot(l):
    x = [dictionary["thresh"] for dictionary in l]
    y_wer = [dictionary["wer"] for dictionary in l]

    plt.figure()
    plt.plot(x, y_wer, "-ob", label="Wer")
    plt.xlabel("sentence confidence")
    plt.ylabel("Wer")
    plt.title(
        "Wer vs sentence confidence for GPT-3.5-Turbo (new prompt-1, medium, clean)"
    )
    # plt.yticks([3,3.5,4,4.5,5,5.5])
    plt.savefig(output_file_wer)

    y_cer = [dictionary["cer"] for dictionary in l]
    plt.figure()
    plt.plot(x, y_cer, "-or", label="Cer")
    plt.xlabel("sentence confidence")
    plt.ylabel("Cer")
    plt.title(
        "Cer vs sentence confidence for GPT-3.5-Turbo (new promt-1, medium, clean)"
    )
    # plt.yticks([1,1.5,2,2.5])
    # plt.show()
    plt.savefig(output_file_cer)


plot(results)


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
