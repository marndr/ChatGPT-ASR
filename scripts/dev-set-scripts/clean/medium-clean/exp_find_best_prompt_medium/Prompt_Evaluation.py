# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maryam Naderi <maryam.naderi@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-chatgpt-asr

"""
Script to evaluate ASR correction performance for the Librispeech dataset based on selected experiments.

This script computes WER, CER, and SER metrics for both original and corrected ASR transcriptions.
It reads JSON data files containing original, reference and corrected transcriptions,
and outputs evaluation results to the console and a markdown file.

Usage:
    python Prompt_Evaluation.py [-d DATASET] [-e EXPERIMENT]

Options:
    -d, --dataset        Select the dataset (Default is 'librispeech').
    -e, --experiment     Select the experiment to evaluate.

Example:
    python Prompt_Evaluation.py -d librispeech -e exp_new_prompt_sentence_confidence_1_medium

Environment Variables:
    ROOT_PATH: Path to the root directory of the project containing data and results folders.
"""

import argparse
import json
import os

# from chat_gpt_asr.utils import remove_punctuations, ser
from chat_gpt_asr.utils import ser
from dotenv import load_dotenv
from jiwer import RemovePunctuation, cer, compute_measures, wer
from tqdm import tqdm

load_dotenv()
Root = os.getenv("ROOT_PATH")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT ASR Correction")
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["librispeech"],
        default="librispeech",
        help="Select the dataset (librispeech)",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        choices=[
            "exp_new_prompt_sentence_confidence_1_medium",
            "exp_new_prompt_lowest_word_confidence_1_medium",
            "exp_new_prompt_sentence_confidence_1_GPT-4-Turbo_medium",
        ],
    )
    args = parser.parse_args()

    if args.experiment == "exp_new_prompt_sentence_confidence_1_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_medium/results_find_best_prompt_medium/corrected_transcriptions_sentence_confidence_prompt_1.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_medium/results_find_best_prompt_medium/results_overall_sentence_confidence_prompt_1_medium.md",
        )

    if args.experiment == "exp_new_prompt_lowest_word_confidence_1_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_lowest_word_confidence_new_prompts_medium/results_find_best_prompt_medium/corrected_transcriptions_lowest_word_confidence_prompt_1.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_lowest_word_confidence_new_prompts_medium/results_find_best_prompt_medium/results_overall_lowest_word_confidence_prompt_1_medium.md",
        )

    if args.experiment == "exp_new_prompt_sentence_confidence_1_GPT-4-Turbo_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_best_prompt_medium/results_GPT-4-Turbo_medium/gpt-4-1106-preview/results_sentence_confidence_new_prompts_medium/results_find_best_prompt_medium/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_prompt_1.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_best_prompt_medium/results_GPT-4-Turbo_medium/gpt-4-1106-preview/results_sentence_confidence_new_prompts_medium/results_find_best_prompt_medium/results_overall_sentence_confidence_prompt_1_medium.md",
        )

    if args.dataset == "librispeech":
        with open(CORRECTED_TRANSCRIPTIONS_LIBRISPEECH) as f:
            json_obj = f.read()
            data = json.loads(json_obj)
        output_filename = OUTPUT_FILE

    ref_l, hyp_l = [], []
    ser_corrected_chatgpt, ser_original = 0.0, 0.0
    count = 0

    hyp_l_original = []
    ref_l_orig = []

    for d in tqdm(data):
        if d["corrected_asr_transcription"] is None:
            ref_l_orig.append(RemovePunctuation()(d["reference_transcription"].lower()))
            hyp_l_original.append(
                RemovePunctuation()(d["asr_transcription"]["text"].lower())
            )

        else:
            ref = RemovePunctuation()(d["reference_transcription"].lower())
            hyp = RemovePunctuation()(d["corrected_asr_transcription"].lower())
            hyp_original = RemovePunctuation()(d["asr_transcription"]["text"].lower())
            ref_l.append(ref)
            hyp_l.append(hyp)
            hyp_l_original.append(hyp_original)
            ref_l_orig.append(ref)
            SER_chatgpt_, SER_original_ = ser(hyp_original, hyp, ref)
            count += 1
            ser_corrected_chatgpt += SER_chatgpt_
            ser_original += SER_original_

    wer_original = wer(ref_l_orig, hyp_l_original) * 100
    wer_corrected_chatgpt = wer(ref_l, hyp_l) * 100

    cer_corrected_chatgpt = cer(ref_l, hyp_l) * 100
    cer_original = cer(ref_l_orig, hyp_l_original) * 100

    measures = compute_measures(ref_l, hyp_l)
    measures_original = compute_measures(ref_l_orig, hyp_l_original)

    ser_corrected_chatgpt /= count
    ser_original /= count

    print(f"Number of audio files:  {count:}\n")
    print(f"WER_original is:                {wer_original:.04f}\n")
    print(f"WER_corrected_chatgpt is:       {wer_corrected_chatgpt:.04f}\n")
    print(f"CER_original is:                {cer_original:.04f}\n")
    print(f"CER_corrected_chatgpt is:       {cer_corrected_chatgpt:.04f}\n")
    print(f"SER_original is:                {ser_original:.04f}\n")
    print(f"SER_corrected_chatgpt is:       {ser_corrected_chatgpt:.04f}\n")
    print(
        f"measures for substitution: {measures['substitutions']}, insertions: {measures['insertions']}, deletions: {measures['deletions']}, wer: {measures['wer']}"
    )
    print(
        f"measures_original for substitution: {measures_original['substitutions']}, insertions: {measures_original['insertions']}, deletions: {measures_original['deletions']}, wer: {measures_original['wer']}"
    )

    with open(output_filename, "w") as f:
        f.write(f"""
        Number of evaluated audio files:  {count:}
        WER_original:                   {wer_original:.04f}%
        WER_corrected_chatgpt:          {wer_corrected_chatgpt:.04f}%
        CER_original:                   {cer_original:.04f}%
        CER_corrected_chatgpt:          {cer_corrected_chatgpt:.04f}%
        SER_original:                   {ser_original:.04f}%
        SER_corrected_chatgpt:          {ser_corrected_chatgpt:.04f}%
        measures for substitution: {measures['substitutions']}, insertions: {measures['insertions']}, deletions: {measures['deletions']}
        measures_original for substitution: {measures_original['substitutions']}, insertions: {measures_original['insertions']}, deletions: {measures_original['deletions']}
        ---
    """)
