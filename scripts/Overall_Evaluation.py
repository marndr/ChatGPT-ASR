"""
Script to evaluate ASR correction performance for the Librispeech dataset based on selected experiments.

This script computes WER, CER, and SER metrics for both original and corrected ASR transcriptions.
It reads JSON data files containing original, reference and corrected transcriptions,
and outputs evaluation results to the console and a markdown file.

Usage:
    python evaluate_asr_correction.py [-d DATASET] [-e EXPERIMENT]

Options:
    -d, --dataset        Select the dataset (Default is 'librispeech').
    -e, --experiment     Select the experiment to evaluate.


Environment Variables:
    ROOT_PATH: Path to the root directory of the project containing data and results folders.
"""

import argparse
import json
import os

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
            "exp_without_sentence_confidence_GPT-3.5-Turbo_tiny",
            "exp_without_lowest_word_confidence_GPT-3.5-Turbo_tiny",
            "exp_without_average_word_confidence_GPT-3.5-Turbo_tiny",
            "exp_without_sentence_confidence_GPT-4-Turbo_tiny",
            "exp_without_lowest_word_confidence_GPT-4-Turbo_tiny",
            "exp_certain_low_confidence_words_Thresh_0.1_tiny",
            "exp_certain_low_confidence_words_Thresh_0.2_tiny",
            "exp_certain_low_confidence_words_Thresh_0.3_tiny",
            "exp_certain_low_confidence_words_Thresh_0.4_tiny",
            "exp_certain_low_confidence_words_Thresh_0.5_tiny",
            "exp_certain_low_confidence_words_Thresh_0.6_tiny",
            "exp_certain_low_confidence_words_Thresh_0.7_tiny",
            "exp_certain_low_confidence_words_Thresh_0.8_tiny",
            "exp_certain_low_confidence_words_Thresh_0.9_tiny",
            "exp_certain_low_confidence_words_Thresh_1_tiny",
            "exp_certain_low_confidence_words_Thresh_0.6_GPT-4-Turbo_tiny",
            "exp_without_average_word_confidence_GPT-4-Turbo_tiny",
            "exp_sentence_confidence_GPT-3.5-Turbo_medium",
            "exp_lowest_word_confidence_GPT-3.5-Turbo_medium",
            "exp_sentence_confidence_GPT-4-Turbo_medium",
            "exp_sentence_confidence_GPT-3.5-Turbo_large-v3",
            "exp_lowest_word_confidence_GPT-3.5-Turbo_large-v3",
            "exp_sentence_confidence_GPT-4-Turbo_large-v3",
            "exp_sentence_confidence_GPT-3.5-Turbo_noisy_large-v3",
            "exp_lowest_word_confidence_GPT-3.5-Turbo_noisy_large-v3",
            "exp_sentence_confidence_GPT-4-Turbo_noisy_large-v3",
            "exp_lowest_word_confidence_GPT-4-Turbo_noisy_large-v3",
            "exp_lowest_word_confidence_GPT-4-Turbo_medium",
            "exp_lowest_word_confidence_GPT-4-Turbo_large-v3",
            "exp_without_sentence_confidence_GPT-3.5-Turbo_noisy_tiny",
            "exp_without_sentence_confidence_GPT-4-Turbo_noisy_tiny",
            "exp_without_lowest_word_confidence_GPT-3.5-Turbo_noisy_tiny",
            "exp_without_lowest_word_confidence_GPT-4-Turbo_noisy_tiny",
            "exp_sentence_confidence_GPT-3.5-Turbo_noisy_medium",
            "exp_lowest_word_confidence_GPT-3.5-Turbo_noisy_medium",
            "exp_sentence_confidence_GPT-4-Turbo_noisy_medium",
            "exp_lowest_word_confidence_GPT-4-Turbo_noisy_medium",
            "exp_without_sentence_confidence_GPT-3.5-Turbo_tiny_test",
            "exp_without_sentence_confidence_GPT-4-Turbo_tiny_test",
            "exp_without_lowest_word_confidence_GPT-3.5-Turbo_tiny_test",
            "exp_without_lowest_word_confidence_GPT-4-Turbo_tiny_test",
            "exp_sentence_confidence_GPT-3.5-Turbo_medium_test",
            "exp_lowest_word_confidence_GPT-3.5-Turbo_medium_test",
            "exp_sentence_confidence_GPT-3.5-Turbo_large-v3_test",
            "exp_sentence_confidence_GPT-4-Turbo_medium_test",
            "exp_lowest_word_confidence_GPT-4-Turbo_medium_test",
            "exp_lowest_word_confidence_GPT-3.5-Turbo_large-v3_test",
            "exp_sentence_confidence_GPT-3.5-Turbo_large-v3_test",
            "exp_sentence_confidence_GPT-4-Turbo_large-v3_test",
            "exp_lowest_word_confidence_GPT-4-Turbo_large-v3_test",
            "exp_without_sentence_confidence_GPT-3.5-Turbo_tiny_noisy_test",
            "exp_without_sentence_confidence_GPT-4-Turbo_tiny_noisy_test",
            "exp_without_lowest_word_confidence_GPT-3.5-Turbo_tiny_noisy_test",
            "exp_without_lowest_word_confidence_GPT-4_tiny_noisy_test",
            "exp_sentence_confidence_GPT-3.5-Turbo_medium_noisy_test",
            "exp_lowest_word_confidence_GPT-3.5-Turbo_medium_noisy_test",
            "exp_sentence_confidence_GPT-4-Turbo_medium_noisy_test",
            "exp_sentence_confidence_GPT-4-Turbo_large-v3_noisy_test",
            "exp_lowest_word_confidence_GPT-4-Turbo_large-v3_noisy_test",
            "exp_lowest_word_confidence_GPT-4-Turbo_medium_noisy_test",
            "exp_sentence_confidence_GPT-3.5-Turbo_large-v3_noisy_test",
            "exp_lowest_word_confidence_GPT-3.5-Turbo_large-v3_noisy_test",
            "exp_lowest_word_confidence_llama3:8b_tiny_noisy",
        ],
        help="Select the experiment",
    )
    args = parser.parse_args()

    if args.experiment == "exp_without_sentence_confidence_GPT-3.5-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_sentence_confidence_tiny/results_overall_sentence_confidence_tiny.md",
        )

    elif args.experiment == "exp_without_sentence_confidence_GPT-4-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/results_overall_sentence_confidence_GPT-4-Turbo_tiny.md",
        )

    elif args.experiment == "exp_without_lowest_word_confidence_GPT-3.5-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_tiny/corrected_transcriptions_lowest_word_confidence_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_tiny/results_overall_lowest_word_confidence_tiny.md",
        )

    elif args.experiment == "exp_without_lowest_word_confidence_GPT-4-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/results_overall_lowest_word_confidence_GPT-4-Turbo_tiny.md",
        )

    elif args.experiment == "exp_without_average_word_confidence_GPT-3.5-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_average_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_average_word_confidence_tiny/corrected_transcriptions_average_word_confidence_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_average_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_average_word_confidence_tiny/results_overall_average_word_confidence_tiny.md",
        )

    elif args.experiment == "exp_without_average_word_confidence_GPT-4-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_average_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_without_average_word_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_average_word_confidence_GPT-4-Turbo_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_average_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_without_average_word_confidence_GPT-4-Turbo_tiny/results_overall_average_word_confidence_GPT-4-Turbo_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.1_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny_new/postprocessed_corrected_transcriptions_processed_Thresh=0.1_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.1_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.2_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny_new/postprocessed_corrected_transcriptions_processed_Thresh=0.2_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.2_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.3_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny_new/postprocessed_corrected_transcriptions_processed_Thresh=0.3_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.3_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.4_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny_new/postprocessed_corrected_transcriptions_processed_Thresh=0.4_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.4_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.5_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny_new/postprocessed_corrected_transcriptions_processed_Thresh=0.5_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.5_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.6_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny_new/postprocessed_corrected_transcriptions_processed_Thresh=0.6_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.6_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.7_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny_new/postprocessed_corrected_transcriptions_processed_Thresh=0.7_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.7_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.8_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny_new/postprocessed_corrected_transcriptions_processed_Thresh=0.8_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.8_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.9_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny_new/postprocessed_corrected_transcriptions_processed_Thresh=0.9_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.9_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_1_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny_new/postprocessed_corrected_transcriptions_processed_Thresh=1_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=1_tiny.md",
        )
    # elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.6_GPT-4-Turbo_tiny":
    # CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/postprocessed_corrected_transcriptions_certain_low_confidence_words_GPT-4-Turbo_tiny/postprocessed_corrected_transcriptions_Thresh=0.6_tiny.json")

    # OUTPUT_FILE = os.path.join(Root,"results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_certain_low_confidence_words_GPT-4-Turbo_tiny/results_certain_low_confidence_words_Thresh=0.6_GPT-4-Turbo_tiny.md")

    elif args.experiment == "exp_sentence_confidence_GPT-3.5-Turbo_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/corrected_transcriptions_sentence_confidence_medium.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/results_overall_sentence_confidence_medium.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-3.5-Turbo_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_medium/corrected_transcriptions_lowest_word_confidence_medium.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_medium/results_overall_lowest_word_confidence_medium.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-4-Turbo_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_medium.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/results_overall_sentence_confidence_GPT-4-Turbo_medium.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-4-Turbo_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_medium/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_medium.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_medium/results_overall_lowest_word_confidence_medium.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-3.5-Turbo_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_sentence_confidence_large-v3/corrected_transcriptions_sentence_confidence_large-v3.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_sentence_confidence_large-v3/results_overall_sentence_confidence_large-v3.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-3.5-Turbo_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/present_confidence_chatgpt/results_without_lowest_word_confidence_large-v3/corrected_transcriptions_lowest_word_confidence_large-v3.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/present_confidence_chatgpt/results_without_lowest_word_confidence_large-v3/results_overall_lowest_word_confidence_large-v3.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-4-Turbo_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-1106-preview/results_without_sentence_confidence_GPT-4-Turbo_large-v3/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_large-v3.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-1106-preview/results_without_sentence_confidence_GPT-4-Turbo_large-v3/results_overall_sentence_confidence_GPT-4-Turbo_large-v3.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-4-Turbo_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-4o_large-v3/without_present_confidence_chatgpt/results_without_lowest_word_confidence_large-v3/corrected_transcriptions_lowest_word_confidence_large-v3.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-4o_large-v3/without_present_confidence_chatgpt/results_without_lowest_word_confidence_large-v3/results_overall_lowest_word_confidence_GPT-4o-Turbo_large-v3.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-3.5-Turbo_noisy_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-1106/results_without_sentence_confidence_noisy_large-v3/corrected_transcriptions_sentence_confidence_noisy_large-v3.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-1106/results_without_sentence_confidence_noisy_large-v3/results_overall_sentence_confidence_noisy_large-v3.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-3.5-Turbo_noisy_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_noisy_large-v3/corrected_transcriptions_lowest_word_confidence_noisy_large-v3.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_noisy_large-v3/results_overall_lowest_word_confidence_noisy_large-v3.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-4-Turbo_noisy_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_noisy_large-v3/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_noisy_large-v3.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_noisy_large-v3/results_overall_sentence_confidence_noisy_large-v3.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-4-Turbo_noisy_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_noisy_large-v3/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_noisy_large-v3.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_noisy_large-v3/results_overall_lowest_word_confidence_noisy_large-v3.md",
        )

    elif args.experiment == "exp_without_sentence_confidence_GPT-3.5-Turbo_noisy_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/results_overall_sentence_confidence_tiny.md",
        )

    elif args.experiment == "exp_without_sentence_confidence_GPT-4-Turbo_noisy_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/results_overall_sentence_confidence_GPT-4-Turbo_tiny.md",
        )

    elif (
        args.experiment == "exp_without_lowest_word_confidence_GPT-3.5-Turbo_noisy_tiny"
    ):
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/present_confidence_chatgpt/results_without_lowest_word_confidence_tiny/corrected_transcriptions_lowest_word_confidence_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/present_confidence_chatgpt/results_without_lowest_word_confidence_tiny/results_overall_lowest_word_confidence_tiny.md",
        )

    elif args.experiment == "exp_without_lowest_word_confidence_GPT-4-Turbo_noisy_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_tiny.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/results_overall_lowest_word_confidence_GPT-4-Turbo_tiny.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-3.5-Turbo_noisy_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/corrected_transcriptions_sentence_confidence_medium.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/results_overall_sentence_confidence_medium.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-3.5-Turbo_noisy_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_without_lowest_word_confidence_medium/corrected_transcriptions_lowest_word_confidence_medium.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_without_lowest_word_confidence_medium/results_overall_lowest_word_confidence_medium.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-4-Turbo_noisy_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_medium.json",
        )

        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/results_overall_sentence_confidence_GPT-4-Turbo_medium.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-4-Turbo_noisy_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_medium/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_medium/results_overall_lowest_word_confidence_medium.md",
        )

    if args.experiment == "exp_without_sentence_confidence_GPT-3.5-Turbo_tiny_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/results_overall_sentence_confidence_tiny.md",
        )

    elif args.experiment == "exp_without_sentence_confidence_GPT-4-Turbo_tiny_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/results_overall_sentence_confidence_GPT-4-Turbo_tiny.md",
        )

    elif (
        args.experiment == "exp_without_lowest_word_confidence_GPT-3.5-Turbo_tiny_test"
    ):
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_tiny/corrected_transcriptions_lowest_word_confidence_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_tiny/results_overall_lowest_word_confidence_tiny.md",
        )

    elif args.experiment == "exp_without_lowest_word_confidence_GPT-4-Turbo_tiny_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/results_overall_lowest_word_confidence_GPT-4-Turbo_tiny.md",
        )
    elif args.experiment == "exp_sentence_confidence_GPT-3.5-Turbo_medium_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/corrected_transcriptions_sentence_confidence_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/results_overall_sentence_confidence_medium.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-3.5-Turbo_medium_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_medium/corrected_transcriptions_lowest_word_confidence_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_medium/results_overall_lowest_word_confidence_medium.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-4-Turbo_medium_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/results_overall_sentence_confidence_GPT-4-Turbo_medium.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-4-Turbo_medium_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_medium/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_medium/results_overall_lowest_word_confidence_medium.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-3.5-Turbo_large-v3_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_sentence_confidence_large-v3/corrected_transcriptions_sentence_confidence_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_sentence_confidence_large-v3/results_overall_sentence_confidence_large-v3.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-3.5-Turbo_large-v3_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_large-v3/corrected_transcriptions_lowest_word_confidence_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_large-v3/results_overall_lowest_word_confidence_large-v3.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-4-Turbo_large-v3_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_large-v3/corrected_transcriptions_sentence_confidence_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_large-v3/results_overall_sentence_confidence_GPT-4-Turbo_large-v3.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-4-Turbo_large-v3_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_large-v3/corrected_transcriptions_lowest_word_confidence_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_large-v3/results_overall_lowest_word_confidence_GPT-4-Turbo_large-v3.md",
        )

    elif (
        args.experiment
        == "exp_without_sentence_confidence_GPT-3.5-Turbo_tiny_noisy_test"
    ):
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/results_overall_sentence_confidence_tiny.md",
        )

    elif (
        args.experiment == "exp_without_sentence_confidence_GPT-4-Turbo_tiny_noisy_test"
    ):
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/results_overall_sentence_confidence_GPT-4-Turbo_tiny.md",
        )

    elif (
        args.experiment
        == "exp_without_lowest_word_confidence_GPT-3.5-Turbo_tiny_noisy_test"
    ):
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_tiny/corrected_transcriptions_lowest_word_confidence_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_tiny/results_overall_lowest_word_confidence_tiny.md",
        )

    elif (
        args.experiment
        == "exp_without_lowest_word_confidence_GPT-4-Turbo_tiny_noisy_test"
    ):
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/results_overall_lowest_word_confidence_GPT-4-Turbo_tiny.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-3.5-Turbo_medium_noisy_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/corrected_transcriptions_sentence_confidence_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/results_overall_sentence_confidence_medium.md",
        )

    elif (
        args.experiment == "exp_lowest_word_confidence_GPT-3.5-Turbo_medium_noisy_test"
    ):
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_medium/corrected_transcriptions_lowest_word_confidence_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_medium/results_overall_lowest_word_confidence_medium.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-4-Turbo_medium_noisy_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/results_overall_sentence_confidence_GPT-4-Turbo_medium.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-4-Turbo_medium_noisy_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_medium/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_medium/results_overall_lowest_word_confidence_medium.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-3.5-Turbo_large-v3_noisy_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-0125/results_without_sentence_confidence_noisy_large-v3/corrected_transcriptions_sentence_confidence_noisy_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-0125/results_without_sentence_confidence_noisy_large-v3/results_overall_sentence_confidence_large-v3.md",
        )

    elif (
        args.experiment
        == "exp_lowest_word_confidence_GPT-3.5-Turbo_large-v3_noisy_test"
    ):
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_noisy_large-v3/corrected_transcriptions_lowest_word_confidence_noisy_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_noisy_large-v3/results_overall_lowest_word_confidence_large-v3.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-4-Turbo_large-v3_noisy_test":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_noisy_large-v3/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_noisy_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_noisy_large-v3/results_overall_sentence_confidence_GPT-4-Turbo_large-v3.md",
        )

    elif (
        args.experiment == "exp_lowest_word_confidence_GPT-4-Turbo_large-v3_noisy_test"
    ):
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_noisy_large-v3/corrected_transcriptions_lowest_word_confidence_noisy_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-test-set/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_noisy_large-v3/results_overall_lowest_word_confidence_GPT-4-Turbo_large-v3.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_llama3:8b_tiny_noisy":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_Ollama-llama3:8b_tiny/without_present_confidence_ollama/ollama-llama3:8b/results_without_lowest_word_confidence_tiny/corrected_transcriptions_lowest_word_confidence_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_Ollama-llama3:8b_tiny/without_present_confidence_ollama/ollama-llama3:8b/results_without_lowest_word_confidence_tiny/results_overall_lowest_word_confidence_llama3:8b.md",
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
    print(f"WER_original is:        {wer_original:.04f}\n")
    print(f"WER_corrected is:       {wer_corrected_chatgpt:.04f}\n")
    print(f"CER_original is:        {cer_original:.04f}\n")
    print(f"CER_corrected is:       {cer_corrected_chatgpt:.04f}\n")
    print(f"SER_original is:        {ser_original:.04f}\n")
    print(f"SER_corrected is:       {ser_corrected_chatgpt:.04f}\n")
    print(
        f"measures for substitution: {measures['substitutions']}, insertions: {measures['insertions']}, deletions: {measures['deletions']}, wer: {measures['wer']}"
    )
    print(
        f"measures_original for substitution: {measures_original['substitutions']}, insertions: {measures_original['insertions']}, deletions: {measures_original['deletions']}, wer: {measures_original['wer']}"
    )

    with open(output_filename, "w") as f:
        f.write(f"""
        Number of evaluated audio files:  {count:}
        WER_original:           {wer_original:.04f}%
        WER_corrected:          {wer_corrected_chatgpt:.04f}%
        CER_original:           {cer_original:.04f}%
        CER_corrected:          {cer_corrected_chatgpt:.04f}%
        SER_original:           {ser_original:.04f}%
        SER_corrected:          {ser_corrected_chatgpt:.04f}%
        measures for substitution: {measures['substitutions']}, insertions: {measures['insertions']}, deletions: {measures['deletions']}
        measures_original for substitution: {measures_original['substitutions']}, insertions: {measures_original['insertions']}, deletions: {measures_original['deletions']}

        ---
    """)
