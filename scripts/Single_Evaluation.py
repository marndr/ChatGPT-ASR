"""
ASR Correction Evaluation Script for specific condition.

This script evaluates ASR transcriptions from the Librispeech dataset based on selected experiments using the ChatGPT.
It computes WER and CER for corrected and original transcriptions (for specific conditions), comparing them and generating
a markdown file with detailed evaluation results.

Usage:
    python Single_Evaluation.py [-d dataset] [-e experiment]

Example:
    python scripts/Single_File_Evaluation.py -d librispeech -e exp_without_sentence_confidence_GPT-3.5-Turbo_tiny

Options:
    -d, --dataset        Select the dataset (Default is 'librispeech').
    -e, --experiment     Select the experiment to evaluate.

Environment Variables:
    ROOT_PATH: Path to the root directory of the project containing data and results folders.

"""

import argparse
import json
import os

from dotenv import load_dotenv
from jiwer import RemovePunctuation, cer, wer
from tqdm import tqdm

EPSILON = 0.00001

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
            "exp_certain_low_confidence_words_Thresh_0.55_tiny",
            "exp_certain_low_confidence_words_Thresh_0.6_tiny",
            "exp_certain_low_confidence_words_Thresh_0.65_tiny",
            "exp_certain_low_confidence_words_Thresh_0.7_tiny",
            "exp_certain_low_confidence_words_Thresh_0.75_tiny",
            "exp_certain_low_confidence_words_Thresh_0.8_tiny",
            "exp_certain_low_confidence_words_Thresh_0.85_tiny",
            "exp_certain_low_confidence_words_Thresh_0.9_tiny",
            "exp_certain_low_confidence_words_Thresh_0.95_tiny",
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
            "exp_without_sentence_confidence_GPT-4-Turbo-Turbo_noisy_tiny",
            "exp_without_lowest_word_confidence_GPT-3.5-Turbo_noisy_tiny",
            "exp_without_lowest_word_confidence_GPT-4-Turbo_noisy_tiny",
            "exp_sentence_confidence_GPT-3.5-Turbo_noisy_medium",
            "exp_lowest_word_confidence_GPT-3.5-Turbo_noisy_medium",
            "exp_sentence_confidence_GPT-4-Turbo_noisy_medium",
            "exp_lowest_word_confidence_GPT-4-Turbo_noisy_medium",
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
            "results/results-dev-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_sentence_confidence_tiny/results_single_sentence_confidence_tiny.md",
        )

    elif args.experiment == "exp_without_sentence_confidence_GPT-4-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/results_single_sentence_confidence_GPT-4-Turbo_tiny.md",
        )

    elif args.experiment == "exp_without_lowest_word_confidence_GPT-3.5-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_lowest_word_confidence_tiny/corrected_transcriptions_lowest_word_confidence_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_lowest_word_confidence_tiny/results_single_lowest_word_confidence_tiny.md",
        )

    elif args.experiment == "exp_without_lowest_word_confidence_GPT-4-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/results_single_lowest_word_confidence_GPT-4-Turbo_tiny.md",
        )

    elif args.experiment == "exp_without_average_word_confidence_GPT-3.5-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_average_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_average_word_confidence_tiny/corrected_transcriptions_average_word_confidence_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_average_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_average_word_confidence_tiny/results_single_average_word_confidence_tiny.md",
        )

    elif args.experiment == "exp_without_average_word_confidence_GPT-4-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_average_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_without_average_word_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_average_word_confidence_GPT-4-Turbo_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_average_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_without_average_word_confidence_GPT-4-Turbo_tiny/results_single_average_word_confidence_GPT-4-Turbo_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.55_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.55_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_single_Thresh=0.55_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.6_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.6_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_single_Thresh=0.6_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.65_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.65_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_single_Thresh=0.65_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.7_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.7_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_single_Thresh=0.7_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.75_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.75_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_single_Thresh=0.75_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.8_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.8_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_single_Thresh=0.8_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.85_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.85_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/vresults-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_single_Thresh=0.85_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.9_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.9_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_single_Thresh=0.9_tiny.md",
        )

    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.95_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.95_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_tiny/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_single_Thresh=0.95_tiny.md",
        )

    elif (
        args.experiment
        == "exp_certain_low_confidence_words_Thresh_0.6_GPT-4-Turbo_tiny"
    ):
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/postprocessed_corrected_transcriptions_certain_low_confidence_words_GPT-4-Turbo_tiny/postprocessed_corrected_transcriptions_Thresh=0.6_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_certain_low_confidence_words_GPT-4-Turbo_tiny/results_certain_low_confidence_words_single_Thresh=0.6_GPT-4-Turbo_tiny.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-3.5-Turbo_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/corrected_transcriptions_sentence_confidence_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/results_single_sentence_confidence_medium.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-3.5_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_medium/corrected_transcriptions_lowest_word_confidence_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_medium/results_single_lowest_word_confidence_medium.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-4-Turbo_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/results_single_sentence_confidence_GPT-4-Turbo_medium.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-4-Turbo_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_medium/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_medium/results_single_lowest_word_confidence_medium.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-3.5-Turbo_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_sentence_confidence_large-v3/corrected_transcriptions_sentence_confidence_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_sentence_confidence_large-v3/results_single_sentence_confidence_large-v3.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-3.5-Turbo_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_large-v3/corrected_transcriptions_lowest_word_confidence_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_large-v3/results_single_lowest_word_confidence_large-v3.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-4-Turbo_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-1106-preview/results_without_sentence_confidence_GPT-4-Turbo_large-v3/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-1106-preview/results_without_sentence_confidence_GPT-4-Turbo_large-v3/results_single_sentence_confidence_GPT-4-Turbo_large-v3.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-4-Turbo_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_large-v3/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_large-v3/results_single_lowest_word_confidence_GPT-4-Turbo_large-v3.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-3.5-Turbo_noisy_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-1106/results_without_sentence_confidence_noisy_large-v3/corrected_transcriptions_sentence_confidence_noisy_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-1106/results_without_sentence_confidence_noisy_large-v3/results_single_sentence_confidence_noisy_large-v3.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-3.5-Turbo_noisy_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_noisy_large-v3/corrected_transcriptions_lowest_word_confidence_noisy_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_noisy_large-v3/results_single_lowest_word_confidence_noisy_large-v3.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-4-Turbo_noisy_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_noisy_large-v3/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_noisy_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_noisy_large-v3/results_single_sentence_confidence_noisy_large-v3.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-4-Turbo_noisy_large-v3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_noisy_large-v3/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_noisy_large-v3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_noisy_large-v3/results_single_lowest_word_confidence_noisy_large-v3.md",
        )

    elif args.experiment == "exp_without_sentence_confidence_GPT-3.5-Turbo_noisy_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/results_single_sentence_confidence_tiny.md",
        )

    elif args.experiment == "exp_without_sentence_confidence_GPT-4-Turbo_noisy_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/results_single_sentence_confidence_GPT-4-Turbo_tiny.md",
        )

    elif (
        args.experiment == "exp_without_lowest_word_confidence_GPT-3.5-Turbo_noisy_tiny"
    ):
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_tiny/corrected_transcriptions_lowest_word_confidence_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_tiny/results_single_lowest_word_confidence_tiny.md",
        )

    elif args.experiment == "exp_without_lowest_word_confidence_GPT-4-Turbo_noisy_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/results_single_lowest_word_confidence_GPT-4-Turbo_tiny.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-3.5-Turbo_noisy_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/corrected_transcriptions_sentence_confidence_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/results_single_sentence_confidence_medium.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-3.5-Turbo_noisy_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_medium/corrected_transcriptions_lowest_word_confidence_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_medium/results_single_lowest_word_confidence_medium.md",
        )

    elif args.experiment == "exp_sentence_confidence_GPT-4-Turbo_noisy_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/results_single_sentence_confidence_GPT-4-Turbo_medium.md",
        )

    elif args.experiment == "exp_lowest_word_confidence_GPT-4-Turbo_noisy_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_medium/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_medium.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_medium/results_lowest_word_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_medium/results_single_lowest_word_confidence_medium.md",
        )
    elif args.experiment == "exp_lowest_word_confidence_llama3:8b_tiny_noisy":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_Ollama-llama3:8b_tiny/without_present_confidence_ollama/ollama-llama3:8b/results_without_lowest_word_confidence_tiny/corrected_transcriptions_lowest_word_confidence_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_Ollama-llama3:8b_tiny/without_present_confidence_ollama/ollama-llama3:8b/results_without_lowest_word_confidence_tiny/results_single_lowest_word_confidence_llama3:8b.md",
        )

    if args.dataset == "librispeech":
        with open(CORRECTED_TRANSCRIPTIONS_LIBRISPEECH) as f:
            json_obj = f.read()
            data = json.loads(json_obj)
        output_filename = OUTPUT_FILE

    with open(output_filename, "w") as f:
        ref, hyp, hyp_original = [], [], []

        for d in tqdm(data):
            if d["corrected_asr_transcription"] is None:
                continue

            ref = RemovePunctuation()(d["reference_transcription"].lower())
            hyp = RemovePunctuation()(d["corrected_asr_transcription"].lower())
            hyp_original = RemovePunctuation()(d["asr_transcription"]["text"].lower())

            wer_corrected_chatgpt = wer([ref], [hyp]) * 100
            wer_original = wer([ref], [hyp_original]) * 100
            cer_corrected_chatgpt = cer([ref], [hyp]) * 100
            cer_original = cer([ref], [hyp_original]) * 100

            # np.sqrt(wer_original**2) <= EPSILON means zero
            # if ( wer_original <= EPSILON and wer_corrected_chatgpt > EPSILON):
            # f.write(
            # f"""
            # data with this condition ( wer_original <= EPSILON and wer_corrected_chatgpt > EPSILON)
            # ASR Transcription:        {hyp_original}
            # Reference Transcription:  {ref}
            # Corrected Transcription:  {hyp}

            # WER_original:             {wer_original:.04f}%
            # WER_corrected_chatgpt:    {wer_corrected_chatgpt:.04f}%

            # ---
            # """
            # )
            # if (wer_corrected_chatgpt < wer_original) and (
            # cer_corrected_chatgpt > cer_original
            # ):
            if wer_corrected_chatgpt > wer_original:
                f.write(
                    f"""
                    data with this condition (wer_corrected > wer_original)
                    ASR Transcription:       {hyp_original}
                    Reference Transcription:  {ref}
                    Corrected Transcription:  {hyp}

                    WER_corrected:    {wer_corrected_chatgpt:.04f}%
                    WER_original:     {wer_original:.04f}%
                    CER_corrected:    {cer_corrected_chatgpt:.04f}%
                    CER_original:     {cer_original:.04f}%

                    ---
                    """
                )

            # if (wer_corrected_chatgpt==0 and cer_corrected_chatgpt !=0) or (wer_original == 0 and cer_original !=0):

            # f.write(
            # f"""
            # Evaluation are wrong in this audio file ((wer_corrected_chatgpt==0 and cer_corrected_chatgpt !=0) or (wer_original == 0 and cer_original !=0)) !!!!

            # ASR Transcription:        {d["asr_transcription"]["text"]}
            # Reference Transcription:  {d["reference_transcription"]}
            # Corrected Transcription:  {d["corrected_asr_transcription"]}

            # WER_corrected_chatgpt:    {wer_corrected_chatgpt:.04f}%
            # WER_original:             {wer_original:.04f}%
            # CER_corrected_chatgpt:    {cer_corrected_chatgpt:.04f}%
            # CER_original:             {cer_original:.04f}%

            # ---
            # """
            # )
