"""
ASR Correction Script Using ChatGPT

This script utilizes GPT-4 Turbo model to correct automatic speech recognition (ASR) transcriptions.
It reads ASR transcriptions from a dataset, processes them to calculate confidence scores, and corrects the
transcriptions using the GPT-4 Turbo model. The corrected transcriptions are then saved to a specified output file.

Environment Variables:
- ROOT_PATH: The root directory path for input and output files.
- OPENAI_API_KEY: The API key for OpenAI.

Usage:
    Run the script from the command line with optional arguments to specify the dataset and number of data points to process.
    -d, --dataset: Specify the dataset to use ('librispeech'). Default is 'librispeech'.
    -n, --num_data: Specify the number of data points to process. Default is -1 (process all data).

Example:
    python exp_without_lowest_word_confidence_GPT-4-Turbo_noisy_tiny.py -d librispeech -n 1

Functions:
    get_messages_exp1(asr_transcription): Constructs the message list for GPT-4 Turbo to correct ASR transcriptions.

"""

import argparse
import json
import os

import openai
from chat_gpt_asr.chatgpt import multithread_parallelization
from chat_gpt_asr.utils import confidence_score_lowest_word_level
from dotenv import load_dotenv

load_dotenv()
Root = os.getenv("ROOT_PATH")


TRANSCRIPTION_FILENAME = os.path.join(
    Root, "data/transcriptions/whisper_tiny_librispeech_dev-other-full.json"
)
CORRECTED_TRANSCRIPTION_FILENAME = os.path.join(
    Root,
    "results/results-dev-set/results_noisy/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_tiny.json",
)


def get_messages_exp1(asr_transcription):
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that corrects ASR errors. \
            You will be presented with an ASR transcription of Librispeech data provided by the Whisper model. \
            Your task is to correct any errors in the transcription.\
            Provide the most probable corrected transcription in string format. \
            If you come across errors in ASR transcription, make corrections that closely match the original transcription acoustically or phonetically.\
            Do not change the case, for example, lower case or upper case, in the transcription. \
            Do not output any additional text that is not the corrected transcription. \
            Do not write any explanatory text that is not the corrected transcription.
            """,
        },
        {
            "role": "user",
            "content": '{"text": "Why not allow your silver tuff to luxuriate in a natural manner?"}',
        },
        {
            "role": "assistant",
            "content": "why not allow your silver tufts to luxuriate in a natural manner?",
        },
        {
            "role": "user",
            "content": '{"text": "Meanwhile, how fair did it with the flowers?"}',
        },
        {
            "role": "assistant",
            "content": "Meanwhile, how fared did it with the flowers?",
        },
        {"role": "user", "content": json.dumps(asr_transcription)},
    ]
    return messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT ASR Correction")
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["librispeech", "dummy"],
        default="librispeech",
        help="Select the dataset (librispeech or dummy)",
    )
    parser.add_argument(
        "-n", "--num_data", type=int, default=-1, help="Select the number of data"
    )
    args = parser.parse_args()

    # Load API key
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY_Idiap")
    # openai.api_key = os.getenv("OPENAI_API_KEY_MARYAM")

    if args.dataset == "librispeech":
        transcription_file = TRANSCRIPTION_FILENAME
        output_file = CORRECTED_TRANSCRIPTION_FILENAME
        with open(transcription_file) as f:
            json_obj = f.read()
            data = json.loads(json_obj)

        if args.num_data > 0:
            data = data[: args.num_data]

        for i, d in enumerate(data):
            asr_transcription = confidence_score_lowest_word_level(
                d["asr_transcription"], confidence=True
            )
            reference_transcription = d["reference_transcription"]
            data[i] = {
                "asr_transcription": asr_transcription,
                "reference_transcription": reference_transcription,
            }

    l = multithread_parallelization(
        data, get_messages_fn=get_messages_exp1, model="gpt-4-0125-preview"
    )

    with open(output_file, "w") as f:
        json_str = json.dumps(l, indent=2)
        f.write(json_str)
