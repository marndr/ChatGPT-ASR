# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maryam Naderi  <maryam.naderi@idiap.ch>
# SPDX-FileContributor: Olivier Bornet  <olivier.bornet@idiap.ch>

#
# SPDX-License-Identifier: LicenseRef-chatgpt-asr

"""
This module provides functions to handle transcription correction using GPT model.
It includes error printing, fetching responses from OpenAI's chat completions API,
and parallel processing of multiple transcription data items using multithreading.
"""

import concurrent.futures
import os
import sys
import time
import traceback

from dotenv import load_dotenv
from openai import OpenAI


def print_error(error_code):
    """
    Prints error details based on the specified error code.
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if error_code <= 1:
        print(f"Exception type: {exc_type}")
    if error_code <= 2:
        print(f"Exception message: {exc_value}")
    if error_code <= 3:
        print(f"Exception traceback: {exc_traceback}")
    if error_code <= 4:
        traceback.print_tb(exc_traceback)


def get_chatgpt_response(client, d, get_messages_fn, model):
    """
    Retrieves a corrected transcription from OpenAI's chat completions API.

    Args:
        client (OpenAI): The OpenAI API client.
        d (dict): A dictionary containing ASR and reference transcriptions.
        get_messages_fn (function): A function to generate messages for the API call.
        model (str): The model name to use for the API call.

    Returns:
        dict: A dictionary containing the original ASR transcription,
              reference transcription, and corrected ASR transcription.
    """
    transcription_has_confidence = "confidence_score" in d["asr_transcription"]
    if transcription_has_confidence:
        cscore = d["asr_transcription"].pop("confidence_score")
    asr_transcription = d["asr_transcription"]
    reference_transcription = d["reference_transcription"]
    messages = get_messages_fn(asr_transcription)

    retries = 3  # Limiting retries to avoid infinite loops
    while retries > 0:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,  # request_timeout=60
            )
            corrected_asr_transcription = response.choices[0].message.content
            if transcription_has_confidence:
                asr_transcription["confidence_score"] = cscore
            return {
                "asr_transcription": asr_transcription,
                "reference_transcription": reference_transcription,
                "corrected_asr_transcription": corrected_asr_transcription,
            }

        # except openai.error.Timeout:  # Timeout, retry after a delay
        # print_error(1)
        # retries -= 1
        # time.sleep(10)  # Wait for a short period before retrying

        # except openai.error.RateLimitError:  # Quota exceeded, retry after a delay
        # print_error(1)
        # retries -= 1
        # time.sleep(60)

        except Exception:
            print_error(4)
            retries -= 1
            time.sleep(10)

    # If all retries fail, return None for this item
    if transcription_has_confidence:
        asr_transcription["confidence_score"] = cscore
    return {
        "asr_transcription": asr_transcription,
        "reference_transcription": reference_transcription,
        "corrected_asr_transcription": None,
    }


def multithread_parallelization(
    data, get_messages_fn, api="openai", model="gpt-3.5-turbo", num_workers=8
):
    """
    Processes a list of transcription data items in parallel using multithreading.

    Args:
        data (list): A list of dictionaries, each containing ASR and reference
        transcriptions.
        get_messages_fn (function): A function to generate messages for the API call.
        api (str): The API to use, default is "openai".
        model (str): The model name to use for the API call, default is "gpt-3.5-turbo".
        num_workers (int): The number of worker threads to use, default is 8.

    Returns:
        list: A list of dictionaries containing the original ASR transcriptions,
              reference transcriptions, and corrected ASR transcriptions.
    """
    outputs = []

    if api == "openai":
        # Load API key
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_Idiap"))
    else:
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="NOT-USED",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                get_chatgpt_response, client, item, get_messages_fn, model
            ): item
            for item in data
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                d = future.result()
                outputs.append(d)
            except Exception as exc:
                print_error(4)
                print(f"{item} generated an exception: {exc}")
    return outputs


# Example usage:
# result = multithread_parallelization(data, get_messages_fn)
# print(result)
