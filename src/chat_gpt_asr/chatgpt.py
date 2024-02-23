import openai
from tqdm import tqdm
import os
import sys
import argparse
from chat_gpt_asr.utils import read_dummy_transcriptions, confidence_score_sentence_level
from dotenv import load_dotenv
import json
import time
import concurrent.futures
import traceback

def print_error(error_code):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if error_code <= 1:
        print(f"Exception type: {exc_type}")
    if error_code <= 2:
        print(f"Exception message: {exc_value}")
    if error_code <= 3:
        print(f"Exception traceback: {exc_traceback}")
    if error_code <= 4:
        traceback.print_tb(exc_traceback)

def get_chatgpt_response(d, get_messages_fn, model):
    asr_transcription = d["asr_transcription"]
    reference_transcription = d["reference_transcription"]
    messages = get_messages_fn(asr_transcription)

    retries = 3  # Limiting retries to avoid infinite loops
    while retries > 0:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
                request_timeout=60
            )
            corrected_asr_transcription = response.choices[0].message["content"]
            return {
                "asr_transcription": asr_transcription,
                "reference_transcription": reference_transcription,
                "corrected_asr_transcription": corrected_asr_transcription
            }

        except openai.error.Timeout:  # Timeout, retry after a delay
            print_error(1)
            retries -= 1
            time.sleep(10)  # Wait for a short period before retrying

        except openai.error.RateLimitError:  # Quota exceeded, retry after a delay
            print_error(1)
            retries -= 1
            time.sleep(60)

        except Exception as e:
            print_error(4)
            retries -= 1
            time.sleep(10)

    # If all retries fail, return None for this item
    return {
        "asr_transcription": asr_transcription,
        "reference_transcription": reference_transcription,
        "corrected_asr_transcription": None
    }

def multithread_parallelization(data, get_messages_fn, model="gpt-3.5-turbo", num_workers=8):
    l = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(get_chatgpt_response, item, get_messages_fn, model): item for item in data}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                d = future.result()
                l.append(d)
            except Exception as exc:
                print_error(4)
                print(f'{item} generated an exception: {exc}')
    return l

# Example usage:
# result = multithread_parallelization(data, get_messages_fn)
# print(result)

