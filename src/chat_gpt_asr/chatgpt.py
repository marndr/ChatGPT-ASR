import openai
from tqdm import tqdm
import os
import sys
import argparse
from chat_gpt_asr.utils import read_dummy_transcriptions, confidence_score_sentence_level
from dotenv import load_dotenv
import json
import time

#import multiprocessing
import concurrent.futures
#import asyncio
import traceback


# experiment 2
def get_messages(asr_transcription, delimiter="####"):
    messages = [
        {
            'role': 'system',
            'content': f"""You are a helpful assistant that corrects ASR errors. \
            You will be provided with the ASR output in JSON format with keys: text and confidence_score. \
            The text is the ASR transcription for an audio and confidence_score is its level of confidence. \
            If the confidence_score is low (lower than 0.7), it's very likely that the ASR system made an error in the transcription. \
            Therefore, your task is to replace low-confidence text in the ASR transcription with better transcription that make sense in the context. \
            Provide the most the corrected transcription in string format. \
            Do not change the case, for example, lower case or upper case, in the transcription. \
            Do not output any additional text that is not the corrected transcription. \
            Do not write any explanatory text that is not the corrected transcription.
            """
        },
        {
            'role': 'user',
            'content': json.dumps({
                'text': 'Why not allow your silver tuff to luxuriate in a natural manner?',
                'confidence_score': 0.66
            })
        },
        #{'role': 'assistant', 'content': '{"text": "why not allow your silver tufts to luxuriate in a natural manner?"}'},
        {'role': 'assistant', 'content': "why not allow your silver tufts to luxuriate in a natural manner?"},
        {'role': 'user', 'content': json.dumps(asr_transcription)}
    ]
    return messages


def print_error(error_code):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if error_code <= 1:
        print(f"Exception type: {exc_type}")
    if 1 < error_code <= 2:
        print(f"Exception message: {exc_value}")
    if 2<error_code <= 3:
        print(f"Exception traceback: {exc_traceback}" )
    if 3<error_code <= 4:
        traceback.print_tb(exc_traceback)
        
def get_chatgpt_response(d):

    asr_transcription = d["asr_transcription"]
    reference_transcription = d["reference_transcription"]
    messages = get_messages(asr_transcription, delimiter)
    #print(messages)
        
    retries = 5 
    while retries > 0:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.1,
                request_timeout=60
            )
            corrected_asr_transcription = response.choices[0].message["content"]
            _ = {"text":corrected_asr_transcription}
            retries = -1

        except openai.error.Timeout: # timeout try again a few seconds later
            print_error(1)
            retries -= 1
            time.sleep(60)
            _ = {"text":None}

        except openai.error.RateLimitError: # quota exceeded
            print_error(1)
            _ = {"text":None}
            sys.exit(1)

        #except json.decoder.JSONDecodeError:
        #    print_error(1)
        #    print(f"transcription: {asr_transcription}\nChatGPT raw output: {corrected_asr_transcription}\n")
        #    _ = {"text":None}
        #    retries = -1

        except Exception as e:
            print_error(4)
            _ = {"text":None}
            sys.exit(1)

        finally:
            corrected_asr_transcription = _["text"]  
    
    return {
            "asr_transcription":asr_transcription,
            "reference_transcription":reference_transcription,
            "corrected_asr_transcription": corrected_asr_transcription
            }

 
if __name__ =="__main__":  
    
    parser = argparse.ArgumentParser(description="ChatGPT ASR Correction")
    parser.add_argument("-d", "--dataset", choices=["librispeech", "dummy"], \
            default="librispeech", help="Select the dataset (librispeech or dummy)")
    parser.add_argument("-n", "--num_data" , type = int, default = -1, help = "Select the number of data")
    args = parser.parse_args()
        
    # Load API key 
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY_Idiap")  
    #openai.api_key = os.getenv("OPENAI_API_KEY_MARYAM")    
    
    delimiter = "####"
    
    
    if args.dataset == "librispeech":
        transcription_file = "../../data/transcriptions/whisper_tiny_librispeech_dev-clean-full.json"  # experiment 2
        output_file = f"../../results/experiment2/whisper_corrected_transcriptions.json"  # experiment 2
        with open(transcription_file, "r") as f:
            json_obj=f.read()
            data=json.loads(json_obj)
            
        # experiment 2
        if args.num_data > 0:
            data = data[:args.num_data]
        for i,d in enumerate(data):
            asr_transcription = confidence_score_sentence_level(d["asr_transcription"])
            reference_transcription= d["reference_transcription"]
            data[i] = {"asr_transcription": asr_transcription, "reference_transcription": reference_transcription}
            
    elif args.dataset == "dummy":
        data = read_dummy_transcriptions()
        output_file = f"../../results/experiment2/dummy_corrected_transcriptions.json" 

    ### multithread parallelization 
    l = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_chatgpt_response, item):item for item in data}
        for future in tqdm(concurrent.futures.as_completed(futures)):
            item = futures[future]
            try:
                d = future.result()
                l.append(d)
            except Exception as exc:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(f"Exception type: {exc_type}")
                print(f"Exception message: {exc_value}")
                print(f"Exception traceback: {exc_traceback}" )
                traceback.print_tb(exc_traceback)
                print('%r generated an exception: %s' % (item, exc))

    ### asyncio parallelization
    # loop = asyncio.get_event_loop()
    # l = loop.run_until_complete(asyncio.gather(*(get_chatgpt_response(item) for item in data)))

    ### no parallelization
    # l =[]
    # for d in tqdm(data):
    #     l.append(get_chatgpt_response(d))
 
    with open(output_file, "w") as f:
        json_str = json.dumps(l, indent=2)
        f.write(json_str)

