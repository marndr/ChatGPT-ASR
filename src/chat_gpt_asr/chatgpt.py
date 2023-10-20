from dotenv import load_dotenv
import json
import openai
from tqdm import tqdm
import os
import argparse
from chat_gpt_asr.utils import read_dummy_transcriptions

import multiprocessing

def chatgpt(messages, openai, model="gpt-3.5-turbo", temperature=0.1):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]

def get_messages(asr_transcription, delimiter="####"):
    messages = [
        {'role': 'system', 'content': f"""You are a helpful assistant that corrects ASR errors. \
        You will be provided with the ASR transcription delimited with {delimiter} characters. \
        correct the errors in the ASR transcription. Provide your output in json format with key: text which is the text with ASR errors being corrected. \
        Do not output any additional text that is not in JSON format. \
        Do not write any explanatory text after outputting the requested JSON. \
            """},
        {'role': 'user', 'content': f'{delimiter}The day he is coming, said Pomeethias. When Jupiter will send a flood the destroy mankind from the earth.{delimiter}'},
        {'role': 'assistant', 'content': '{"text": "THE DAY IS COMING SAID PROMETHEUS WHEN JUPITER WILL SEND A FLOOD TO DESTROY MANKIND FROM THE EARTH"}'},
        {'role': 'user', 'content': f'{delimiter}{asr_transcription}{delimiter}'}

    ]
    return messages


if __name__ =="__main__":

    parser = argparse.ArgumentParser(description="ChatGPT ASR Correction")
    parser.add_argument("-d", "--dataset", choices=["librispeech", "dummy"], default="librispeech", help="Select the dataset (librispeech or dummy)")
    parser.add_argument("-n", type = int, default = -1, help = "Select the number of data")

    args = parser.parse_args()

    # Load API key
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY_MARYAM")

    delimiter = "####"
    whisper_model = "tiny"

    if args.dataset == "librispeech":
        subset = "dev-clean"
        with open(
            f"data/transcriptions/whisper_{whisper_model}_librispeech_{subset}.json"
        ) as f:
            json_obj=f.read()
            data=json.loads(json_obj)
            if args.n!=-1:
                data =data[:args.n]
        filename = "whisper_corrected_transcriptions.json"

    elif args.dataset == "dummy":
        data = read_dummy_transcriptions()
        filename = "dummy_corrected_transcriptions.json"

    def get_chatgpt_response(d):
        asr_transcription = d["asr_transcription"]
        reference_transcription = d["reference_transcription"]
        messages = get_messages(asr_transcription, delimiter)
        try:
            corrected_asr_transcription = chatgpt(messages, openai)
            _ = json.loads(corrected_asr_transcription)

        except json.decoder.JSONDecodeError:
            _ = {"text":None}

        except:
            _ = {"text":None}

        finally:
            corrected_asr_transcription = _["text"]

        # print(f"cpu {multiprocessing.current_process()} is running ...")
        return {
                "asr_transcription":asr_transcription,
                "reference_transcription":reference_transcription,
                "corrected_asr_transcription": corrected_asr_transcription
                }


    with multiprocessing.Pool(processes=8) as pool:
        l = pool.map(get_chatgpt_response, data)

    with open("../../data/"+filename, "w") as f:
        json_str = json.dumps(l, indent=2)
        f.write(json_str)
