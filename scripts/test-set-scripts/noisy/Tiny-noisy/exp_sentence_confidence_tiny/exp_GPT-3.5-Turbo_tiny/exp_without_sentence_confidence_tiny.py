import argparse
import json
import os

from dotenv import load_dotenv
import openai

from chat_gpt_asr.chatgpt import multithread_parallelization
from chat_gpt_asr.utils import confidence_score_sentence_level

load_dotenv()
Root = os.getenv("ROOT_PATH")


TRANSCRIPTION_FILENAME = os.path.join(Root,"data/transcriptions/whisper_tiny_librispeech_test-other-full.json") 

CORRECTED_TRANSCRIPTION_FILENAME = os.path.join(Root,"results/results-test-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json")  
    
def get_messages_exp1(asr_transcription):
    messages = [
        {
            'role': 'system',
            'content': f"""You are a helpful assistant that corrects ASR errors. \
            You will be presented with an ASR transcription of Librispeech data provided by the Whisper model. \
            Your task is to correct any errors in the transcription.\
            Provide the most probable corrected transcription in string format. \
            If you come across errors in ASR transcription, make corrections that closely match the original transcription acoustically or phonetically.\                                                                                                                  
            Do not change the case, for example, lower case or upper case, in the transcription. \
            Do not output any additional text that is not the corrected transcription. \
            Do not write any explanatory text that is not the corrected transcription.
            """
        },
        {
            'role': 'user',
            'content': '{"text": "Why not allow your silver tuff to luxuriate in a natural manner?"}'
        },
        {'role': 'assistant', 'content': "why not allow your silver tufts to luxuriate in a natural manner?"},
        
        {
            'role': 'user',
            'content': '{"text": "Meanwhile, how fair did it with the flowers?"}'
        },
        {'role': 'assistant', 'content': "Meanwhile, how fared did it with the flowers?"},
        {'role': 'user', 'content': json.dumps(asr_transcription)}
    ]
    return messages


if __name__ =="__main__": 
     
    parser = argparse.ArgumentParser(description="ChatGPT ASR Correction")
    parser.add_argument("-d", "--dataset", choices=["librispeech"], \
            default="librispeech", help="Select the dataset (librispeech)")
    parser.add_argument("-n", "--num_data" , type = int, default = -1, help = "Select the number of data")
    args = parser.parse_args()

    
    # Load API key 
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY_Idiap")  
    #openai.api_key = os.getenv("OPENAI_API_KEY_MARYAM")    

    if args.dataset == "librispeech":
        transcription_file = TRANSCRIPTION_FILENAME
        output_file = CORRECTED_TRANSCRIPTION_FILENAME
        with open(transcription_file, "r") as f:
            json_obj=f.read()
            data=json.loads(json_obj)
            data = [d for d in data if d["asr_transcription"]]
            
        if args.num_data > 0:
            data = data[:args.num_data]

        # experiment 1
        for i,d in enumerate(data):
            asr_transcription = confidence_score_sentence_level(d["asr_transcription"] , confidence = True) 
            reference_transcription= d["reference_transcription"]
            data[i] = {"asr_transcription": asr_transcription, "reference_transcription": reference_transcription}
            
    elif args.dataset == "dummy":
        data = read_dummy_transcriptions()
        output_file = os.path.join(root,"results/experiment_without_confidence/dummy_corrected_transcriptions.json") 

    l= multithread_parallelization(data, get_messages_fn=get_messages_exp1 , model = "gpt-3.5-turbo-0125" )
 
 
 
    with open(output_file, "w") as f:
        json_str = json.dumps(l, indent=2)
        f.write(json_str)

