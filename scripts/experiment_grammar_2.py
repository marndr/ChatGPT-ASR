

# experiment 2
import argparse
import json
import os

from dotenv import load_dotenv
import openai
from chat_gpt_asr.chatgpt import multithread_parallelization

from chat_gpt_asr.utils import confidence_score_sentence_level, read_dummy_transcriptions

root = "/home/mnaderi/Documents/thesis/chat-gpt-asr"

delimiter = "####"
CONFIDENCE = True
TRANSCRIPTION_FILENAME = os.path.join(root, "data/transcriptions/whisper_tiny_librispeech_dev-clean-full.json") # experiment 2
CORRECTED_TRANSCRIPTION_FILENAME = os.path.join(root, "results/experiment2/whisper_corrected_transcriptions.json")  # experiment 2
    
def get_messages_exp2(asr_transcription, delimiter="####"):
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
    
    if args.dataset == "librispeech":
        transcription_file = TRANSCRIPTION_FILENAME
        output_file = CORRECTED_TRANSCRIPTION_FILENAME
        with open(transcription_file, "r") as f:
            json_obj=f.read()
            data=json.loads(json_obj)
            
        if args.num_data > 0:
            data = data[:args.num_data]

        
        for i,d in enumerate(data):
            asr_transcription = confidence_score_sentence_level(d["asr_transcription"],CONFIDENCE) 
            reference_transcription= d["reference_transcription"]
            data[i] = {"asr_transcription": asr_transcription, "reference_transcription": reference_transcription}
            
    elif args.dataset == "dummy":
        data = read_dummy_transcriptions()
        output_file = os.path.join(root,"results/experiment2/dummy_corrected_transcriptions.json") 

    l= multithread_parallelization(data, get_messages_fn=get_messages_exp2)
 
    with open(output_file, "w") as f:
        json_str = json.dumps(l, indent=2)
        f.write(json_str)