import argparse
import json
import os

from dotenv import load_dotenv
import openai

from chat_gpt_asr.chatgpt import multithread_parallelization
from chat_gpt_asr.utils import confidence_score_word_level

load_dotenv()
Root = os.getenv("ROOT_PATH")

THRESH = 0.2


TRANSCRIPTION_FILENAME = os.path.join(Root, "data/transcriptions/whisper_tiny_librispeech_dev-clean-full.json") 
CORRECTED_TRANSCRIPTION_FILENAME = os.path.join(Root,"results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/preprocessed_corrected_transcriptions_certain_low_confidence_words_tiny_new/preprocessed_corrected_transcriptions_Thresh=0.2_tiny.json")
    
def get_messages_exp(asr_transcription):
    messages = [
        {
            'role': 'system',
            'content': f"""You are a helpful assistant that corrects ASR errors. \
             You will be presented with an ASR transcription (from Librispeech data provided by the Whisper model) and a list of words in the transcription with low confidence scores. \ 
             The input will be formatted as json with keys: text and low_confidence_words,\
             where the text is the ASR transcription and low_confidence_words contains the list of words in the transcription with low confidence scores. \
             Your task is to correct any errors in the transcription.
             If you come across errors in ASR transcription, make sure that \
             you correct only words from within the low_confidence_words list and \
             your corrections should closely match the original transcription acoustically or phonetically.\
             Provide the most probable corrected transcription in string format. \
             Do not change the case, for example, lower case or upper case, in the transcription. \
             Do not output any additional text that is not the corrected transcription. \
             Do not write any explanatory text that is not the corrected transcription.
             """
        },
        {
            'role': 'user',
            'content': '{"text": "Why not allow your silver tuff to luxuriate in a natural manner?", "low_confidence_words":["tuff"]}'
        },
        {'role': 'assistant', 'content': "why not allow your silver tufts to luxuriate in a natural manner?"},
        
        {
        'role': 'user',
        'content': '{"text": "Meanwhile, how fair did it with the flowers?", "low_confidence_words":["fared"]}'
        },
        {
        'role': 'assistant', 
        'content': "Meanwhile, how fared did it with the flowers?"
        },
        
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
            
        if args.num_data > 0:
            data = data[:args.num_data]

        data1 = []
        for i,d in enumerate(data):
            asr_transcription = confidence_score_word_level(d["asr_transcription"], confidence = True) 

            low_confidence_words = [word["text"] for word in asr_transcription["words"] 
                                    if word["confidence"] <= THRESH]
            if len(low_confidence_words) > 0:
                                       
            	asr_transcription = {"text": asr_transcription["text"], "low_confidence_words": low_confidence_words}
    
            	reference_transcription= d["reference_transcription"]

            	# data[i] = {"asr_transcription": asr_transcription, "reference_transcription": reference_transcription}
            	data1.append({"asr_transcription": asr_transcription, "reference_transcription": reference_transcription})
   

    l= multithread_parallelization(data1, get_messages_fn=get_messages_exp,model = "gpt-3.5-turbo-1106")
 
    with open(output_file, "w") as f:
        json_str = json.dumps(l, indent=2)
        f.write(json_str)
        
        
        
        
        
