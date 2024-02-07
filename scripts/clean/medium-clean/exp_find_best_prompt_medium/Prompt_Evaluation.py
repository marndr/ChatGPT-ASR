import os
from chat_gpt_asr.utils import remove_punctuations, ser
import json
from jiwer import cer, wer
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

load_dotenv()
Root = os.getenv("ROOT_PATH")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ChatGPT ASR Correction")
    parser.add_argument("-d", "--dataset", choices=["librispeech"], default="librispeech", help="Select the dataset (librispeech)")
    parser.add_argument("-e","--experiment", choices = ["exp_new_prompt_sentence_confidence_1_medium","exp_new_prompt_lowest_word_confidence_1_medium"])
    args = parser.parse_args()
    
    
    
    if args.experiment =="exp_new_prompt_sentence_confidence_1_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_medium/results_find_best_prompt_medium/corrected_transcriptions_sentence_confidence_prompt_1.json")
        
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_medium/results_find_best_prompt_medium/results_overall_sentence_confidence_prompt_1_medium.md")  
    
    if args.experiment =="exp_new_prompt_lowest_word_confidence_1_medium":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_lowest_word_confidence_new_prompts_medium/results_find_best_prompt_medium/corrected_transcriptions_lowest_word_confidence_prompt_1.json")
        
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_lowest_word_confidence_new_prompts_medium/results_find_best_prompt_medium/results_overall_lowest_word_confidence_prompt_1_medium.md") 
                                       
    if args.dataset == "librispeech":
        with open(CORRECTED_TRANSCRIPTIONS_LIBRISPEECH, "r") as f:
            json_obj=f.read()
            data=json.loads(json_obj)
        output_filename= OUTPUT_FILE
 
        
   
    ref_l, hyp_l, hyp_l_original = [], [], []
    ser_corrected_chatgpt, ser_original = 0.,0.
    count = 0
    
    for d in tqdm(data):
        
        if d["corrected_asr_transcription"] is None:
            continue 
            
        
        ref_l.append(remove_punctuations(d["reference_transcription"].lower()))
        hyp_l.append(remove_punctuations(d["corrected_asr_transcription"].lower()))
        hyp_l_original.append(remove_punctuations(d["asr_transcription"]["text"].lower()))  
        
        SER_chatgpt_, SER_original_ = ser(d["asr_transcription"]["text"],
        d["corrected_asr_transcription"],d["reference_transcription"])
    
        count += 1
        ser_corrected_chatgpt += SER_chatgpt_
        ser_original += SER_original_
    
    
    
    wer_corrected_chatgpt= wer(hyp_l,ref_l) * 100
    wer_original= wer(hyp_l_original, ref_l) * 100
    cer_corrected_chatgpt=cer(hyp_l,ref_l) * 100
    cer_original = cer(hyp_l_original, ref_l) * 100
    
    ser_corrected_chatgpt /=count
    ser_original /= count
    
    print(f"Number of audio files:  {count:}\n")
    print(f"WER_original is:                {wer_original:.04f}\n")
    print(f"WER_corrected_chatgpt is:       {wer_corrected_chatgpt:.04f}\n")
    print(f"CER_original is:                {cer_original:.04f}\n")
    print(f"CER_corrected_chatgpt is:       {cer_corrected_chatgpt:.04f}\n")
    print(f"SER_original is:                {ser_original:.04f}\n")
    print(f"SER_corrected_chatgpt is:       {ser_corrected_chatgpt:.04f}\n")
    
    with open(output_filename, "w") as f:
        f.write(f"""
        Number of evaluated audio files:  {count:}
        WER_original:                   {wer_original:.04f}%    
        WER_corrected_chatgpt:          {wer_corrected_chatgpt:.04f}%
        CER_original:                   {cer_original:.04f}%    
        CER_corrected_chatgpt:          {cer_corrected_chatgpt:.04f}%
        SER_original:                   {ser_original:.04f}%    
        SER_corrected_chatgpt:          {ser_corrected_chatgpt:.04f}%
        ---
    """)
