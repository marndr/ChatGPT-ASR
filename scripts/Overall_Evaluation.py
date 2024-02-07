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
    parser.add_argument("-e","--experiment", choices = ["exp_without_sentence_confidence_tiny","exp_without_lowest_word_confidence_tiny",  "exp_without_average_word_confidence_tiny", "exp_without_sentence_confidence_GPT-4-Turbo_tiny","exp_without_lowest_word_confidence_GPT-4_tiny", "exp_certain_low_confidence_words_Thresh_0.55_tiny","exp_certain_low_confidence_words_Thresh_0.6_tiny","exp_certain_low_confidence_words_Thresh_0.65_tiny",
"exp_certain_low_confidence_words_Thresh_0.7_tiny","exp_certain_low_confidence_words_Thresh_0.75_tiny","exp_certain_low_confidence_words_Thresh_0.8_tiny",
"exp_certain_low_confidence_words_Thresh_0.85_tiny","exp_certain_low_confidence_words_Thresh_0.9_tiny","exp_certain_low_confidence_words_Thresh_0.95_tiny", "exp_certain_low_confidence_words_Thresh_0.6_GPT-4-Turbo_tiny", "exp_without_average_word_confidence_GPT-4-Turbo_tiny", "exp_sentence_confidence_GPT-3.5_medium", "exp_lowest_word_confidence_GPT-3.5_medium","exp_sentence_confidence_GPT-4_medium", "exp_sentence_confidence_GPT-3.5_large-v3", "exp_lowest_word_confidence_GPT-3.5_large-v3", "exp_sentence_confidence_GPT-4_large-v3", "exp_sentence_confidence_GPT-3.5_noisy_large-v3","exp_lowest_word_confidence_GPT-3.5_noisy_large-v3","exp_sentence_confidence_GPT-4_noisy_large-v3"], help = "Select the experiment")
    args = parser.parse_args()

    if args.experiment == "exp_without_sentence_confidence_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_sentence_confidence_tiny/results_overall_sentence_confidence_tiny.md")
    
    elif args.experiment == "exp_without_sentence_confidence_GPT-4-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_without_sentence_confidence_GPT-4-Turbo_tiny/results_overall_sentence_confidence_GPT-4-Turbo_tiny.md")
        
    elif args.experiment == "exp_without_lowest_word_confidence_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_lowest_word_confidence_tiny/corrected_transcriptions_lowest_word_confidence_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_lowest_word_confidence_tiny/results_overall_lowest_word_confidence_tiny.md")   
    
    elif args.experiment == "exp_without_lowest_word_confidence_GPT-4-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/results_overall_lowest_word_confidence_GPT-4-Turbo_tiny.md")
    
    elif args.experiment == "exp_without_average_word_confidence_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_average_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_average_word_confidence_tiny/corrected_transcriptions_average_word_confidence_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_average_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_average_word_confidence_tiny/results_overall_average_word_confidence_tiny.md") 
        
    elif args.experiment == "exp_without_average_word_confidence_GPT-4-Turbo_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_average_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_average_word_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_average_word_confidence_GPT-4-Turbo_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_average_word_confidence_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_without_average_word_confidence_GPT-4-Turbo_tiny/results_overall_average_word_confidence_GPT-4-Turbo_tiny.md")     
            
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.55_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clrean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.55_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.55_tiny.md")   
            
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.6_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.6_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.6_tiny.md")   
        
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.65_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.65_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.65_tiny.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.7_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.7_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.7_tiny.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.75_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.75_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.75_tiny.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.8_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.8_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.8_tiny.md") 
                
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.85_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.85_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.85_tiny.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.9_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.9_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.9_tiny.md")   
        
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.95_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.95_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_certain_low_confidence_words_thresh_tiny/results_certain_low_confidence_words_Thresh=0.95_tiny.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.6_GPT-4-Turbo_tiny":
       CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/postprocessed_corrected_transcriptions_certain_low_confidence_words_GPT-4-Turbo_tiny/postprocessed_corrected_transcriptions_Thresh=0.6_tiny.json")
       OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_certain_low_confidence_words_GPT-4-Turbo_tiny/results_certain_low_confidence_words_Thresh=0.6_GPT-4-Turbo_tiny.md")
    
    
    elif args.experiment == "exp_sentence_confidence_GPT-3.5_medium":
       CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/corrected_transcriptions_sentence_confidence_medium.json")
       OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_sentence_confidence_medium/results_overall_sentence_confidence_medium.md")
    
    elif args.experiment == "exp_lowest_word_confidence_GPT-3.5_medium":
       CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_medium/corrected_transcriptions_lowest_word_confidence_medium.json")
       OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_medium/results_lowest_word_confidence_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_medium/results_overall_lowest_word_confidence_medium.md")        
    
    elif args.experiment == "exp_sentence_confidence_GPT-4_medium":
       CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-1106-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_medium.json")
       OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_medium/results_sentence_confidence_medium/results_GPT-4-Turbo_medium/gpt-4-1106-preview/results_without_sentence_confidence_GPT-4-Turbo_medium/results_overall_sentence_confidence_GPT-4-Turbo_medium.md")
       
    elif args.experiment == "exp_sentence_confidence_GPT-3.5_large-v3":
       CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_sentence_confidence_large-v3/corrected_transcriptions_sentence_confidence_large-v3.json")
       OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_sentence_confidence_large-v3/results_overall_sentence_confidence_large-v3.md")
    
    elif args.experiment == "exp_lowest_word_confidence_GPT-3.5_large-v3":
       CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_large-v3/corrected_transcriptions_lowest_word_confidence_large-v3.json")
       OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_large-v3/results_overall_lowest_word_confidence_large-v3.md")    
    
    elif args.experiment == "exp_sentence_confidence_GPT-4_large-v3":
       CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-1106-preview/results_without_sentence_confidence_GPT-4-Turbo_large-v3/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_large-v3.json")
       OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-4-Turbo_large-v3/gpt-4-1106-preview/results_without_sentence_confidence_GPT-4-Turbo_large-v3/results_overall_sentence_confidence_GPT-4-Turbo_large-v3.md")
    
    elif args.experiment == "exp_sentence_confidence_GPT-3.5_noisy_large-v3":
       CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-0125/results_without_sentence_confidence_noisy_large-v3/corrected_transcriptions_sentence_confidence_noisy_large-v3.json")
       OUTPUT_FILE = os.path.join(Root,"results/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-0125/results_without_sentence_confidence_noisy_large-v3/results_overall_sentence_confidence_noisy_large-v3.md")          
    
    elif args.experiment == "exp_lowest_word_confidence_GPT-3.5_noisy_large-v3":
       CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_noisy_large-v3/corrected_transcriptions_lowest_word_confidence_noisy_large-v3.json")
       OUTPUT_FILE = os.path.join(Root,"results/results_noisy/results_noisy_large-v3/results_lowest_word_confidence_noisy_large-v3/results_GPT-3.5-Turbo_noisy_large-v3/gpt-3.5-turbo-0125/results_without_lowest_word_confidence_noisy_large-v3/results_overall_lowest_word_confidence_noisy_large-v3.md") 
    
    elif args.experiment == "exp_sentence_confidence_GPT-4_noisy_large-v3":
       CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_sentence_confidence_noisy_large-v3/corrected_transcriptions_sentence_confidence_noisy_large-v3.json")
       OUTPUT_FILE = os.path.join(Root,"results/results_noisy/results_noisy_large-v3/results_sentence_confidence_noisy_large-v3/results_GPT-4-Turbo_noisy_large-v3/gpt-4-0125-preview/results_without_sentence_confidence_noisy_large-v3/results_overall_sentence_confidence_noisy_large-v3.md") 
               
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
    
    
    
