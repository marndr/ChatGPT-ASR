import os
from chat_gpt_asr.utils import remove_punctuations
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
    parser.add_argument("-e","--experiment", choices = ["exp_without_sentence_confidence_single_tiny","exp_without_lowest_word_confidence_single_tiny",  "exp_without_average_word_confidence_single_tiny", "exp_without_sentence_confidence_GPT-4-Turbo_single_tiny","exp_without_lowest_word_confidence_GPT-4-Turbo_single_tiny","exp_certain_low_confidence_words_Thresh_0.55","exp_certain_low_confidence_words_Thresh_0.6","exp_certain_low_confidence_words_Thresh_0.65","exp_certain_low_confidence_words_Thresh_0.7"
    ,"exp_certain_low_confidence_words_Thresh_0.75","exp_certain_low_confidence_words_Thresh_0.8","exp_certain_low_confidence_words_Thresh_0.85",
    "exp_certain_low_confidence_words_Thresh_0.9","exp_certain_low_confidence_words_Thresh_0.95", "exp_certain_low_confidence_words_Thresh_0.6_GPT-4-Turbo", "exp_without_average_word_confidence_GPT-4-Turbo_single_tiny"], help = "Select the experiment")
    args = parser.parse_args()

    if args.experiment == "exp_without_sentence_confidence_single_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/results_without_sentence_confidence_tiny/results_single_sentence_confidence_tiny.md")
    
    elif args.experiment == "exp_without_sentence_confidence_GPT-4-Turbo_single_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/results_without_sentence_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_sentence_confidence_tiny/results_GPT-4-Turbo_tiny/results_without_sentence_confidence_GPT-4-Turbo_tiny/results_single_sentence_confidence_GPT-4-Turbo_tiny.md")
        
    elif args.experiment == "exp_without_lowest_word_confidence_single_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/results_without_lowest_word_confidence_tiny/corrected_transcriptions_lowest_word_confidence_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_lowest_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/results_without_lowest_word_confidence_tiny/results_single_lowest_word_confidence_tiny.md")   
    
    elif args.experiment == "exp_without_lowest_word_confidence_GPT-4-Turbo_single_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_lowest_word_confidence_tiny/results_GPT-4-Turbo_tiny/results_without_lowest_word_confidence_GPT-4-Turbo_tiny/results_single_lowest_word_confidence_GPT-4-Turbo_tiny.md")
    
    elif args.experiment == "exp_without_average_word_confidence_single_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_average_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/results_without_average_word_confidence_tiny/corrected_transcriptions_average_word_confidence_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_average_word_confidence_tiny/results_GPT-3.5-Turbo_tiny/results_without_average_word_confidence_tiny/results_single_average_word_confidence_tiny.md") 
        
    elif args.experiment == "exp_without_average_word_confidence_GPT-4-Turbo_single_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_average_word_confidence_tiny/results_GPT-4-Turbo_tiny/results_without_average_word_confidence_GPT-4-Turbo_tiny/corrected_transcriptions_average_word_confidence_GPT-4-Turbo_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_average_word_confidence_tiny/results_GPT-4-Turbo_tiny/results_without_average_word_confidence_GPT-4-Turbo_tiny/results_single_average_word_confidence_GPT-4-Turbo_tiny.md")     
            
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.55_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/postprocessed_corrected_transcription_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.55_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/results_certain_low_confidence_words_tiny/results_single_certain_low_confidence_words_thresh_tiny/results_single_certain_low_confidence_words_Thresh=0.55_tiny.md")   
            
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.6_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/postprocessed_corrected_transcription_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.6_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/results_certain_low_confidence_words_tiny/results_single_certain_low_confidence_words_thresh_tiny/results_single_certain_low_confidence_words_Thresh=0.6_tiny.md")   
        
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.65_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/postprocessed_corrected_transcription_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.65_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/results_certain_low_confidence_words_tiny/results_single_certain_low_confidence_words_thresh_tiny/results_single_certain_low_confidence_words_Thresh=0.65_tiny.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.7_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/postprocessed_corrected_transcription_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.7_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/results_certain_low_confidence_words_tiny/results_single_certain_low_confidence_words_thresh_tiny/results_single_certain_low_confidence_words_Thresh=0.7_tiny.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.75_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/postprocessed_corrected_transcription_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.75_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/results_certain_low_confidence_words_tiny/results_single_certain_low_confidence_words_thresh_tiny/results_single_certain_low_confidence_words_Thresh=0.75_tiny.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.8_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/postprocessed_corrected_transcription_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.8_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/results_certain_low_confidence_words_tiny/results_single_certain_low_confidence_words_thresh_tiny/results_single_certain_low_confidence_words_Thresh=0.8_tiny.md") 
                
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.85_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/postprocessed_corrected_transcription_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.8_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/results_certain_low_confidence_words_tiny/results_single_certain_low_confidence_words_thresh_tiny/results_single_certain_low_confidence_words_Thresh=0.8_tiny.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.9_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/postprocessed_corrected_transcription_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.8_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/results_certain_low_confidence_words_tiny/results_single_certain_low_confidence_words_thresh_tiny/results_single_certain_low_confidence_words_Thresh=0.8_tiny.md")   
        
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.95_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/postprocessed_corrected_transcription_certain_low_confidence_words_tiny/postprocessed_corrected_transcriptions_Thresh=0.8_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/results_certain_low_confidence_words_tiny/results_single_certain_low_confidence_words_thresh_tiny/results_single_certain_low_confidence_words_Thresh=0.8_tiny.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.6_GPT-4-Turbo_tiny":
       CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-4-Turbo_tiny/postprocessed_corrected_transcriptions_certain_low_confidence_words_GPT-4-Turbo_tiny/postprocessed_corrected_transcriptions_Thresh=0.6_tiny.json")
       OUTPUT_FILE = os.path.join(Root,"results/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-4-Turbo_tiny/results_certain_low_confidence_words_GPT-4-Turbo_tiny/results_single_certain_low_confidence_words_Thresh=0.6_GPT-4-Turbo_tiny")
    
      
    if args.dataset == "librispeech":
        with open(CORRECTED_TRANSCRIPTIONS_LIBRISPEECH, "r") as f:
            json_obj=f.read()
            data=json.loads(json_obj)
        output_filename= OUTPUT_FILE
 
        
    with open(output_filename, "w") as  f:
    
        ref_l, hyp_l, hyp_l_original = [], [], []
        
    
        for d in tqdm(data):
        
            if d["corrected_asr_transcription"] is None:
                continue 
            
      
            ref_l = remove_punctuations(d["reference_transcription"].lower())
            hyp_l = remove_punctuations(d["corrected_asr_transcription"].lower())
            hyp_l_original = remove_punctuations(d["asr_transcription"]["text"].lower())

            wer_corrected_chatgpt = wer([hyp_l], [ref_l]) * 100
            wer_original = wer([hyp_l_original], [ref_l]) * 100
            cer_corrected_chatgpt = cer([hyp_l], [ref_l]) * 100
            cer_original = cer([hyp_l_original], [ref_l]) * 100
        
            if (wer_corrected_chatgpt < wer_original) and (cer_corrected_chatgpt > cer_original):
            
                f.write(
                    f"""
                    ASR Transcription:        {d["asr_transcription"]["text"]}
                    Reference Transcription:  {d["reference_transcription"]}
                    Corrected Transcription:  {d["corrected_asr_transcription"]}
                
                    WER_corrected_chatgpt:    {wer_corrected_chatgpt:.04f}%
                    WER_original:             {wer_original:.04f}%
                    CER_corrected_chatgpt:    {cer_corrected_chatgpt:.04f}%
                    CER_original:             {cer_original:.04f}%
                
                    ---
                    """
                 )

            if (wer_corrected_chatgpt==0 and cer_corrected_chatgpt !=0) or (wer_original == 0 and cer_original !=0):
                
                f.write(
                    f"""
                    Evaluation are wrong in this audio file ((wer_corrected_chatgpt==0 and cer_corrected_chatgpt !=0) or (wer_original == 0 and cer_original !=0)) !!!!
                    
                    ASR Transcription:        {d["asr_transcription"]["text"]}
                    Reference Transcription:  {d["reference_transcription"]}
                    Corrected Transcription:  {d["corrected_asr_transcription"]}
                
                    WER_corrected_chatgpt:    {wer_corrected_chatgpt:.04f}%
                    WER_original:             {wer_original:.04f}%
                    CER_corrected_chatgpt:    {cer_corrected_chatgpt:.04f}%
                    CER_original:             {cer_original:.04f}%
                
                    ---
                    """
                 )
                
            
