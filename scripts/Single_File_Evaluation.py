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
    parser.add_argument("-e","--experiment", choices = ["exp_without_sentence_confidence","exp_without_lowest_confidence_word",  "exp_without_average_word_confidence", "exp_without_sentence_confidence_GPT-4","exp_without_lowest_confidence_word_GPT-4", "exp_certain_low_confidence_words_Thresh_0.55","exp_certain_low_confidence_words_Thresh_0.6","exp_certain_low_confidence_words_Thresh_0.65","exp_certain_low_confidence_words_Thresh_0.7"
    ,"exp_certain_low_confidence_words_Thresh_0.75","exp_certain_low_confidence_words_Thresh_0.8","exp_certain_low_confidence_words_Thresh_0.85",
    "exp_certain_low_confidence_words_Thresh_0.9","exp_certain_low_confidence_words_Thresh_0.95", "exp_certain_low_confidence_words_Thresh_0.6_GPT-4-Turbo", "exp_without_average_word_confidence_GPT-4"], help = "Select the experiment")
    args = parser.parse_args()

    if args.experiment == "exp_without_sentence_confidence":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_sentence_confidence/results_GPT-3.5-Turbo/results_without_sentence_confidence/whisper_corrected_transcriptions.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_sentence_confidence/results_GPT-3.5-Turbo/results_without_sentence_confidence/results_whisper_single.md")
    
    elif args.experiment == "exp_without_sentence_confidence_GPT-4":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_sentence_confidence/results_GPT-4-Turbo/results_without_sentence_confidence_GPT-4-Turbo/whisper_corrected_transcriptions.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_sentence_confidence/results_GPT-4-Turbo/results_without_sentence_confidence_GPT-4-Turbo/results_whisper_single.md")
        
    elif args.experiment == "exp_without_lowest_confidence_word":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_lowest_confidence_word/results_GPT-3.5-Turbo/results_without_lowest_confidence_word/whisper_corrected_transcriptions.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_lowest_confidence_word/results_GPT-3.5-Turbo/results_without_lowest_confidence_word/results_whisper_single.md")   
    
    elif args.experiment == "exp_without_lowest_confidence_word_GPT-4":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_lowest_confidence_word/results_GPT-4-Turbo/results_without_lowest_confidence_word_GPT-4-Turbo/whisper_corrected_transcriptions.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_lowest_confidence_word/results_GPT-4-Turbo/results_without_lowest_confidence_word_GPT-4-Turbo/results_whisper_single.md")
    
    elif args.experiment == "exp_without_average_word_confidence":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_average_word_confidence/results_GPT-3.5-Turbo/results_without_average_word_confidence/whisper_corrected_transcriptions.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_average_word_confidence/results_GPT-3.5-Turbo/results_without_average_word_confidence/results_whisper_single.md") 
        
    elif args.experiment == "exp_without_average_word_confidence_GPT-4":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_average_word_confidence/results_GPT-4-Turbo/results_without_average_word_confidence_GPT-4-Turbo/whisper_corrected_transcriptions.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_average_word_confidence/results_GPT-4-Turbo/results_without_average_word_confidence_GPT-4-Turbo/results_whisper_single.md")     
            
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.55":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/postprocessed_corrected_transcription/whisper_corrected_transcriptions_processed_Thresh=0.55.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/evaluation_results/evaluation_single/results_whisper_Thresh=0.55_single.md")   
            
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.6":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/postprocessed_corrected_transcription/whisper_corrected_transcriptions_processed_Thresh=0.6.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/evaluation_results/evaluation_single/results_whisper_Thresh=0.6_single.md")   
        
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.65":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/postprocessed_corrected_transcription/whisper_corrected_transcriptions_processed_Thresh=0.65.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/evaluation_results/evaluation_single/results_whisper_Thresh=0.65_single.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.7":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/postprocessed_corrected_transcription/whisper_corrected_transcriptions_processed_Thresh=0.7.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/evaluation_results/evaluation_single/results_whisper_Thresh=0.7_single.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.75":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/postprocessed_corrected_transcription/whisper_corrected_transcriptions_processed_Thresh=0.75.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/evaluation_results/evaluation_single/results_whisper_Thresh=0.75_single.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.8":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/postprocessed_corrected_transcription/whisper_corrected_transcriptions_processed_Thresh=0.8.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/evaluation_results/evaluation_single/results_whisper_Thresh=0.8_single.md") 
                
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.85":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/postprocessed_corrected_transcription/whisper_corrected_transcriptions_processed_Thresh=0.85.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/evaluation_results/evaluation_single/results_whisper_Thresh=0.85_single.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.9":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/postprocessed_corrected_transcription/whisper_corrected_transcriptions_processed_Thresh=0.9.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/evaluation_results/evaluation_single/results_whisper_Thresh=0.9_single.md")   
        
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.95":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/postprocessed_corrected_transcription/whisper_corrected_transcriptions_processed_Thresh=0.95.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-3.5-Turbo/evaluation_results/evaluation_single/results_whisper_Thresh=0.95_single.md")   
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.6_GPT-4-Turbo":
       CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-4-Turbo/postprocessed_corrected_transcription_GPT-4-Turbo/whisper_corrected_transcriptions_processed_Thresh=0.6.json")
       OUTPUT_FILE = os.path.join(Root,"results/results_certain_low_confidence_words/results_GPT-4-Turbo/evaluation_results_GPT-4-Turbo/results_whisper_Thresh=0.6_single.md")
    
      
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
                
            
