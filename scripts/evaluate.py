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
    parser.add_argument("-e","--experiment", choices = ["exp_without_sentence_confidence","exp_without_lowest_word_confidence", "exp_certain_low_confidence_words", "exp_without_sentence_confidence_GPT-4","exp_without_lowest_word_confidence_GPT-4", "exp_certain_low_confidence_words_Thresh_0.55","exp_certain_low_confidence_words_Thresh_0.6","exp_certain_low_confidence_words_Thresh_0.65","exp_certain_low_confidence_words_Thresh_0.7"
    ,"exp_certain_low_confidence_words_Thresh_0.75","exp_certain_low_confidence_words_Thresh_0.8","exp_certain_low_confidence_words_Thresh_0.85",
    "exp_certain_low_confidence_words_Thresh_0.9","exp_certain_low_confidence_words_Thresh_0.95", "exp_certain_low_confidence_words_Thresh_0.6_GPT-4-Turbo"], help = "Select the experiment")
    args = parser.parse_args()

    if args.experiment == "exp_without_sentence_confidence":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-3.5-Turbo/results_sentence_confidence/results_without_sentence_confidence/whisper_corrected_transcriptions.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-3.5-Turbo/results_sentence_confidence/results_without_sentence_confidence/results_whisper.md")

    elif args.experiment == "exp_without_lowest_word_confidence":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-3.5-Turbo/results_lowest_word_confidence/results_without_lowest_word_confidence/whisper_corrected_transcriptions.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-3.5-Turbo/results_lowest_word_confidence/results_without_lowest_word_confidence/results_whisper.md")   
        
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.55":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/whisper_corrected_transcriptions_Thresh=0.55.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/results_whisper_Thresh=0.55.md")   
            
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.6":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/whisper_corrected_transcriptions_Thresh=0.6.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/results_whisper_Thresh=0.6.md")   
        
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.65":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/whisper_corrected_transcriptions_Thresh=0.65.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/results_whisper_Thresh=0.65.md") 
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.7":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/whisper_corrected_transcriptions_Thresh=0.7.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/results_whisper_Thresh=0.7.md") 
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.75":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/whisper_corrected_transcriptions_Thresh=0.75.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/results_whisper_Thresh=0.75.md") 
            
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.8":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/whisper_corrected_transcriptions_Thresh=0.8.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/results_whisper_Thresh=0.8.md")
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.85":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/whisper_corrected_transcriptions_Thresh=0.85.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/results_whisper_Thresh=0.85.md")
                 
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.9":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/whisper_corrected_transcriptions_Thresh=0.9.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/results_whisper_Thresh=0.9.md") 
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.95":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/whisper_corrected_transcriptions_Thresh=0.95.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-3.5-Turbo/results_certain_low_confidence_words/results_whisper_Thresh=0.95.md") 
    
    
    elif args.experiment == "exp_without_sentence_confidence_GPT-4":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-4-Turbo/results_sentence_confidence_GPT-4/results_without_sentence_confidence_GPT-4/whisper_corrected_transcriptions.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-4-Turbo/results_sentence_confidence_GPT-4/results_without_sentence_confidence_GPT-4/results_whisper.md")

    elif args.experiment == "exp_without_lowest_word_confidence_GPT-4":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-4-Turbo/results_lowest_word_confidence_GPT-4/results_without_lowest_word_confidence_GPT-4/whisper_corrected_transcriptions.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-4-Turbo/results_lowest_word_confidence_GPT-4/results_without_lowest_word_confidence_GPT-4/results_whisper.md")
    
    
    elif args.experiment == "exp_certain_low_confidence_words_Thresh_0.6_GPT-4-Turbo":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_GPT-4-Turbo/results_certain_low_confidence_words_GPT-4-Turbo/whisper_corrected_transcriptions_Thresh=0.6.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_GPT-4-Turbo/results_certain_low_confidence_words_GPT-4-Turbo/results_whisper_Thresh=0.6.md")   
      
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
            
        #evaluation for experiment_without_confidence
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
    
    
    print(f"WER_original is:           {wer_original:.04f}\n")
    print(f"WER_corrected_chatgpt is:  {wer_corrected_chatgpt:.04f}\n")
    print(f"CER_original is:           {cer_original:.04f}\n")
    print(f"CER_corrected_chatgpt is:  {cer_corrected_chatgpt:.04f}\n")
    print(f"SER_original is:           {ser_original:.04f}\n")
    print(f"SER_corrected_chatgpt is:  {ser_corrected_chatgpt:.04f}\n")
    
    with open(output_filename, "w") as f:
        f.write(f"""
        WER_original:           {wer_original:.04f}%    
        WER_corrected_chatgpt:  {wer_corrected_chatgpt:.04f}%
        CER_original:           {cer_original:.04f}%    
        CER_corrected_chatgpt:  {cer_corrected_chatgpt:.04f}%
        SER_original:           {ser_original:.04f}%    
        SER_corrected_chatgpt:  {ser_corrected_chatgpt:.04f}%
        ---
    """)
    
    
    
