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
    parser.add_argument("-e","--experiment", choices = ["exp_new_prompt_sentence_confidence_temp_zero", "exp_new_prompt_sentence_confidence_temp_0.5","exp_new_prompt_lowest_word_confidence_temp_zero","exp_new_prompt_lowest_word_confidence_temp_0.5","exp_prompt_1","exp_prompt_2", "exp_prompt_3", "exp_prompt_4", "exp_prompt_5", "exp_prompt_original", "exp_prompt_6", "exp_prompt_7", "exp_prompt_8", "exp_prompt_9", "exp_prompt_10", "exp_prompt_11", "exp_prompt_12", "exp_prompt_13"], help = "Select the experiment")
    args = parser.parse_args()
    
    if args.experiment == "exp_new_prompt_sentence_confidence_temp_zero":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_sentence_confidence_new_prompt/temp_zero/results_without_sentence_confidence/whisper_corrected_transcriptions_sentence_confidence_temp_zero.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_sentence_confidence_new_prompt/temp_zero/results_without_sentence_confidence/results_sentence_confidence_temp_zero.md")
    
    if args.experiment == "exp_new_prompt_sentence_confidence_temp_0.5":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_sentence_confidence_new_prompt/temp_0.5/results_without_sentence_confidence/whisper_corrected_transcriptions_sentence_confidence_temp_0.5.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_sentence_confidence_new_prompt/temp_0.5/results_without_sentence_confidence/results_sentence_confidence_temp_0.5.md")
        
    if args.experiment == "exp_new_prompt_lowest_word_confidence_temp_zero":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_lowest_word_confidence_new_prompt/temp_zero/results_without_lowest_word_confidence/whisper_corrected_transcriptions_lowest_word_confidence_temp_zero.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_lowest_word_confidence_new_prompt/temp_zero/results_without_lowest_word_confidence/results_lowest_word_confidence_temp_zero.md")
    
    if args.experiment == "exp_new_prompt_lowest_word_confidence_temp_0.5":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_lowest_word_confidence_new_prompt/temp_0.5/results_without_lowest_word_confidence/whisper_corrected_transcriptions_lowest_word_confidence_temp_0.5.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_lowest_word_confidence_new_prompt/temp_0.5/results_without_lowest_word_confidence/results_lowest_word_confidence_temp_0.5.md")
                    
    if args.experiment == "exp_prompt_original":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_original.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_original.md")
    
    if args.experiment == "exp_prompt_1":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_1.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_1.md")
        
    if args.experiment == "exp_prompt_2":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_2.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_2.md")
    
    if args.experiment == "exp_prompt_3":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_3.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_3.md")
        
    if args.experiment == "exp_prompt_4":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_4.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_4.md")
    
    if args.experiment == "exp_prompt_5":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_5.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_5.md")
        
    if args.experiment == "exp_prompt_6":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_6.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_6.md")
    
    if args.experiment == "exp_prompt_7":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_7.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_7.md")
    
    if args.experiment == "exp_prompt_8":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_8.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_8.md")
        
    
    if args.experiment == "exp_prompt_9":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_9.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_9.md")
    
    if args.experiment == "exp_prompt_10":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_10.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_10.md")
    
    if args.experiment == "exp_prompt_11":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_11.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_11.md")
    
    if args.experiment == "exp_prompt_12":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_12.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_12.md")
    
    if args.experiment == "exp_prompt_13":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH=os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/whisper_corrected_transcriptions_13.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_best_prompt/results_try_different_prompts/results_13.md")
                                    
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
