from chat_gpt_asr.utils import remove_punctuations
import json
from jiwer import cer, wer
from tqdm import tqdm
import argparse

def ser(asr_transcription, corrected_asr_transcription, reference_transcription):

    asr_transcription = remove_punctuations(d["asr_transcription"]["text"].lower())
    
    corrected_asr_transcription = remove_punctuations(corrected_asr_transcription.lower())
    reference_transcription = remove_punctuations(reference_transcription.lower())

    SER = 100 - (corrected_asr_transcription == reference_transcription)*100
    SER_original = 100 - (asr_transcription == reference_transcription)*100
    
    
    return SER,SER_original


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ChatGPT ASR Correction")
    parser.add_argument("-d", "--dataset", choices=["librispeech", "dummy"], default="librispeech", help="Select the dataset (librispeech or dummy)")
    args = parser.parse_args()

    if args.dataset == "librispeech":
        with open("../../results/experiment2/whisper_corrected_transcriptions.json", "r") as f:
            json_obj=f.read()
            data=json.loads(json_obj)
        output_filename= "../../results/experiment2/results_whisper.md"    

    elif args.dataset == "dummy":
        with open("../../results/experiment2/results_dummy", "r") as f:
            json_obj=f.read()
            data=json.loads(json_obj)
        output_filename= "../../results/experiment2/results_dummy.md"    
  

    ref_l, hyp_l, hyp_l_original = [],  [], []
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
    
