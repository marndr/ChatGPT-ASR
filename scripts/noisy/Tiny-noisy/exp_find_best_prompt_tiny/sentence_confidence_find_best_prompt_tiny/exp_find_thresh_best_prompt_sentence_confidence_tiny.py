import os
from chat_gpt_asr.utils import remove_punctuations, ser
import json
import jiwer 
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
Root = os.getenv("ROOT_PATH")

l=os.path.join(Root,"results/results_noisy/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_sentence_confidence_new_prompts_tiny/results_find_best_prompt_tiny/corrected_transcriptions_sentence_confidence_prompt_1.json")

OUTPUT_FILE = os.path.join(Root,"results/results_noisy/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_sentence_confidence_new_prompts_tiny/results_find_thresh_best_prompt_tiny/results_thresh_sentence_confidence_prompt_1_tiny.md")

output_file_wer = os.path.join(Root,"results/results_noisy/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_sentence_confidence_new_prompts_tiny/results_find_thresh_best_prompt_tiny/plots_sentence_confidence_tiny/Wer_vs_sentence_confidence_prompt_1_tiny.png")

output_file_cer = os.path.join(Root,"results/results_noisy/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_sentence_confidence_new_prompts_tiny/results_find_thresh_best_prompt_tiny/plots_sentence_confidence_tiny/Cer_vs_sentence_confidence_prompt_1_tiny.png")

with open(l , "r") as f:
    json_obj=f.read()
    data=json.loads(json_obj)
    

        
def evaluate_with_thresh(items, thresh):
    hyp_l, ref_l = [], []
   
    for item in items:
        
        if item["corrected_asr_transcription"] is None:
            continue 
           

        asr_transcription = item["asr_transcription"]["text"]
        confidence = item["asr_transcription"]["confidence_score"] 
        ref_transcription = item["reference_transcription"]  
        cor_transcription = item ["corrected_asr_transcription"]
        
        # remove punc and lower
        asr_transcription = remove_punctuations(item["asr_transcription"]["text"].lower())
        ref_transcription = remove_punctuations(item["reference_transcription"].lower())
        cor_transcription = remove_punctuations(item["corrected_asr_transcription"].lower())
        
        ref_l.append(ref_transcription)
        
        # check confidence and append the suitable value to hyp_l       
        if confidence < thresh:
            hyp_l.append(cor_transcription)
            
        else:
            hyp_l.append(asr_transcription)
            
    wer = jiwer.wer(ref_l, hyp_l) * 100
    cer = jiwer.cer(ref_l, hyp_l) * 100
    
    return wer, cer


thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
results = []


for thresh in thresholds:
    Wer, Cer = evaluate_with_thresh(data, thresh)
    results.append({
        "thresh":thresh,
        "wer": Wer,
        "cer": Cer   
    })


def plot(l):

    x = [dictionary["thresh"] for dictionary in l]
    y_wer = [dictionary["wer"] for dictionary in l]
    
    plt.figure()
    plt.plot(x, y_wer, "-ob", label="Wer")
    plt.xlabel("sentence confidence")
    plt.ylabel("Wer")
    plt.title("Wer vs sentence confidence for GPT-3.5-Turbo (new prompt-1, tiny, noisy)")
    #plt.yticks([6,6.5,7,7.5,8,8.5,9])
    plt.savefig(output_file_wer)
    
    
    y_cer = [dictionary["cer"] for dictionary in l]
    plt.figure()
    plt.plot(x, y_cer, "-or", label="Cer")
    plt.xlabel("sentence confidence")
    plt.ylabel("Cer")
    plt.title("Cer vs sentence confidence for GPT-3.5-Turbo (new prompt-1, tiny, noisy)")
    #plt.yticks([2.5,3,3.5,4,4.5,5])
    #plt.show()
    plt.savefig(output_file_cer)
   
   
plot(results)


# Sort results based on WER
results.sort(key=lambda x: x["wer"])

# Print results
for result in results:
    
    print(f'Threshold: {result["thresh"]:.02f}, {result["wer"]:.02f}, {result["cer"]:.02f}')


# Write the JSON data to a file
with open(OUTPUT_FILE, 'w') as f:
    json_obj = json.dumps(results , indent = 2)
    f.write(json_obj)




