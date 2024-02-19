import os
from chat_gpt_asr.utils import ser
import json
from jiwer import cer, wer, compute_measures, RemovePunctuation
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
Root = os.getenv("ROOT_PATH")

l=os.path.join(Root,"results/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_lowest_word_confidence_new_prompts_medium/results_find_best_prompt_medium/corrected_transcriptions_lowest_word_confidence_prompt_1.json")

OUTPUT_FILE = os.path.join(Root,"results/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_lowest_word_confidence_new_prompts_medium/results_find_thresh_best_prompt_medium/results_thresh_lowest_word_confidence_prompt_1_medium.md")

output_file_wer = os.path.join(Root,"results/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_lowest_word_confidence_new_prompts_medium/results_find_thresh_best_prompt_medium/plots_lowest_word_confidence_medium/Wer_vs_lowest_word_confidence_prompt_1_medium.png")

output_file_cer = os.path.join(Root,"results/results_clean/results_medium/results_best_prompt_medium/results_GPT-3.5-Turbo_medium/gpt-3.5-turbo-1106/results_lowest_word_confidence_new_prompts_medium/results_find_thresh_best_prompt_medium/plots_lowest_word_confidence_medium/Cer_vs_lowest_word_confidence_prompt_1_medium.png")

with open(l , "r") as f:
    json_obj=f.read()
    data=json.loads(json_obj)
    

        
def evaluate_with_thresh(items, thresh):
    hyp_l, ref_l = [], []
   
    for item in items:
        
        if item["corrected_asr_transcription"] is None:
            continue 
       
        
        ref_transcription=RemovePunctuation()(item["reference_transcription"].lower())
        cor_transcription=RemovePunctuation()(item["corrected_asr_transcription"].lower())
        asr_transcription=RemovePunctuation()(item["asr_transcription"]["text"].lower())
        confidence = item["asr_transcription"]["confidence_score"] 
        
        
        ref_l.append(ref_transcription)
        
        # check confidence and append the suitable value to hyp_l       
        if confidence <= thresh:
            hyp_l.append(cor_transcription)
            
        else:
            hyp_l.append(asr_transcription)
            
    WER = wer(ref_l, hyp_l) * 100
    CER = cer(ref_l, hyp_l) * 100
    
    return WER, CER


thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
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
    plt.xlabel("lowest word confidence")
    plt.ylabel("Wer")
    plt.title("Wer vs lowest word confidence for GPT-3.5-Turbo (new prompt-1, medium, clean )")
    #plt.yticks([3,3.5,4,4.5,5,5.5])
    plt.savefig(output_file_wer)
    
    
    y_cer = [dictionary["cer"] for dictionary in l]
    plt.figure()
    plt.plot(x, y_cer, "-or", label="Cer")
    plt.xlabel("lowest word confidence")
    plt.ylabel("Cer")
    plt.title("Cer vs lowest word confidence for GPT-3.5-Turbo (new promt-1, medium, clean)")
    #plt.yticks([1,1.5,2,2.5])
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




