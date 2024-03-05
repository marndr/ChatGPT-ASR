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

l=os.path.join(Root,"results/results-test-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_without_sentence_confidence_large-v3/corrected_transcriptions_sentence_confidence_large-v3.json")

OUTPUT_FILE = os.path.join(Root,"results/results-test-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_find_thresh_sentence_confidence_large-v3/results_thresh_sentence_confidence_large-v3.md")

output_file_wer = os.path.join(Root,"results/results-test-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_find_thresh_sentence_confidence_large-v3/plots_sentence_confidence_large-v3/Wer_vs_sentence_confidence_plot_large-v3.png")

output_file_cer = os.path.join(Root,"results/results-test-set/results_clean/results_large-v3/results_sentence_confidence_large-v3/results_GPT-3.5-Turbo_large-v3/gpt-3.5-turbo-0125/results_find_thresh_sentence_confidence_large-v3/plots_sentence_confidence_large-v3/Cer_vs_sentence_confidence_plot_large-v3.png")

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



thresholds = [0.95]
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
    plt.title("Wer vs sentence confidence for GPT-3.5-Turbo(clean, large-v3, test-set)")
    #plt.yticks([3.5,4,4.5,5,5.5])
    #plt.show()
    plt.savefig(output_file_wer)
    
    
    y_cer = [dictionary["cer"] for dictionary in l]
    plt.figure()
    plt.plot(x, y_cer, "-or", label="Cer")
    plt.xlabel("sentence confidence")
    plt.ylabel("Cer")
    plt.title("Cer vs sentence confidence for GPT-3.5-Turbo(clean, large-v3, test-set)")
    #plt.yticks([1,1.5,2,2.5])
    #plt.show()
    plt.savefig(output_file_cer)
   
   
#plot(results)


# Sort results based on WER
results.sort(key=lambda x: x["wer"])

# Print results
for result in results:
    
    print(f'Threshold: {result["thresh"]:.02f}, {result["wer"]:.02f}, {result["cer"]:.02f}')


# Write the JSON data to a file
with open(OUTPUT_FILE, 'w') as f:
    json_obj = json.dumps(results , indent = 2)
    f.write(json_obj)




