import spacy
from jiwer import compute_measures, RemovePunctuation
import pandas as pd
import json
import os

import argparse
from dotenv import load_dotenv

load_dotenv()
Root = os.getenv("ROOT_PATH")

SPACY_MODEL = "en_core_web_sm"
nlp = spacy.load(SPACY_MODEL)

def tokenizer_fn(texts):
    tokens = []
    for text in texts:
        text = RemovePunctuation()(text.lower().strip())
        tokens.append([t.text for t in nlp.tokenizer(text)])
    return tokens
def extract_words(measures):

    truths, hypotheses, ops = measures['truth_texts'], measures['hypothesis_texts'], measures['ops']
    n_samples = len(truths)
    words = [[] for i in range(n_samples)]
    pos = [[] for i in range(n_samples)]
    operations = [[] for i in range(n_samples)]
    
    for idx in range(n_samples):
        truth_doc = nlp(truths[idx])
        truth_text = [token.text for token in truth_doc]
        truth_pos = [token.pos_ for token in truth_doc]
        
        hypothesis_doc = nlp(hypotheses[idx])
        hypothesis_text = [token.text for token in hypothesis_doc]
        hypothesis_pos = [token.pos_ for token in hypothesis_doc]

        for op in ops[idx]:
            if op.type == "delete":
                words[idx].extend(truth_text[op.ref_start_idx:op.ref_end_idx])
                pos[idx].extend(truth_pos[op.ref_start_idx:op.ref_end_idx])
                operations[idx].extend(["delete" for _ in truth_pos[op.ref_start_idx:op.ref_end_idx]])
                
            elif op.type == "insert":
                words[idx].extend(hypothesis_text[op.hyp_start_idx:op.hyp_end_idx])
                pos[idx].extend(hypothesis_pos[op.hyp_start_idx:op.hyp_end_idx])
                operations[idx].extend(["insert" for _ in hypothesis_pos[op.hyp_start_idx:op.hyp_end_idx]])
                
            elif op.type == "substitute":
                # words[idx].extend(truth_text[op.ref_start_idx:op.ref_end_idx])
                # pos[idx].extend(truth_pos[op.ref_start_idx:op.ref_end_idx])
                words[idx].extend(hypothesis_text[op.hyp_start_idx:op.hyp_end_idx])
                pos[idx].extend(hypothesis_pos[op.hyp_start_idx:op.hyp_end_idx])
                operations[idx].extend(["substitute" for _ in hypothesis_pos[op.hyp_start_idx:op.hyp_end_idx]])
                
            elif op.type == "equal":
                # words[idx].extend(truth_text[op.ref_start_idx:op.ref_end_idx])
                # pos[idx].extend(truth_pos[op.ref_start_idx:op.ref_end_idx])
                words[idx].extend(hypothesis_text[op.hyp_start_idx:op.hyp_end_idx])
                pos[idx].extend(hypothesis_pos[op.hyp_start_idx:op.hyp_end_idx])
                operations[idx].extend(["equal" for _ in hypothesis_pos[op.hyp_start_idx:op.hyp_end_idx]])

    return words, pos, operations
       
def create_table(pos, ops):
    # Flatten the nested lists of pos and ops
    flat_pos = [pos_row for sublist in pos for pos_row in sublist]
    flat_ops = [ops_row for sublist in operations for ops_row in sublist]

    # Get unique parts of speech
    unique_pos = sorted(set(flat_pos))
    unique_ops = set(flat_ops)
    # Create a dictionary to store the cumulative sums for each operation and part of speech
    data = {op: {p: sum(1 for i, j in zip(flat_pos, flat_ops) if i == p and j == op) for p in unique_pos} for op in unique_ops}

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(data)

    return df.T


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ChatGPT ASR Correction")
    parser.add_argument("-d", "--dataset", choices=["librispeech"], default="librispeech", help="Select the dataset (librispeech)")
    parser.add_argument("-e","--experiment", choices = ["exp_without_sentence_confidence_tiny","exp_without_lowest_word_confidence_tiny",  "exp_without_average_word_confidence_tiny", "exp_without_sentence_confidence_GPT-4-Turbo_tiny","exp_without_lowest_word_confidence_GPT-4_tiny", "exp_certain_low_confidence_words_Thresh_0.55_tiny","exp_certain_low_confidence_words_Thresh_0.6_tiny","exp_certain_low_confidence_words_Thresh_0.65_tiny",
"exp_certain_low_confidence_words_Thresh_0.7_tiny","exp_certain_low_confidence_words_Thresh_0.75_tiny","exp_certain_low_confidence_words_Thresh_0.8_tiny",
"exp_certain_low_confidence_words_Thresh_0.85_tiny","exp_certain_low_confidence_words_Thresh_0.9_tiny","exp_certain_low_confidence_words_Thresh_0.95_tiny", "exp_certain_low_confidence_words_Thresh_0.6_GPT-4-Turbo_tiny", "exp_without_average_word_confidence_GPT-4-Turbo_tiny", "exp_sentence_confidence_GPT-3.5_medium", "exp_lowest_word_confidence_GPT-3.5_medium","exp_sentence_confidence_GPT-4_medium", "exp_sentence_confidence_GPT-3.5_large-v3", "exp_lowest_word_confidence_GPT-3.5_large-v3", "exp_sentence_confidence_GPT-4_large-v3", "exp_sentence_confidence_GPT-3.5_noisy_large-v3","exp_lowest_word_confidence_GPT-3.5_noisy_large-v3","exp_sentence_confidence_GPT-4_noisy_large-v3", "exp_lowest_word_confidence_GPT-4_noisy_large-v3","exp_lowest_word_confidence_GPT-4_medium","exp_lowest_word_confidence_GPT-4_large-v3", "exp_without_sentence_confidence_noisy_tiny","exp_without_sentence_confidence_GPT-4-Turbo_noisy_tiny","exp_without_lowest_word_confidence_noisy_tiny","exp_without_lowest_word_confidence_GPT-4_noisy_tiny", "exp_sentence_confidence_GPT-3.5_noisy_medium", "exp_lowest_word_confidence_GPT-3.5_noisy_medium", "exp_sentence_confidence_GPT-4_noisy_medium", "exp_lowest_word_confidence_GPT-4_noisy_medium"], help = "Select the experiment")
    args = parser.parse_args()

    if args.experiment == "exp_without_sentence_confidence_noisy_tiny":
        path=os.path.join(Root,"results/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json")
        OUTPUT_FILE = os.path.join(Root,"results/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/results_spacy_sentence_confidence_tiny.md")



    with open(path, "r") as f:
        data = json.load(f)

    transcriptions = [d["asr_transcription"]["text"] for d in data]
    corrected_transcriptions = [d["corrected_asr_transcription"] for d in data]
    
    assert all([type(r) == str for r in transcriptions]) == True, "transcriptions should be a list of str!"
    assert all([type(r) == str for r in corrected_transcriptions]) == True, "transcriptions should be a list of str!"
    
    # __main__
    meas = compute_measures(truth=transcriptions, hypothesis=corrected_transcriptions, 
                 truth_transform=tokenizer_fn, hypothesis_transform=tokenizer_fn)
    meas['truth_texts'] = transcriptions
    meas['hypothesis_texts'] = corrected_transcriptions
    words, pos, operations = extract_words(meas)
    
    df = create_table(pos, operations)
    #df = df.drop(["SYM", "PUNCT", "SPACE", "X"], axis=1) 
    df = (df.div(df.sum(axis=1), axis=0)*100).round(2)
    df.to_markdown(OUTPUT_FILE)

