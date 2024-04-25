import json

import pandas as pd
from chat_gpt_asr.alignment import align3
from jiwer import RemovePunctuation

path = "/home/mnaderi/Documents/thesis/chat-gpt-asr/results/results-dev-set/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json"  # noqa: E501

with open(path) as f:
    data = json.load(f)

    transcriptions = [
        RemovePunctuation()(d["asr_transcription"]["text"]).lower().strip()
        for d in data
    ]
    reference_transcriptions = [
        RemovePunctuation()(d["reference_transcription"]).lower().strip() for d in data
    ]
    corrected_transcriptions = [
        RemovePunctuation()(d["corrected_asr_transcription"]).lower().strip()
        for d in data
    ]


def identify_edit_type(ref, asr, llm):
    edit_types = []
    edits = {"A": [], "B": [], "C": [], "D": []}
    rr, aa, ll = align3(ref, asr, llm)
    if len(rr) == len(aa) == len(ll):
        for r, a, l in zip(rr, aa, ll, strict=False):  # noqa: E741
            if a == l == r:
                edit_types.append("C")  # left it correct
                edits["C"].append((a, l, r))
            elif a != l and l == r:
                edit_types.append("A")  # improve it
                edits["A"].append((a, l, r))
            elif a != r and l != r:
                edit_types.append("D")  # left it incorrect
                edits["D"].append((a, l, r))
            elif a == r and l != r:
                edit_types.append("B")  # introducing an error
                edits["B"].append((a, l, r))
    else:
        raise Exception
    return edit_types, edits


data = []
for i, (ref, asr, llm) in enumerate(
    zip(
        reference_transcriptions, transcriptions, corrected_transcriptions, strict=False
    )
):
    # if i > 10:
    #     break
    edit_types, edits = identify_edit_type(ref, asr, llm)
    word_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for edit_type in edit_types:
        word_counts[edit_type] += 1

    # Create a DataFrame to store word counts by edit type
    data.append(word_counts)
    # df_edit_types = pd.DataFrame(word_counts.items(), columns=['Type', 'Count'])

    # Display the DataFrame
    # print(f"asr: {asr} len:{len(asr)}")
    # print(f"llm: {llm} len: {len(llm)}")
    # print(f"ref: {ref} len: {len(ref)}")
    # print('edits: ', edits)
    # print(i, df_edit_types , "\n")

df = pd.DataFrame(data)
df.to_csv("data.csv")
