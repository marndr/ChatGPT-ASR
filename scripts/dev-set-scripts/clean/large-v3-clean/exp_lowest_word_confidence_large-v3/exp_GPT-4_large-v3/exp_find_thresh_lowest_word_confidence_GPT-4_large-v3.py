import json
import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from jiwer import RemovePunctuation, cer, wer

load_dotenv()
Root = os.getenv("ROOT_PATH")

l = os.path.join(
    Root,
    "results/results-dev-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-4o_large-v3/without_present_confidence_chatgpt/results_without_lowest_word_confidence_large-v3/corrected_transcriptions_lowest_word_confidence_large-v3.json",
)

OUTPUT_FILE = os.path.join(
    Root,
    "results/results-dev-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-4o_large-v3/without_present_confidence_chatgpt/results_find_thresh_lowest_word_confidence_GPT-4o_large-v3/results_thresh_lowest_word_confidence_large-v3.json",
)

output_file_wer = os.path.join(
    Root,
    "results/results-dev-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-4o_large-v3/without_present_confidence_chatgpt/results_find_thresh_lowest_word_confidence_GPT-4o_large-v3/plots_lowest_word_confidence_GPT-4o_large-v3/Wer_vs_lowest_word_confidence_GPT-4o_plot_large-v3.png",
)
output_file_cer = os.path.join(
    Root,
    "results/results-dev-set/results_clean/results_large-v3/results_lowest_word_confidence_large-v3/results_GPT-4o_large-v3/without_present_confidence_chatgpt/results_find_thresh_lowest_word_confidence_GPT-4o_large-v3/plots_lowest_word_confidence_GPT-4o_large-v3/Cer_vs_lowest_word_confidence_GPT-4o_plot_large-v3.png",
)

with open(l) as f:
    json_obj = f.read()
    data = json.loads(json_obj)


def evaluate_with_thresh(items, thresh):
    hyp_l, ref_l = [], []

    for item in items:
        if item["corrected_asr_transcription"] is None:
            continue

        ref_transcription = RemovePunctuation()(item["reference_transcription"].lower())
        cor_transcription = RemovePunctuation()(
            item["corrected_asr_transcription"].lower()
        )
        asr_transcription = RemovePunctuation()(
            item["asr_transcription"]["text"].lower()
        )
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


thresholds = [
    0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1,
]
results = []


for thresh in thresholds:
    Wer, Cer = evaluate_with_thresh(data, thresh)
    results.append({"thresh": thresh, "wer": Wer, "cer": Cer})


def plot(l):
    x = [dictionary["thresh"] for dictionary in l]
    y_wer = [dictionary["wer"] for dictionary in l]

    plt.figure()
    plt.plot(x, y_wer, "-ob", label="Wer")
    plt.xlabel("lowest word confidence")
    plt.ylabel("Wer")
    plt.title("Wer vs lowest word confidence for GPT-4o (clean, large-v3)")
    # plt.yticks([6,6.5,7,7.5,8,8.5,9])
    # plt.show()
    plt.savefig(output_file_wer)

    y_cer = [dictionary["cer"] for dictionary in l]
    plt.figure()
    plt.plot(x, y_cer, "-or", label="Cer")
    plt.xlabel("lowest word confidence")
    plt.ylabel("Cer")
    plt.title("Cer vs lowest word confidence for GPT-4o (clean, large-v3)")
    # plt.yticks([2.5,3,3.5,4,4.5,5])
    # plt.show()
    plt.savefig(output_file_cer)


plot(results)


# Sort results based on WER
results.sort(key=lambda x: x["wer"])

# Print results
for result in results:
    print(
        f'Threshold: {result["thresh"]:.02f}, {result["wer"]:.02f}, {result["cer"]:.02f}'
    )


# Write the JSON data to a file
with open(OUTPUT_FILE, "w") as f:
    json_obj = json.dumps(results, indent=2)
    f.write(json_obj)
