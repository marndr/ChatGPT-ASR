import argparse
import json
import os

# from chat_gpt_asr.utils import remove_punctuations, ser
from chat_gpt_asr.utils import ser
from dotenv import load_dotenv
from jiwer import RemovePunctuation, cer, compute_measures, wer
from tqdm import tqdm

load_dotenv()
Root = os.getenv("ROOT_PATH")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT ASR Correction")
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["librispeech"],
        default="librispeech",
        help="Select the dataset (librispeech)",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        choices=[
            "exp_new_prompt_sentence_confidence_1_tiny",
            "exp_new_prompt_sentence_confidence_2_tiny",
            "exp_new_prompt_sentence_confidence_3_tiny",
            "exp_new_prompt_sentence_confidence_4_tiny",
            "exp_new_prompt_lowest_word_confidence_2_tiny_1106",
            "exp_new_prompt_lowest_word_confidence_4_tiny",
            "exp_GPT_4_new_prompt_sentence_confidence_4_tiny",
            "exp_GPT_4_new_prompt_lowest_word_confidence_4_tiny_1106",
            "exp_new_prompt_lowest_word_confidence_2_tiny_0125",
            "exp_GPT_4_new_prompt_lowest_word_confidence_4_tiny_0125",
        ],
    )
    args = parser.parse_args()

    if args.experiment == "exp_new_prompt_sentence_confidence_1_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_tiny/results_find_best_prompt_tiny/corrected_transcriptions_sentence_confidence_prompt_1.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_tiny/results_find_best_prompt_tiny/results_overall_sentence_confidence_prompt_1_tiny.md",
        )

    if args.experiment == "exp_new_prompt_sentence_confidence_2_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_tiny/results_find_best_prompt_tiny/corrected_transcriptions_sentence_confidence_prompt_2.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_tiny/results_find_best_prompt_tiny/results_overall_sentence_confidence_prompt_2_tiny.md",
        )

    if args.experiment == "exp_new_prompt_sentence_confidence_3_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_tiny/results_find_best_prompt_tiny/corrected_transcriptions_sentence_confidence_prompt_3.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_tiny/results_find_best_prompt_tiny/results_overall_sentence_confidence_prompt_3_tiny.md",
        )

    if args.experiment == "exp_new_prompt_sentence_confidence_4_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_tiny/results_find_best_prompt_tiny/corrected_transcriptions_sentence_confidence_prompt_4.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_sentence_confidence_new_prompts_tiny/results_find_best_prompt_tiny/results_overall_sentence_confidence_prompt_4_tiny.md",
        )

    if args.experiment == "exp_new_prompt_lowest_word_confidence_2_tiny_1106":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_lowest_word_confidence_new_prompts_tiny/results_find_best_prompt_tiny/corrected_transcriptions_lowest_word_confidence_prompt_2.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_lowest_word_confidence_new_prompts_tiny/results_find_best_prompt_tiny/results_overall_lowest_word_confidence_prompt_2_tiny.md",
        )

    if args.experiment == "exp_new_prompt_lowest_word_confidence_4_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_lowest_word_confidence_new_prompts_tiny/results_find_best_prompt_tiny/corrected_transcriptions_lowest_word_confidence_prompt_4.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_lowest_word_confidence_new_prompts_tiny/results_find_best_prompt_tiny/results_overall_lowest_word_confidence_prompt_4_tiny.md",
        )

    if args.experiment == "exp_GPT_4_new_prompt_sentence_confidence_4_tiny":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_sentence_confidence_new_prompts_tiny/results_find_best_prompt_tiny/corrected_transcriptions_sentence_confidence_GPT-4-Turbo_prompt_4.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_sentence_confidence_new_prompts_tiny/results_find_best_prompt_tiny/results_overall_sentence_confidence_prompt_4_tiny.md",
        )

    if args.experiment == "exp_GPT_4_new_prompt_lowest_word_confidence_4_tiny_1106":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_lowest_word_confidence_new_prompts_tiny/results_find_best_prompt_tiny/corrected_transcriptions_lowest_word_confidence_GPT-4-Turbo_prompt_4_tiny.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-4-Turbo_tiny/gpt-4-1106-preview/results_lowest_word_confidence_new_prompts_tiny/results_find_best_prompt_tiny/results_overall_lowest_word_confidence_prompt_4_tiny.md",
        )

    if args.experiment == "exp_new_prompt_lowest_word_confidence_2_tiny_0125":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_lowest_word_confidence_new_prompts_tiny/results_find_best_prompt_tiny/corrected_transcriptions_lowest_word_confidence_prompt_2.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_lowest_word_confidence_new_prompts_tiny/results_find_best_prompt_tiny/results_overall_lowest_word_confidence_prompt_2_tiny.md",
        )

    if args.experiment == "exp_GPT_4_new_prompt_lowest_word_confidence_4_tiny_0125":
        CORRECTED_TRANSCRIPTIONS_LIBRISPEECH = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_lowest_word_confidence_new_prompts_tiny/results_find_best_prompt_tiny/corrected_transcriptions_lowest_word_confidence_prompt_2.json",
        )
        OUTPUT_FILE = os.path.join(
            Root,
            "results/results-dev-set/results_clean/results_tiny/results_best_prompt_tiny/results_GPT-4-Turbo_tiny/gpt-4-0125-preview/results_lowest_word_confidence_new_prompts_tiny/results_find_best_prompt_tiny/results_overall_lowest_word_confidence_prompt_2_tiny.md",
        )

    if args.dataset == "librispeech":
        with open(CORRECTED_TRANSCRIPTIONS_LIBRISPEECH) as f:
            json_obj = f.read()
            data = json.loads(json_obj)
        output_filename = OUTPUT_FILE

    ref_l, hyp_l = [], []
    ser_corrected_chatgpt, ser_original = 0.0, 0.0
    count = 0

    hyp_l_original = []
    ref_l_orig = []

    for d in tqdm(data):
        if d["corrected_asr_transcription"] is None:
            ref_l_orig.append(RemovePunctuation()(d["reference_transcription"].lower()))
            hyp_l_original.append(
                RemovePunctuation()(d["asr_transcription"]["text"].lower())
            )

        else:
            ref = RemovePunctuation()(d["reference_transcription"].lower())
            hyp = RemovePunctuation()(d["corrected_asr_transcription"].lower())
            hyp_original = RemovePunctuation()(d["asr_transcription"]["text"].lower())
            ref_l.append(ref)
            hyp_l.append(hyp)
            hyp_l_original.append(hyp_original)
            ref_l_orig.append(ref)
            SER_chatgpt_, SER_original_ = ser(hyp_original, hyp, ref)
            count += 1
            ser_corrected_chatgpt += SER_chatgpt_
            ser_original += SER_original_

    wer_original = wer(ref_l_orig, hyp_l_original) * 100
    wer_corrected_chatgpt = wer(ref_l, hyp_l) * 100

    cer_corrected_chatgpt = cer(ref_l, hyp_l) * 100
    cer_original = cer(ref_l_orig, hyp_l_original) * 100

    measures = compute_measures(ref_l, hyp_l)
    measures_original = compute_measures(ref_l_orig, hyp_l_original)

    ser_corrected_chatgpt /= count
    ser_original /= count

    print(f"Number of audio files:  {count:}\n")
    print(f"WER_original is:                {wer_original:.04f}\n")
    print(f"WER_corrected_chatgpt is:       {wer_corrected_chatgpt:.04f}\n")
    print(f"CER_original is:                {cer_original:.04f}\n")
    print(f"CER_corrected_chatgpt is:       {cer_corrected_chatgpt:.04f}\n")
    print(f"SER_original is:                {ser_original:.04f}\n")
    print(f"SER_corrected_chatgpt is:       {ser_corrected_chatgpt:.04f}\n")
    print(
        f"measures for substitution: {measures['substitutions']}, insertions: {measures['insertions']}, deletions: {measures['deletions']}, wer: {measures['wer']}"
    )
    print(
        f"measures_original for substitution: {measures_original['substitutions']}, insertions: {measures_original['insertions']}, deletions: {measures_original['deletions']}, wer: {measures_original['wer']}"
    )

    with open(output_filename, "w") as f:
        f.write(f"""
        Number of evaluated audio files:  {count:}
        WER_original:                   {wer_original:.04f}%
        WER_corrected_chatgpt:          {wer_corrected_chatgpt:.04f}%
        CER_original:                   {cer_original:.04f}%
        CER_corrected_chatgpt:          {cer_corrected_chatgpt:.04f}%
        SER_original:                   {ser_original:.04f}%
        SER_corrected_chatgpt:          {ser_corrected_chatgpt:.04f}%
        measures for substitution: {measures['substitutions']}, insertions: {measures['insertions']}, deletions: {measures['deletions']}
        measures_original for substitution: {measures_original['substitutions']}, insertions: {measures_original['insertions']}, deletions: {measures_original['deletions']}
        ---
    """)
