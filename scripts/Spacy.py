import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import spacy
from dotenv import load_dotenv
from jiwer import compute_measures

load_dotenv()
Root = os.getenv("ROOT_PATH")

SPACY_MODEL = "en_core_web_sm"
nlp = spacy.load(SPACY_MODEL)


def wrapper(remove_punct=True, return_texts=False, return_pos=False):
    def tokenizer_fn(texts):
        tokens = []
        texts_new = []
        poses = []
        for text in texts:
            if remove_punct:
                # remove punctuation
                text = re.sub(r"[\s]+", " ", text).strip().lower()
                tokens.append(
                    [token.text for token in nlp(text) if token.pos_ != "PUNCT"]
                )
                poses.append(
                    [token.pos_ for token in nlp(text) if token.pos_ != "PUNCT"]
                )
                text = "".join(
                    [
                        token.text_with_ws if token.pos_ != "PUNCT" else " "
                        for token in nlp(text)
                    ]
                )
                text = re.sub(r"[\s]+", " ", text).strip().lower()
                # text = RemovePunctuation()(text.lower().strip())
                texts_new.append(text)
            else:
                text = re.sub(r"[\s]+", " ", text).strip().lower()
                tokens.append([token.text for token in nlp(text)])
                poses.append([token.pos_ for token in nlp(text)])
                text = "".join([token.text_with_ws for token in nlp(text)])
                text = re.sub(r"[\s]+", " ", text).strip().lower()
                texts_new.append(text)

        if not return_texts and not return_pos:
            return tokens

        out = [tokens, None, None]
        if return_texts:
            out[1] = texts_new
        if return_pos:
            out[2] = poses
        return out

    return tokenizer_fn


# extract the tokens and their parts of speech
def extract_words(measures):
    truths_pos, hypotheses_pos, ops, truths, hypotheses = (
        measures["truth_pos"],
        measures["hypothesis_pos"],
        measures["ops"],
        measures["truth"],
        measures["hypothesis"],
    )
    n_samples = len(truths)
    words = [[] for i in range(n_samples)]
    pos = [[] for i in range(n_samples)]
    operations = [[] for i in range(n_samples)]

    for idx in range(n_samples):
        truth_pos = truths_pos[idx]
        truth = truths[idx]

        hypothesis = hypotheses[idx]
        hypothesis_pos = hypotheses_pos[idx]

        for op in ops[idx]:
            if op.type == "delete":
                words[idx].extend(truth[op.ref_start_idx : op.ref_end_idx])
                pos[idx].extend(truth_pos[op.ref_start_idx : op.ref_end_idx])
                operations[idx].extend(
                    ["delete" for _ in truth_pos[op.ref_start_idx : op.ref_end_idx]]
                )

            elif op.type == "insert":
                words[idx].extend(hypothesis[op.hyp_start_idx : op.hyp_end_idx])
                pos[idx].extend(hypothesis_pos[op.hyp_start_idx : op.hyp_end_idx])
                operations[idx].extend(
                    [
                        "insert"
                        for _ in hypothesis_pos[op.hyp_start_idx : op.hyp_end_idx]
                    ]
                )

            elif op.type == "substitute":
                # words[idx].extend(truth_text[op.ref_start_idx:op.ref_end_idx])
                # pos[idx].extend(truth_pos[op.ref_start_idx:op.ref_end_idx])
                words[idx].extend(hypothesis[op.hyp_start_idx : op.hyp_end_idx])
                pos[idx].extend(hypothesis_pos[op.hyp_start_idx : op.hyp_end_idx])
                operations[idx].extend(
                    [
                        "substitute"
                        for _ in hypothesis_pos[op.hyp_start_idx : op.hyp_end_idx]
                    ]
                )

            elif op.type == "equal":
                # words[idx].extend(truth_text[op.ref_start_idx:op.ref_end_idx])
                # pos[idx].extend(truth_pos[op.ref_start_idx:op.ref_end_idx])
                words[idx].extend(hypothesis[op.hyp_start_idx : op.hyp_end_idx])
                pos[idx].extend(hypothesis_pos[op.hyp_start_idx : op.hyp_end_idx])
                operations[idx].extend(
                    ["equal" for _ in hypothesis_pos[op.hyp_start_idx : op.hyp_end_idx]]
                )

    return words, pos, operations


# create table where rows are operations (delete, substitute, insert, equal)
# and columns are part of speech
def create_table(pos, ops):
    # Flatten the nested lists of pos and ops
    flat_pos = [pos_row for sublist in pos for pos_row in sublist]
    flat_ops = [ops_row for sublist in operations for ops_row in sublist]

    # Get unique parts of speech
    unique_pos = sorted(set(flat_pos))
    unique_ops = set(flat_ops)
    # Create a dictionary to store the cumulative sums for each operation and part of speech
    data = {
        op: {
            p: sum(1 for i, j in zip(flat_pos, flat_ops) if i == p and j == op)
            for p in unique_pos
        }
        for op in unique_ops
    }

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(data)

    return df.T


# create table where rows are the data examples and columns are
# combinations of operations (delete, substitute, insert, equal) and pos (for example, VERB, NOUN...)
def create_table_wide(pos, operations):
    # Flatten the nested lists of pos and ops
    flat_pos = [pos_row for sublist in pos for pos_row in sublist]
    flat_ops = [ops_row for sublist in operations for ops_row in sublist]

    # Get unique parts of speech
    unique_pos = sorted(set(flat_pos))
    unique_ops = set(flat_ops)
    unique_ops.difference_update({"equal"})

    # Create an empty DataFrame
    df = pd.DataFrame(
        index=range(len(pos)),
        columns=[(op, pos_) for op in unique_ops for pos_ in unique_pos],
    )

    # Iterate over the examples, counting occurrences of each unique combination
    for i, (p, o) in enumerate(zip(pos, operations)):
        for pos_, op in zip(p, o):
            if (op, pos_) in df.columns:
                df.at[i, (op, pos_)] = (
                    df.at[i, (op, pos_)] + 1 if not pd.isna(df.at[i, (op, pos_)]) else 1
                )

    # Fill missing values with zeros
    df.fillna(0, inplace=True)
    return df


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
            "exp_without_sentence_confidence_tiny",
            "exp_without_lowest_word_confidence_tiny",
            "exp_without_average_word_confidence_tiny",
            "exp_without_sentence_confidence_GPT-4-Turbo_tiny",
            "exp_without_lowest_word_confidence_GPT-4_tiny",
            "exp_without_average_word_confidence_GPT-4-Turbo_tiny",
            "exp_sentence_confidence_GPT-3.5_medium",
            "exp_lowest_word_confidence_GPT-3.5_medium",
            "exp_sentence_confidence_GPT-4_medium",
            "exp_sentence_confidence_GPT-3.5_large-v3",
            "exp_lowest_word_confidence_GPT-3.5_large-v3",
            "exp_sentence_confidence_GPT-4_large-v3",
            "exp_sentence_confidence_GPT-3.5_noisy_large-v3",
            "exp_lowest_word_confidence_GPT-3.5_noisy_large-v3",
            "exp_sentence_confidence_GPT-4_noisy_large-v3",
            "exp_lowest_word_confidence_GPT-4_noisy_large-v3",
            "exp_lowest_word_confidence_GPT-4_medium",
            "exp_lowest_word_confidence_GPT-4_large-v3",
            "exp_without_sentence_confidence_noisy_tiny",
            "exp_without_sentence_confidence_GPT-4-Turbo_noisy_tiny",
            "exp_without_lowest_word_confidence_noisy_tiny",
            "exp_without_lowest_word_confidence_GPT-4_noisy_tiny",
            "exp_sentence_confidence_GPT-3.5_noisy_medium",
            "exp_lowest_word_confidence_GPT-3.5_noisy_medium",
            "exp_sentence_confidence_GPT-4_noisy_medium",
            "exp_lowest_word_confidence_GPT-4_noisy_medium",
        ],
        help="Select the experiment",
    )
    args = parser.parse_args()

    if args.experiment == "exp_without_sentence_confidence_noisy_tiny":
        path = os.path.join(
            Root,
            "results/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json",
        )
        OUTPUT_FILE_1 = os.path.join(
            Root,
            "results/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/results_table_spacy_sentence_confidence_tiny.md",
        )
        # OUTPUT_FILE_2 = os.path.join(Root,"results/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/results_wide_table_spacy_sentence_confidence_tiny.md")

        OUTPUT_FILE_3 = os.path.join(
            Root,
            "results/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/results_groupby_improved_table_spacy_sentence_confidence_tiny.md",
        )
        OUTPUT_FILE_4 = os.path.join(
            Root,
            "results/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/results_visualization_spacy_sentence_confidence_tiny.png",
        )
        OUTPUT_FILE_5 = os.path.join(
            Root,
            "results/results_noisy/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-0125/results_without_sentence_confidence_tiny/results_groupby_size_spacy_sentence_confidence_tiny.md",
        )

    with open(path) as f:
        data = json.load(f)

    transcriptions = [d["asr_transcription"]["text"] for d in data]
    reference_transcriptions = [d["reference_transcription"] for d in data]
    corrected_transcriptions = [d["corrected_asr_transcription"] for d in data]

    assert all(
        [isinstance(r, str) for r in transcriptions]
    ), "transcriptions should be a list of str!"
    assert all(
        [isinstance(r, str) for r in reference_transcriptions]
    ), "transcriptions should be a list of str!"
    assert all(
        [isinstance(r, str) for r in corrected_transcriptions]
    ), "transcriptions should be a list of str!"

    # __main__
    meas = compute_measures(
        transcriptions,
        corrected_transcriptions,
        truth_transform=wrapper(),
        hypothesis_transform=wrapper(),
    )
    trans_tokens, trans_texts, trans_poses = wrapper(
        return_texts=True, return_pos=True
    )(transcriptions)
    c_trans_tokens, c_trans_texts, c_trans_poses = wrapper(
        return_texts=True, return_pos=True
    )(corrected_transcriptions)
    meas["truth_pos"] = trans_poses
    meas["hypothesis_pos"] = c_trans_poses

    # extract changed words and poses
    words, pos, operations = extract_words(meas)

    # create table
    df = create_table(pos, operations)

    # convert to percentage
    df = (df.div(df.sum(axis=1), axis=0) * 100).round(2)

    # sort columns based on the sum of ops delete, substitute, and insert
    k = df.shape[1] - 1
    selected_ops = ["delete", "substitute", "insert"]
    sorted_sum_of_selected_ops = df.loc[selected_ops].sum().sort_values(ascending=False)
    df[sorted_sum_of_selected_ops.index[:k]].to_markdown(OUTPUT_FILE_1)

    # create a wide table
    df_wide = create_table_wide(pos, operations)
    wer_l = []
    for idx in df_wide.index:
        # calculate WER for reference vs. corrected transcription
        wer1 = compute_measures(
            reference_transcriptions[idx],
            corrected_transcriptions[idx],
            truth_transform=wrapper(),
            hypothesis_transform=wrapper(),
        )["wer"]
        # calculate WER for reference vs. original transcription
        wer2 = compute_measures(
            reference_transcriptions[idx],
            transcriptions[idx],
            truth_transform=wrapper(),
            hypothesis_transform=wrapper(),
        )["wer"]
        # check if corrected transcription has lower WER (improvement)
        wer_l.append([wer1, wer2, wer1 < wer2])

    # df_wide_percent = (df_wide.div(df_wide.sum(axis=1), axis=0)*100).round(2)
    df_wide["wer_reference_and_corrected_transcription"] = [_[0] for _ in wer_l]
    df_wide["wer_reference_and_transcription"] = [_[1] for _ in wer_l]
    df_wide["improved"] = [_[2] for _ in wer_l]

    # grouping data by whether or not chatgpt improved transcription (reduced WER)
    # and count the number of improved and not improved examples
    df_wide.groupby("improved").size().to_markdown(OUTPUT_FILE_5)

    # sorting based on selected columns (all columns except for wer_reference_and_corrected_transcription,
    # wer_reference_and_transcription, and improved)
    selected_columns = list(
        set(df_wide.columns).difference(
            {
                "wer_reference_and_corrected_transcription",
                "wer_reference_and_transcription",
                "improved",
            }
        )
    )

    # take the average value of changes (for example insert_VERB, ...)
    # for both improved and not improved cases and take the top 5 improtant changes
    tt = df_wide.groupby("improved").mean()
    # Calculate the sum of each column in the subset
    subset_sum = tt[selected_columns].sum()
    # Sort the summed values in descending order
    sorted_subset_sum = subset_sum.sort_values(ascending=False)
    # Define the value of k
    k = 5  # Change this to your desired value
    # Select the top k columns
    top_k_columns = sorted_subset_sum.head(k)

    tt1 = tt[
        top_k_columns.index.tolist()
        + [
            "wer_reference_and_corrected_transcription",
            "wer_reference_and_transcription",
        ]
    ]
    tt1.T.to_markdown(OUTPUT_FILE_3)

    # draw top 5 improtant changes that chatgpt has made
    x = [f"{i}_{j}" for i, j in top_k_columns.index.tolist()]
    plt.figure(figsize=(6, 5), constrained_layout=True)
    plt.plot(
        x,
        tt1.loc[False, top_k_columns.index.tolist()].values,
        "r-o",
        label="Not improved",
    )
    plt.plot(
        x, tt1.loc[True, top_k_columns.index.tolist()].values, "g-o", label="Improved"
    )
    plt.xticks(rotation=45)
    plt.ylabel("Avg value")
    plt.title(f"top {k} ChatGPT changes")
    plt.legend()
    plt.savefig(OUTPUT_FILE_4, bbox_inches="tight", dpi=200)
