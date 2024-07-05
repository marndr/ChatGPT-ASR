# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maryam Naderi  <maryam.naderi@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-chatgpt-asr

"""
This module provides various functions for text processing and ASR evaluation.
It includes functions for reading dummy transcriptions, reading transcriptions from
Librispeech dataset,
removing punctuations from text, computing SER, and calculating confidence scores
at different levels (sentence-level, lowest word-level, average word-level, and
word-level).

Functions:
- read_dummy_transcriptions(): Returns a list of dummy transcriptions for testing
  purposes.
- read_librispeech_transcriptions(root_folder="."): Reads audio filenames and their
  corresponding transcriptions
  from the Librispeech dataset located in the specified root folder.
- remove_punctuations(s): Removes punctuations from the input strings.
- ser(asr_transcription, corrected_asr_transcription, reference_transcription):
  Computes SER between corrected ASR transcription and reference transcription.
- confidence_score_sentence_level(trans, confidence=True): Computes confidence
  score at sentence level.
- confidence_score_lowest_word_level(trans, confidence=True): Computes confidence
  score at lowest word level.
- confidence_score_average_word_level(trans, confidence=True): Computes average
  confidence score at word level.
- confidence_score_word_level(trans, confidence=True): Computes confidence score for
  each word in transcription.
"""

import os
import re


def read_dummy_transcriptions():
    """
    Returns a list of dummy transcriptions for testing purposes.

    Each entry in the list contains:
    - 'asr_transcription': ASR transcription with errors.
    - 'reference_transcription': Reference transcription without errors.
    """
    return [
        {
            "asr_transcription": "I prefer see over coffee.",
            "reference_transcription": "I prefer tea over coffee.",
        },
        {
            "asr_transcription": "I need to catch a drain to London.",
            "reference_transcription": "I need to catch a train to London.",
        },
        {
            "asr_transcription": "I'll beat you at the coffee shop.",
            "reference_transcription": "I'll meet you at the coffee shop.",
        },
        {
            "asr_transcription": "He's a professional sheaf.",
            "reference_transcription": "He's a professional chef.",
        },
        {
            "asr_transcription": "The leather is nice today.",
            "reference_transcription": "The weather is nice today.",
        },
        {
            "asr_transcription": "Please pass me the vault.",
            "reference_transcription": "Please pass me the salt.",
        },
        {
            "asr_transcription": "I'll be their in five minuets.",
            "reference_transcription": "I'll be there in five minutes.",
        },
        {
            "asr_transcription": "I'm going to the gross restore.",
            "reference_transcription": "I'm going to the grocery store.",
        },
        {
            "asr_transcription": "I won't to go two the beech.",
            "reference_transcription": "I want to go to the beach.",
        },
        {
            "asr_transcription": "The son is shining brightly in the sky.",
            "reference_transcription": "The sun is shining brightly in the sky.",
        },
        {
            "asr_transcription": "The son sets at the beech are always stunting.",
            "reference_transcription": "The sun sets at the beach are always stunning.",
        },
    ]


# Function to read audio filenames and their corresponding transcriptions
def read_librispeech_transcriptions(root_folder="."):
    """
    Reads audio filenames and their corresponding transcriptions from the Librispeech
    dataset.

    Args:
    - root_folder (str): Root folder path where the Librispeech dataset is located.

    Returns:
    - data (dict): A dictionary where keys are absolute paths to audio files and
      values are transcriptions.
    """
    data = {}

    # Regular expression to match audio filenames
    # audio_file_pattern = re.compile(r"(\d+-\d+-\d+)\.flac")
    audio_file_pattern_no_ending = re.compile(r"(\d+-\d+-\d+)")

    for root, dirs, files in os.walk(root_folder):
        trans_files = [f for f in files if f.endswith(".trans.txt")]
        for trans_file in trans_files:
            trans_file_path = os.path.join(root, trans_file)
            audio_folder = os.path.dirname(trans_file_path)
            transcription_text = {}

            # Read transcription lines from the .trans.txt file
            with open(trans_file_path) as trans_file:
                for line in trans_file:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        audio_filename = parts[0]
                        transcription = parts[1]

                        # Find matching audio files using the regular expression
                        audio_match = audio_file_pattern_no_ending.match(audio_filename)
                        if audio_match:
                            audio_basename = audio_match.group(1)
                            audio_abs_path = os.path.join(
                                audio_folder, f"{audio_basename}.flac"
                            )
                            if os.path.isfile(audio_abs_path):
                                data[audio_abs_path] = transcription.strip()
                                transcription_text[audio_filename] = (
                                    transcription.strip()
                                )

    return data


def remove_punctuations(s):
    """
    Removes punctuations from the input strings.

    Args:
    - s (str): Input string from which punctuations will be removed.

    Returns:
    - str: String with punctuations removed.
    """
    ls = list(filter(None, re.split(r'[ ",.!?]+', s)))
    return " ".join(ls)


def ser(asr_transcription, corrected_asr_transcription, reference_transcription):
    """
    Computes SER between corrected ASR transcription and reference transcription.

    Args:
    - asr_transcription (str): Original ASR transcription.
    - corrected_asr_transcription (str): Corrected ASR transcription.
    - reference_transcription (str): Reference transcription.

    Returns:
    - float: SER score between corrected ASR transcription and reference transcription.
    - float: SER score between original ASR transcription and reference transcription.
    """
    asr_transcription = remove_punctuations(asr_transcription.lower())
    corrected_asr_transcription = remove_punctuations(
        corrected_asr_transcription.lower()
    )
    reference_transcription = remove_punctuations(reference_transcription.lower())

    SER = 100 - (corrected_asr_transcription == reference_transcription) * 100
    SER_original = 100 - (asr_transcription == reference_transcription) * 100

    return SER, SER_original


def confidence_score_sentence_level(trans, confidence=True):
    """
    Computes confidence score at sentence level.

    Args:
    - trans (dict): Dictionary containing ASR transcription text.
    - confidence (bool): Whether to compute confidence score (default is True).

    Returns:
    - dict: Dictionary containing text and optionally confidence score.
    """
    d = {}
    d["text"] = trans["text"]
    if confidence:
        ls = []
        for seg in trans["segments"]:
            ls.append(seg["confidence"])
        confidence_score = sum(ls) / len(ls)
        d["confidence_score"] = confidence_score
    return d


def confidence_score_lowest_word_level(trans, confidence=True):
    """
    Computes confidence score at lowest word level.

    Args:
    - trans (dict): Dictionary containing ASR transcription text.
    - confidence (bool): Whether to compute confidence score (default is True).

    Returns:
    - dict: Dictionary containing text and optionally confidence score.
    """
    d = {}
    d["text"] = trans["text"]
    words = []
    for seg in trans["segments"]:
        words.extend(seg["words"])

    lowest = 10000.0
    for word in words:
        if word["confidence"] < lowest:
            lowest = word["confidence"]

    d["confidence_score"] = lowest
    return d


def confidence_score_average_word_level(trans, confidence=True):
    """
    Computes average confidence score at word level.

    Args:
    - trans (dict): Dictionary containing ASR transcription text.
    - confidence (bool): Whether to compute confidence score (default is True).

    Returns:
    - dict: Dictionary containing text and optionally confidence score.
    """
    d = {}
    d["text"] = trans["text"]
    words = []
    for seg in trans["segments"]:
        words.extend(seg["words"])

    avg_confidence_score = 0
    count = 0
    for word in words:
        count += 1
        avg_confidence_score += word["confidence"]

    d["confidence_score"] = avg_confidence_score / count
    return d


def confidence_score_word_level(trans, confidence=True):
    """
    Computes confidence score for each word in transcription.

    Args:
    - trans (dict): Dictionary containing ASR transcription text.
    - confidence (bool): Whether to compute confidence score (default is True).

    Returns:
    - dict: Dictionary containing text and list of words with confidence scores.
    """
    d = {}
    d["text"] = trans["text"]
    words = []
    for seg in trans["segments"]:
        words.extend(seg["words"])

    ls = []
    for word in words:
        ls.append({"text": word["text"], "confidence": word["confidence"]})

    d["words"] = ls
    return d
