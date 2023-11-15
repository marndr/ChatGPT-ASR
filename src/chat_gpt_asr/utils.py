import os
import re

def read_dummy_transcriptions():
    dataset =[
    {"asr_transcription":"I prefer see over coffee.",
        "reference_transcription": "I prefer tea over coffee."
    },
    {"asr_transcription":"I need to catch a drain to London." ,"reference_transcription": "I need to catch a train to London."
    },
    {"asr_transcription":"I'll beat you at the coffee shop.", "reference_transcription": "I'll meet you at the coffee shop."
    },
    {"asr_transcription":"He's a professional sheaf.","reference_transcription":"He's a professional chef."
    },
    {"asr_transcription":"The leather is nice today." ,"reference_transcription" :"The weather is nice today."
    },
    {"asr_transcription":"Please pass me the vault.", "reference_transcription" : "Please pass me the salt."
    },
    {"asr_transcription":"I'll be their in five minuets.","reference_transcription" : "I'll be there in five minutes."
    },
    {"asr_transcription":"I'm going to the gross restore.", "reference_transcription" :"I'm going to the grocery store."
    },
    {"asr_transcription":"I won't to go two the beech.","reference_transcription" : "I want to go to the beach."
    },
    {"asr_transcription":"The son is shining brightly in the sky.", "reference_transcription" : "The sun is shining brightly in the sky."
    },
    {"asr_transcription":"The son sets at the beech are always stunting.", "reference_transcription" : "The sun sets at the beach are always stunning."
}]
    return dataset


# Function to read audio filenames and their corresponding transcriptions
def read_librispeech_transcriptions(root_folder="."):
    data = {}

    # Regular expression to match audio filenames
    audio_file_pattern = re.compile(r'(\d+-\d+-\d+)\.flac')
    audio_file_pattern_no_ending = re.compile(r'(\d+-\d+-\d+)')

    for root, dirs, files in os.walk(root_folder):
        trans_files = [f for f in files if f.endswith('.trans.txt')]
        for trans_file in trans_files:

            trans_file_path = os.path.join(root, trans_file)
            audio_folder = os.path.dirname(trans_file_path)
            transcription_text = {}

            # Read transcription lines from the .trans.txt file
            with open(trans_file_path, 'r') as trans_file:
                for line in trans_file:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        audio_filename = parts[0]
                        transcription = parts[1]

                        # Find matching audio files using the regular expression
                        audio_match = audio_file_pattern_no_ending.match(audio_filename)
                        if audio_match:
                            audio_basename = audio_match.group(1)
                            audio_abs_path = os.path.join(audio_folder, f"{audio_basename}.flac")
                            if os.path.isfile(audio_abs_path):
                                data[audio_abs_path] = transcription.strip()
                                transcription_text[audio_filename] = transcription.strip()

    return data


def remove_punctuations(s):
    l = list(filter(None, re.split(r'[ ,.!?]+', s)))
    return " ".join(l)


def confidence_score_lowest_word_level(trans):
    d={}
    d["text"]= trans["text"]
    words=[]
    for i, seg in enumerate(trans["segments"]):
        words.extend(seg["words"])

    lowest=10000.
    for word in words:
    	if word["confidence"] < lowest:
    		lowest = word["confidence"]

    d["confidence_score"]=lowest
    return d


def confidence_score_word_level(trans):
    d={}
    d["text"]= trans["text"]
    words=[]
    for i, seg in enumerate(trans["segments"]):
        words.extend(seg["words"])

    l=[]
    for word in words:
        l.append({"text": word["text"], "confidence": word["confidence"]})

    d["words"]= l
    return d
    
    
def confidence_score_sentence_level(trans, confidence = True):
    d = {}
    d["text"] = trans["text"]
    if confidence:
        
        l=[]
        for i, seg in enumerate (trans["segments"]):
            l.append(seg["confidence"])
        confidence_score = sum(l)/len(l)
        d["confidence_score"]=confidence_score
    return d

def ser(asr_transcription, corrected_asr_transcription, reference_transcription):

    asr_transcription = remove_punctuations(asr_transcription.lower())
    corrected_asr_transcription = remove_punctuations(corrected_asr_transcription.lower())
    reference_transcription = remove_punctuations(reference_transcription.lower())

    SER = 100 - (corrected_asr_transcription == reference_transcription)*100
    SER_original = 100 - (asr_transcription == reference_transcription)*100
    
    
    return SER,SER_original

