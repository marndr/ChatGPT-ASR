import json
import os

from chat_gpt_asr.utils import confidence_score_word_level
from dotenv import load_dotenv

load_dotenv()
Root = os.getenv("ROOT_PATH")

THRESH = 0.6

transcription_file = os.path.join(
    Root, "data/transcriptions/whisper_tiny_librispeech_dev-clean-full.json"
)


with open(transcription_file) as f:
    json_obj = f.read()
    data_1 = json.loads(json_obj)

data_list_1 = []
for i, d in enumerate(data_1):
    asr_transcription = confidence_score_word_level(
        d["asr_transcription"], confidence=True
    )

    low_confidence_words = [
        word["text"]
        for word in asr_transcription["words"]
        if word["confidence"] <= THRESH
    ]
    if len(low_confidence_words) == 0:
        asr_transcription = {
            "text": asr_transcription["text"],
            "low_confidence_words": low_confidence_words,
        }
        reference_transcription = d["reference_transcription"]
        data_list_1.append(
            {
                "asr_transcription": asr_transcription,
                "reference_transcription": reference_transcription,
                "corrected_asr_transcription": asr_transcription["text"],
            }
        )


corrected_transcription_file = os.path.join(
    Root,
    "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/preprocessed_corrected_transcriptions_certain_low_confidence_words_tiny_new/preprocessed_corrected_transcriptions_Thresh=0.6_tiny.json",
)

with open(corrected_transcription_file) as f:
    json_obj = f.read()
    data_list_2 = json.loads(json_obj)


data_total = data_list_1 + data_list_2

output_file = os.path.join(
    Root,
    "results/results-dev-set/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/postprocessed_corrected_transcriptions_certain_low_confidence_words_tiny_new/postprocessed_corrected_transcriptions_processed_Thresh=0.6_tiny.json",
)

with open(output_file, "w") as f:
    json_str = json.dumps(data_total, indent=2)
    f.write(json_str)
