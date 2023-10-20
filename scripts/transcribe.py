import json
from pathlib import Path

from chat_gpt_asr.transcribe_whisper import transcribe
from chat_gpt_asr.utils import read_librispeech_transcriptions

if __name__ == "__main__":
    whisper_model = "tiny"
    subset = "dev-clean"
    root = Path("/home/mnaderi/Documents/thesis/whisperii/LibriSpeech")
    output_folder = Path("data/transcriptions")
    output_file = output_folder / f"whisper_{whisper_model}_librispeech_{subset}.json"

    data = read_librispeech_transcriptions(root_folder=str(root / subset))
    l = []
    length = len(data)
    for audio_path, reference_transcription in data.items():
        asr_transcription = transcribe(audio_path, whisper_model)
        l.append(
            {
                "asr_transcription": asr_transcription,
                "reference_transcription": reference_transcription,
            }
        )
        print(f"{len(l)}/{length} completed!")

    with open(output_file, "w") as f:
        json_str = json.dumps(l, indent=2)
        f.write(json_str)
