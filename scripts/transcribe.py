#!/usr/bin/env python3

import argparse
import json
import os

from dotenv import load_dotenv
from tqdm import tqdm

from chat_gpt_asr.transcribe_whisper import Transcriber
from chat_gpt_asr.utils import read_librispeech_transcriptions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="whisper model")
    parser.add_argument(
        "-wm",
        "--whisper_model",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="tiny",
        help="Select the Whisper model",
    )
    parser.add_argument(
        "-s",
        "--subset",
        choices=["dev-clean", "dev-other", "test-clean", "test-other"],
        default="dev-clean",
        help="Librispeech subset to transcribe",
    )
    parser.add_argument("-d", "--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    load_dotenv()
    Root_Librispeech = os.getenv("ROOT_LIBRISPEECH")
    Root = os.getenv("ROOT_PATH")

    output_folder = os.path.join(Root, "data/transcriptions")
    output_file = os.path.join(
        output_folder,
        f"whisper_{args.whisper_model}_librispeech_{args.subset}-full.json",
    )

    data = read_librispeech_transcriptions(
        root_folder=os.path.join(Root_Librispeech, args.subset)
    )
    l = []
    whisper = Transcriber(whisper_model=args.whisper_model, device=args.device)

    for audio_path, reference_transcription in tqdm(data.items()):
        try:
            asr_transcription = whisper.transcribe(audio_path)
        except Exception as e:
            asr_transcription = ""
            print(e)
        l.append(
            {
                "asr_transcription": asr_transcription,
                "reference_transcription": reference_transcription,
            }
        )

    with open(output_file, "w") as f:
        json_str = json.dumps(l, indent=2)
        f.write(json_str)
