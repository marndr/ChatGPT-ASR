import json
import argparse
from dotenv import load_dotenv
import os


from chat_gpt_asr.transcribe_whisper import transcribe
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
        choices=["dev-clean", "dev-other"],
        default="dev-clean",
        help="Librispeech subset to transcribe",
    )
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
    length = len(data)

    for i, (audio_path, reference_transcription) in enumerate(data.items()):
        try:
            asr_transcription = transcribe(audio_path, args.whisper_model)
            l.append(
                {
                    "asr_transcription": asr_transcription,
                    "reference_transcription": reference_transcription,
                }
            )
            print(f"{len(l)}/{length} completed!")
        except:
            continue

    with open(output_file, "w") as f:
        json_str = json.dumps(l, indent=2)
        f.write(json_str)
