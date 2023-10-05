from utils import read_librispeech_transcriptions
import json

def transcribe(audio_filename):
    audio = whisper.load_audio(audio_filename)
    model = whisper.load_model("tiny", device="cpu")
    result = whisper.transcribe(model, audio, language="en")
    return result["text"]

if __name__=="__main__":
    root = "/home/mnaderi/Documents/thesis/whisperii/LibriSpeech/dev-clean"
    data = read_librispeech_transcriptions(root_folder=root)
        
    l=[]
    length = len(data)
    for i, (audio_path, reference_transcription) in enumerate(data.items()):
        asr_transcription  = transcribe(audio_path)
        l.append({
            "asr_transcription":asr_transcription,
            "reference_transcription":reference_transcription
            })
        print(f"{i+1}/{length} completed!")
    
    with open("../data/whisper_transcriptions.json", "w") as f:
        json_str = json.dumps(l, indent=2)
        f.write(json_str)
