import whisper_timestamped as whisper


def transcribe(audio_filename, whisper_model="tiny"):
    audio = whisper.load_audio(audio_filename)
    model = whisper.load_model(whisper_model, device="cpu")
    result = whisper.transcribe(model, audio, language="en")
    return result
