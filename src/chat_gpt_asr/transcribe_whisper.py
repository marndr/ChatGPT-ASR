import whisper_timestamped as whisper


class Transcriber:
    def __init__(self, whisper_model="tiny", device="cpu"):
        self.model = whisper.load_model(whisper_model, device=device)

    def transcribe(self, audio_filename):
        audio = whisper.load_audio(audio_filename)
        result = whisper.transcribe(self.model, audio, language="en", verbose=None)
        return result
