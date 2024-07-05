# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Maryam Naderi  <maryam.naderi@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-chatgpt-asr

"""
This module provides a Transcriber class for loading a Whisper model and
transcribing audio files.
"""

import whisper_timestamped as whisper


class Transcriber:
    """
    A class to handle the transcription of audio files using the Whisper model.

    Attributes:
        model: The Whisper model loaded with the specified configuration.
    """

    def __init__(self, whisper_model="tiny", device="cpu"):
        """
        Initialize the Transcriber with a specified Whisper model and device.

        Args:
            whisper_model (str): The name of the Whisper model to load.
            Default is "tiny".
            device (str): The device to run the model on ("cpu" or "cuda").
            Default is "cpu".
        """
        self.model = whisper.load_model(whisper_model, device=device)

    def transcribe(self, audio_filename):
        """
        Transcribe an audio file using the loaded Whisper model.

        Args:
            audio_filename (str): The path to the audio file to transcribe.

        Returns:
            dict: The transcription results from the Whisper model.
        """
        audio = whisper.load_audio(audio_filename)
        return whisper.transcribe(self.model, audio, language="en", verbose=None)
