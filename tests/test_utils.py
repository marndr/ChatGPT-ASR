
from chat_gpt_asr.utils import confidence_score_sentence_level


def test_confidence_score_with_segments():
    
    input_transcription = {
        "text": "Sample transcription",
        "segments": [
            {"confidence": 0.8},
            {"confidence": 0.9},
            {"confidence": 0.85}
        ]
    }
    
    expected_output = {
        "text": "Sample transcription",
        "confidence_score": 0.85
    }
    
    assert confidence_score_sentence_level(input_transcription) == expected_output

def test_confidence_score_without_segments():
    
    input_transcription = {
        "text": "Sample transcription",
        "confidence_score": 0.75
    }
    
    expected_output = {
        "text": "Sample transcription",
        "confidence_score": 0.75
    }
    
    assert confidence_score_sentence_level(input_transcription) == expected_output



