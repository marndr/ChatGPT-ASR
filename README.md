# ChatGPT-ASR

## Setup

1. Clone this repository.

2. Navigate into the project directory:
```bash
cd chat-gpt-asr
```

3. Create a new virtual environment and install the requirements:
```bash
conda create -n chat-gpt-asr python=3.10
conda activate chat-gpt-asr
pip install -r requirements.txt
pip install -e .

# Download Spacy model for POS tag analysis
python -m spacy download en_core_web_sm
```

4. Make a copy of the example environment variables file:
```bash
cp .env.example .env
```

5. Add your API key to the newly created .env file.

## Scripts and experiments

### Transcribe data with Whisper

Transcribe the Librispeech `dev-clean` subset with Whisper's `tiny` model and
save the output in `data/transcriptions/`:
```bash
python scripts/transcribe.py
```
