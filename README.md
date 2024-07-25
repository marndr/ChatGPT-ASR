# ChatGPT-ASR
## MASTER AI
This git is the implementation of the master thesis of Maryam Naderi.
The master thesis took place between 2023 and 2024.
The goal of the master thesis is to study correcting ASR trancriptions using LLMs.
A part of this project has already been accepted in the form of a peer-reviewed paper
titled "Towards interfacing large language models with ASR systems using
confidence measures and prompting" at the Interspeech 2024 Conference.

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

## Development

Install the package with the development dependencies.

```bash
pip install -e .[dev]
```

The code is formatted and linted with [ruff](https://docs.astral.sh/ruff/). It runs
automatically in [pre-commit hooks](.pre-commit-config.yaml) by the CI, but you
can also install them for local use:

```bash
pre-commit install
```

The following then automatically formats your code and tells you whether any further
changes are required to pass the lint checks:

```bash
pre-commit run --all-files
```

Documentation of all lint rules included with ruff:
https://docs.astral.sh/ruff/rules Specific rules can be ignored with a `noqa`
comment:

```python
path = "chat-gpt-asr/results/results-dev-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json"  # noqa: E501
```

Also see the the `pyproject.toml` file on how to turn on/off rules in general or
for specific folders.

## Datasets

We have four datasets categorized into two sets:

1. **Development Set (`dev-set`):**
   - **Clean (`dev-clean`):** Referred to as "clean" in this repository.
   - **Noisy (`dev-other`):** Referred to as "noisy" in this repository.

2. **Test Set (`test-set`):**
   - **Clean (`test-clean`):** Referred to as "clean" in this repository.
   - **Noisy (`test-other`):** Referred to as "noisy" in this repository.


## Running experiments

To run sentence confidence experiment for tiny model and `dev-clean`, first run:
```bash
cd scripts/dev-set-scripts/clean/Tiny-clean/exp_sentence_confidence_tiny/exp_GPT-3.5-Turbo_tiny
python exp_without_sentence_confidence_tiny.py
```

Then run:
```bash
python exp_find_thresh_sentence_confidence_tiny.py
```
## Evaluation

To evaluate the results in overall, run:
```bash
cd script
python Overall_Evaluation.py
```

The expected results for WER, CER and SER are saved in this path:
```
results/results-dev-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.-Turbo_tiny/gpt-3.5-turbo-1106/results_without_sentence_confidence_tiny/results_overall_sentence_confidence_tiny.md
```
## LICENSE
```
Copyright 2023 Idiap Research Institute <contact@idiap.ch

SPDX-FileContributor: Maryam Naderi <maryam.naderi@idiap.ch>

SPDX-License-Identifier: LicenseRef-chatgpt-asr

```
