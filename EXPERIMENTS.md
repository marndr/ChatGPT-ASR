# Overview

This document provides detailed instructions on how to re-run the experiments for correcting ASR transcriptions using GPT models in this project.
It includes steps to prepare the environment, run the experiments, and the expected results for this project.

## Preparing the Environment

1. **Clone the Repository:**

    ```bash
    git clone https://gitlab.idiap.ch/mnaderi/chat-gpt-asr.git
    cd chat-gpt-asr
    ```

2. **Set Up the Virtual Environment:**

    ```bash
    conda create -n chat-gpt-asr python=3.10
    conda activate chat-gpt-asr
    pip install -r requirements.txt
    pip install -e .
    ```

3. **Download Spacy Model:**

    ```bash
    python -m spacy download en_core_web_sm
    ```

4. **Set Up Environment Variables:**

    ```bash
    cp .env.example .env
    ```

5. **Add Your OpenAI API Key:**

    Edit the `.env` file to include your OpenAI API key.

    ```
    OPENAI_API_KEY=<your_openai_api_key>
    ROOT_PATH=<your_root_path>
    ```

## Running the Experiments

### Experiment 1: ASR Correction with GPT model

#### Description

This experiment corrects ASR transcriptions using GPT model without considering confidence measures.

#### Steps

1. **Transcribe Data with Whisper:**

    ```bash
    python scripts/transcribe.py
    ```

2. **Run the Correction Script (one experiment for example):**

    ```bash
    cd scripts/dev-set-scripts/clean/Tiny-clean/exp_sentence_confidence_tiny/exp_GPT-3.5-Turbo_tiny
    python exp_without_sentence_confidence_tiny.py -d librispeech -n 1
    ```

    **Parameters:**

    - `-d` or `--dataset`: Specify the dataset to use (`librispeech`). Default is `librispeech`.
    - `-n` or `--num_data`: Specify the number of data points to process. Default is `-1` (process all data).

3. **Expected Results:**

    The corrected transcriptions will be saved in:

    ```
    results/results-dev-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_without_sentence_confidence_tiny/corrected_transcriptions_sentence_confidence_tiny.json
    ```

### Experiment 2: Confidence Threshold Evaluation and Plotting

#### Description

This experiment evaluates the impact of various confidence thresholds on Word Error Rate (WER) and Character Error Rate (CER) for the corrected ASR transcriptions.

#### Steps

1. **Run the Threshold Evaluation Script:**

    ```bash
    python exp_find_thresh_sentence_confidence_tiny.py
    ```

2. **Expected Results:**

    The WER and CER results for various confidence thresholds will be saved in:

    ```
    results/results-dev-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_find_thresh_sentence_confidence_tiny/results_thresh_sentence_confidence_tiny.md
    ```

    Plots of WER and CER against confidence thresholds will be saved in:

    ```
    results/results-dev-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-Turbo_tiny/gpt-3.5-turbo-1106/results_find_thresh_sentence_confidence_tiny/plots_sentence_confidence_tiny/Wer_vs_sentence_confidence_plot_tiny.png
    results/results-dev-set/results_clean/results_tiny/results_sentence_confidence_tiny/results_GPT-3.5-turbo-1106/results_find_thresh_sentence_confidence_tiny/plots_sentence_confidence_tiny/Cer_vs_sentence_confidence_plot_tiny.png
    ```

3. **Evaluation:**

    The overall evaluation results including WER, CER, and SER are expected to be documented and can be reviewed for performance analysis (by Overall_Evaluation.py in scripts).
