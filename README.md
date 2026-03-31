# Local Speaker Attribution in German Medical Conversations: Comparing LLM Pipelines with Audio-Based Diarization

This archive contains the pipeline and evaluation code accompanying the paper:

> **Local Speaker Attribution in German Medical Conversations: Comparing LLM Pipelines with Audio-Based Diarization**  
> Liegel P, Christoph J, Demus C, Wattar A, Jäger C (2026)

**Note:** This Zenodo archive contains code only. Audio files, generated result folders, manual reference annotations, and other data artifacts are not included. To reproduce the workflow, users must provide their own audio files and, for full quantitative evaluation, their own reference speaker annotations.

---

## Overview and Motivation

In clinical and other privacy-sensitive settings, audio and text data must often be processed locally to comply with data protection requirements (e.g. GDPR). For reusing recorded conversations — such as for documentation, quality management, or research — two components are particularly relevant:

1. **Local transcription**: *What was said?*
2. **Speaker attribution**: *Who said it?*

This repository implements and compares two local approaches to speaker attribution:

- **Transcript-only LLM attribution (Pipeline A):** A locally hosted LLM receives a transcript and infers speaker turns from textual context alone — no audio required.
- **Audio-guided Pyannote attribution (Pipelines B–D):** Pyannote uses acoustic features to produce speaker turns, which are then mapped onto the ASR transcript via post-processing.

A core design decision is the **shared-transcript design**: faster-whisper is run once per file, and all pipelines operate on the same canonical transcript. This isolates speaker attribution quality from ASR variability and enables a fair comparison across methods.

---

## Pipeline Overview

### Pipeline A — Transcript-only LLM attribution

Two locally hosted instruction-tuned models, each evaluated in four settings (raw vs. presegmented input × known vs. unknown speaker count):

| Pipeline | Model |
|----------|-------|
| A1 | MedGemma-27B-IT |
| A2 | Llama-3.1-8B-Instruct |

- **Raw input**: ASR transcript with line breaks removed
- **Presegmented input**: Transcript preserving faster-whisper segment boundaries
- **Known condition**: Number of speakers specified in the prompt
- **Unknown condition**: Number of speakers must be inferred from the text

Models are served via vLLM with deterministic decoding (temperature 0.0). Prompts require exact transcript preservation and consecutive speaker labels of the form `[Sprecher N]:`.

### Pipeline B — Segment-level Pyannote mapping

Pyannote community-1 exclusive speaker turns are mapped to faster-whisper ASR segments by maximum temporal overlap.

### Pipeline C — Word-level Viterbi mapping

Pyannote turns are mapped to the word level using native faster-whisper word timestamps and Viterbi smoothing.

### Pipeline D — Word-level mapping with WhisperX alignment (supplementary)

Same as C, but native faster-whisper word timestamps are replaced with WhisperX forced alignment results. Serves as a sensitivity analysis.

All pipelines produce output in the format:

```
[Sprecher 1]: Inhalt
[Sprecher 2]: Inhalt
```

---

## Repository Structure

```
data/
  input_audio/          # user-provided audio files for batch processing; not included in this archive
  test_audio/           # user-provided audio files for single tests; not included in this archive
  normalised_audio/     # auto-generated during pipeline execution; not included in this archive
results/                # auto-generated during pipeline execution; not included in this archive
  <file_stem>/
    transcript.txt              # canonical faster-whisper transcript
    whisper_segments.json       # segment-level ASR output
    whisper_words.json          # word-level ASR output (with timestamps)
    diarization_ref.txt         # manual reference annotations (user-created; see below)
    LLM_diarized_<config>.txt   # LLM attribution outputs (one per config)
    Pyannote_diarized_<config>.txt  # Pyannote attribution outputs
    meta.json                   # runtime metadata per pipeline
  batch_summary.json    # aggregated metadata across all files (overwritten on re-run)
01_pipeline.ipynb            # main pipeline notebook
02_evaluation.ipynb          # evaluation, metrics, and figures
requirements.txt
constraints_linux_gpu.txt    # pinned dependencies for Linux with CUDA
constraints_linux_cpu.txt    # pinned dependencies for Linux CPU-only
constraints_windows_cpu.txt  # pinned dependencies for Windows CPU-only
```

---

## Prerequisites

- **Python 3.12**
- **CUDA-capable GPU** recommended (≥24 GB VRAM for MedGemma-27B; paper experiments ran on NVIDIA RTX PRO 6000 Blackwell with 96 GB VRAM). CPU-only execution is possible but substantially slower.
- **ffmpeg** for audio normalization:
  - Ubuntu/Debian: `sudo apt-get install -y ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: download from https://www.gyan.dev/ffmpeg/builds/#release-builds
- **Hugging Face token** with access to Pyannote models (accept model terms at https://huggingface.co/pyannote/speaker-diarization-community-1)
- A running **vLLM server** for LLM-based pipelines (Pipeline A)

---

## Setup

### 1. Virtual environment and dependencies

Choose the constraints file matching your setup:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Linux with GPU (recommended):
pip install -c constraints_linux_gpu.txt -r requirements.txt

# Linux CPU-only:
pip install -c constraints_linux_cpu.txt -r requirements.txt

# Windows CPU-only:
pip install -c constraints_windows_cpu.txt -r requirements.txt
```

### 2. Create required directories

```bash
mkdir -p data/input_audio data/test_audio data/normalised_audio results
```

### 3. Create `.env` file

```bash
touch .env
```

---

## Environment Configuration (`.env`)

```bash
# vLLM / LLM
VLLM_BASE_URL=http://127.0.0.1:8001/v1
VLLM_MODEL=google/medgemma-27b-it   # or meta-llama/Llama-3.1-8B-Instruct

# Repository
REPO_ROOT=~/jupyter/diarization-benchmark
TEST_AUDIO=data/test_audio/test.wav

# Pyannote
PYANNOTE_MODEL_ID=pyannote/speaker-diarization-community-1
HUGGINGFACE_TOKEN=hf_xxx
```

**Notes:**
- `VLLM_BASE_URL` must include `/v1`
- `TEST_AUDIO` can be relative to `REPO_ROOT` or an absolute path
- Do **not** commit `.env` — it contains your Hugging Face token (see `.gitignore`)

---

## Software Versions (as used in the paper)

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| faster-whisper | 1.2.1 (large-v3) |
| pyannote.audio | 4.0.4 (community-1, exclusive mode) |
| WhisperX | 3.8.2 |
| vLLM | v.0.18.0 |
| PyTorch | 2.8.0+cu128 |

---

## Usage

### Single-file tests

`01_pipeline.ipynb` contains test switches for individual components. Set the corresponding flag to `True` to run a test on `TEST_AUDIO`:

| Switch | Tests |
|--------|-------|
| `conversion_test` | Audio normalization |
| `whisper_test` | faster-whisper transcription |
| `llm_test` | vLLM connectivity |
| `pyannote_segment_test` | Segment-level Pyannote mapping |
| `pyannote_word_test` | Word-level Viterbi mapping |

Leave all switches at `False` for batch runs.

### Batch run

1. Place audio files in `data/input_audio/`
2. Name files using the pattern `<name>_<N>S.<ext>` where `N` is the number of speakers (e.g. `Konsil_2S.wav`, `Meeting_4S.mp3`). This is necessary to enable known speaker conditions.
3. Run all cells in `01_pipeline.ipynb`

For each file, a folder `results/<file_stem>/` is created with all pipeline outputs and a `meta.json`. An aggregated `results/batch_summary.json` is written after the full run (overwritten on re-run).

### Running additional LLM configurations

Cell 32 in `01_pipeline.ipynb` allows running additional LLM models or prompt configurations over existing transcripts without re-running Pyannote or faster-whisper.

---

## Evaluation

`02_evaluation.ipynb` computes all metrics and reproduces the figures from the paper.

**Important:** This notebook expects generated outputs under `results/<file_stem>/`, including transcripts, system outputs, metadata, and reference annotations. These artifacts are not part of this archive and must be created locally by running `01_pipeline.ipynb` first.

### Reference annotations

For quantitative evaluation (WSER), manual reference speaker labels must be created in `diarization_ref.txt` within each `results/<file_stem>/` folder, directly on the canonical faster-whisper transcript, using the format:

```
[Sprecher 1]: text
[Sprecher 2]: text
```

The token stream should not be manually corrected: substitutions remain as-is, insertions are attributed to the active speaker, and deletions are not reintroduced. This ensures strict comparability across pipelines.

### Metrics

| Metric | Description |
|--------|-------------|
| WSER (primary) | Word Speaker Error Rate: proportion of words assigned to the wrong speaker, after optimal permutation matching of predicted and reference labels |
| Validity Rate (VR) | Proportion of LLM outputs that preserved the transcript exactly |
| Real-Time Factor (RTF) | Processing time relative to audio duration |
| Speaker Count Accuracy (SCA) | Proportion of files where predicted speaker count matched reference |

---

## Output and Runtime Metadata

`meta.json` stores per file:

- Model IDs (faster-whisper, LLM, Pyannote)
- Pipeline status (ok/error + message)
- Audio duration (`audio_duration_s`)
- Runtimes in seconds: `normalize`, `transcribe`, `llm`, `pyannote`, `postproc`, `total`

---

## Methodological Notes and Limitations

### Segment-level mapping (Pipeline B)
Whisper segments are assigned to the speaker with maximum temporal overlap with Pyannote turns. Fast and transparent, but can be inaccurate when speaker changes occur within a long segment or for very short interjections.

### Word-level Viterbi mapping (Pipeline C/D)
More fine-grained than segment mapping. Viterbi smoothing prevents single misattributed words from creating spurious speaker switches. Generally expected to help more when overlapping speech and interruptions are frequent.

### WhisperX alignment (Pipeline D)
Replaces native faster-whisper word timestamps with forced alignment via WhisperX/wav2vec 2.0. In the paper benchmark, this yielded no measurable WSER improvement over Pipeline C.

### Context length
LLM attribution is limited by model context windows, which may restrict applicability to longer recordings.

### Filename collisions
If multiple audio files share the same stem, results will be overwritten since outputs are stored under `results/<stem>/`. Use unique filenames.

---

## Data Availability

Audio data and generated results are not part of this archive:

- Ärztesprech dialogues: https://www.aerztesprech.de (used with written consent)
- DGD corpus: https://www.dgd.ids-mannheim.de (used under IDS terms for non-commercial research), FOLK
- Simulated audio: available on request from the authors

## Citation

If you use this code, please cite the accompanying GMDS 2026 paper and the Zenodo archive once the DOI is available.