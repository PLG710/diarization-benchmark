.venv einrichten
pip install -r requirements.txt

mkdir data results src .env
mkdir data/input_audio data/test_audio data/normalised_audio

In .env setzen:
VLLM_BASE_URL=## (z.B. http://127.0.0.1:8001/v1, wenn auf localhost Port 8001 verwendet wird)
VLLM_MODEL=## (Das benutzte lokale Modell, z.B. google/medgemma-27b-it wie in unserem Test)
REPO_ROOT=## (Hauptpfad des Repositories, z.B. "~/jupyter/diarization-benchmark")
TEST_FILE=## (Pfad angeben relativ zur REPO-ROOT, z.B: "/data/test/test2.mp3")
PYANNOTE_MODEL_ID=## (z.B. "pyannote/speaker-diarization-community-1")
HUGGINGFACE_TOKEN=## (Hugging-Face-Token anlegen unter https://huggingface.co/pyannote/speaker-diarization-community-1)