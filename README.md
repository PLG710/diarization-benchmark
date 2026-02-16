.venv einrichten
pip install -r requirements.txt

mkdir data results src .env
mkdir data/input_audio data/test_audio data/normalised_audio

In .env setzen:
VLLM_BASE_URL=## (z.B. http://127.0.0.1:8001/v1, wenn auf localhost Port 8001 verwendet wird)
VLLM_MODEL=## (Das benutzte lokale Modell, z.B. openai/gpt-oss-120b, wie in unserem Test)
REPO_ROOT=## (Hauptpfad des Repositories, z.B. "~/jupyter/diarization-benchmark")