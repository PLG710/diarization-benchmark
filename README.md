.venv einrichten
pip install -r requirements.txt

mkdir data results src .env
mkdir data/input_audio data/test_audio data/normalised_audio

In .env setzen:
VLLM_BASE_URL=## (z.B. http://127.0.0.1:8001/v1)
VLLM_MODEL=## (z.B. openai/gpt-oss-120b, wie in unserem Test)


## ggf. Teile im Config-Block des Jupyter-Notebooks anpassen