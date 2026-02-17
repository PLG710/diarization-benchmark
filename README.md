# Diarization Benchmark: LLM-basierte vs. audio-basierte Sprecherdiarisierung in lokalem Setting

## Kontext und Motivation

In vielen Anwendungsfällen (z.B. Medizin, Verwaltung, Forschung, Industrie) ist eine Verarbeitung von Audio- und Textdaten **lokal** notwendig, um Datenschutz-, Compliance- oder Governance-Anforderungen zu erfüllen. Für die Nachnutzung von Gesprächen (z.B. Dokumentation, Qualitätsmanagement, Forschungsauswertung) sind dabei zwei Bausteine besonders relevant:

1) **Lokale Transkription**: *Was wurde gesagt?*  
2) **Sprecherdiarisierung**: *Wer hat gesprochen?*

Eine Sprecherzuordnung erleichtert u.a. die Nachvollziehbarkeit, die Verständlichkeit sowie die spätere Auswertung von Gesprächen, Meetings oder ähnlichem.

Dieses Repository stellt eine reproduzierbare Benchmark-Pipeline bereit, um zwei Ansätze zur Sprecherzuordnung zu vergleichen:

- **Text-/Kontext-basierte Diarisierung (LLM):** Ein lokal gehostetes LLM über vLLM erhält ein Transkript und soll Sprecherwechsel und Sprechersegmente rein aus dem textuellen Kontext ableiten.
- **Audio-basierte Diarisierung (Pyannote):** Pyannote nutzt akkustische Merkmale und liefert anhand des Audios Speaker-Turns, über welche sich mit etwas Post-Processing Segmente und gesagte Worte verschiedenen Sprechern zuordnen lassen.

Beide Varianten erfüllen das Kriterium der **lokalen Datenverarbeitung**. Im Notebook sind Code-Ansätze zur Integration beider Methoden in ein eigenes lokales Setup aufgeführt. Es ist so konfiguriert, dass beliebige LLMs mit beliebigen Pyannote-Versionen verglichen und daraus Daten erzeugt werden können. Für die Erzeugung des Transkripts und der benötigten Segmente für Pyannote wird **faster-whisper** verwendet, ein Tool, welches lokal Audio-Dateien transkribiert.

**Zentrale Fragestellung:**  
Reicht eine kontextbasierte LLM-Diarisierung auf Transkripten für die meisten Anwendungsfälle aus oder liefern audio-basierte Methoden (z.B. Pyannote)per se robustere Sprecherzuordnung - wo liegen die Grenzen und Stärken beider Ansätze? 

---

## Überblick über die Pipeline

Das zentrale Notebook ist `01_pipeline.ipynb`. Es führt pro Audiodatei zwei Pipelines aus:

### Pipeline A: Transkription → LLM (textbasiert)
- Audio-Normalisierung auf **WAV, 16 kHz, mono** für Vergleichbarkeit mit Pyannote
- Transkription via **faster-whisper** (lokal verfügbares Tool zur Transkription von Audio-Input)
- LLM (lokal gehostet über vLLM) erzeugt über individuell konfigurierbare Prompt ein diarisiertes Format

### Pipeline B: Transkription → Pyannote → Post-Processing (audiobasiert)
- Audio-Normalisierung auf **WAV, 16 kHz, mono** (bestes Verarbeitungsformat für Pyannote)
- **Pyannote** erzeugt Speaker-Turns (wer spricht wann)
- Post-Processing: Mapping und nicht-rechenintensive Formatierungen zur Erzeugung einer menschenlesbaren Ausgabe

In beiden Pipelines entsteht zur Vergleichbarkeit folgender Text-Output:

`[Sprecher 1]: Inhalt`<br>
`[Sprecher 2]: Inhalt`

---

## Repository-Struktur

Die folgende Struktur wird vom Notebook erwartet/verwendet:

    data/
      input_audio/         # Zu testende Audios, die batch-weise durch beide Pipelines laufen und Metadaten generieren sollen
      test_audio/          # Audios für Einzeltests können hier abgelegt werden
      normalised_audio/    # konvertierte Audios im Format WAV 16kHz Mono
    results/
      <file_stem>/         # pro Audio ein Ordner: Transcript, Outputs, Meta
    src/                   # optional (wenn Python-Code später ausgelagert werden soll)
    01_pipeline.ipynb      # Hauptnotebook
    02_analyse.ipynb       # optionales Notebook zur Metadaten-Auswertung, Statistik-Erstellung, etc. - Noch nicht implementiert

---

## Voraussetzungen

- **Python 3.10+**
- **ffmpeg** (für Audio-Normalisierung)
- **Hugging Face Token** für Pyannote-Modelle (Model Terms auf Hugging Face akzeptieren und eigenen Token erstellen):
https://huggingface.co/pyannote/speaker-diarization-community-1
- Ein laufender **vLLM**-Server zur Nutzung der LLM-Pipeline genutzt werden soll

### ffmpeg installieren (Beispiele)
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`
- macOS (brew): `brew install ffmpeg`

---

## Setup (Quickstart)

1) Virtuelle Umgebung + Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Ordner anlegen

```bash
mkdir -p data/input_audio data/test_audio data/normalised_audio results src
```

3) `.env` erstellen

```bash
touch .env
```

---

## .env Konfiguration

Beispiel (bitte die individuelle Konfiguration anpassen):

```bash
# vLLM / LLM
VLLM_BASE_URL=http://127.0.0.1:8001/v1
VLLM_MODEL=google/medgemma-27b-it

# Repo
REPO_ROOT=~/jupyter/diarization-benchmark
TEST_AUDIO=data/test_audio/test2.mp3

# Pyannote
PYANNOTE_MODEL_ID=pyannote/speaker-diarization-3.1
HUGGINGFACE_TOKEN=hf_xxx
```

**Wichtige Hinweise**
- `VLLM_BASE_URL` sollte **inkl. `/v1`** gesetzt werden
- `TEST_AUDIO` ist **relativ zu `REPO_ROOT`** (wie oben) oder ein absoluter Pfad
- `.env` bitte **nicht committen** aufgrund sensibler Daten (HF-Token), s. .gitignore

---

## Nutzung

### Einzeltests

Im Notebook gibt es Schalter wie `llm_test` / `pyannote_test`, die Test-Zellen bei Bedarf aktivieren.
Per Default auf False lassen, damit Run All nicht unnötig lange dauert.

### Batch-Run

1) Audios nach `data/input_audio/` legen  
2) Batch-Zellen im Notebook ausführen  
3) Für jede Datei entsteht ein Ordner unter `results/<filename_stem>/` mit:

- `transcript.txt` – Transkript (Whisper)
- `llm_diarized.txt` – LLM-Ausgabe (Dialog)
- `pyannote_diarized.txt` – Pyannote-Ausgabe (Dialog, via Mapping)
- `meta.json` – Meta (Model-IDs, Status, Laufzeiten)
- `whisper_segments.json`, `whisper_info.json` für Debugging-Infos oder Details zur Transkription

Zusätzlich schreibt der Batch-Run sämtliche Meta-Informationen zu Auswertungszwecken in:
- `results/batch_summary.json` (**wird bei erneutem Run überschrieben**)

---

## Output / Laufzeitmessung

In `meta.json` wird pro Datei u.a. gespeichert:

- Modell-IDs (Whisper / LLM / Pyannote)
- Status pro Pipeline (ok/error + Fehlermeldung)
- Laufzeiten (Sekunden):
  - `normalize` (Audio → WAV 16k mono)
  - `transcribe` (faster-whisper)
  - `llm` (LLM-Diarisierung)
  - `pyannote` (Pyannote Inferenz)
  - `postproc` (Mapping/Formatierung)
  - `total` (gesamt)

Damit lassen sich Laufzeitvergleiche und Fail-Analysen reproduzierbar auswerten.

---

## Methodische Hinweise und Limitationen

### Segment-basiertes Pyannote→Whisper Mapping (Post-Processing)
Das Mapping ordnet **Whisper-Segmente** per maximalem Zeit-Overlap einem Speaker zu.  
Das ist transparent und reproduzierbar, kann aber in diesen Situationen ungenau sein:

- Whisper-Segmente sind **lang** und Sprecherwechsel passieren **innerhalb** eines Segments  
- sehr kurze Einwürfe („ja“, „mh“) innerhalb längerer Segmente  
- sehr ähnliche Stimmen / geringe akustische Trennbarkeit

Für mehr Präzision lässt sich word-basiertes Mapping verwenden, was allerdings aufwändigeres Post-Processing und Änderungen bei der Transkription mit sich zieht.

### Dateinamen-Kollisionen
Wenn mehrere Audios den gleichen **Dateinamen ohne Endung** haben (z.B. `session1.mp3` in zwei Unterordnern), werden Ergebnisse überschrieben, da `results/<stem>/` genutzt wird.

Empfehlung: Test-Dateien eindeutig benennen.

### Pyannote
Pyannote läuft per Default auf der CPU. GPU-Integration ist möglich und kann die Laufzeit deutlich beschleunigen.
