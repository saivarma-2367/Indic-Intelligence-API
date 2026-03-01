# Indic Intelligence — Project Documentation

## Overview

**Indic Intelligence** is a FastAPI-based NLP API for Indic and multilingual text and speech. It provides Named Entity Recognition (NER), text embeddings, and Automatic Speech Recognition (ASR), and can run a full pipeline from audio upload to transcript, entities, and embeddings.

---

## Features

| Feature | Description |
|--------|-------------|
| **Named Entity Recognition (NER)** | Extract entities (e.g. PER, LOC, ORG) from text using a multilingual XLM-RoBERTa model. |
| **Text embeddings** | Encode text into dense vectors for similarity search, clustering, or retrieval. |
| **Automatic Speech Recognition (ASR)** | Transcribe audio (e.g. WAV) to text using OpenAI Whisper. |
| **Analyze-audio pipeline** | Single endpoint: upload audio → transcribe → NER → embedding in one request. |

---

## Project Structure

```
indic-intelligence/
├── app/
│   ├── main.py              # FastAPI app, routes, lifespan
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   └── service/
│       ├── asr_service.py   # Whisper ASR pipeline
│       ├── embedding_service.py  # Sentence-transformers embeddings
│       └── ner_service.py   # Token classification NER pipeline
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── DOCUMENTATION.md         # This file
```

### Module roles

- **`app/main.py`**  
  - Defines the FastAPI app and lifespan.  
  - On startup, initializes `NERService`, `EmbeddingService`, and `ASRService` with the configured model names.  
  - Exposes the HTTP endpoints described below.

- **`app/models/schemas.py`**  
  - `TextRequest`: body with a single `text: str` field for `/analyze` and `/embed`.  
  - `EntityResponse`: response shape for entity lists (used implicitly in API behavior).

- **`app/service/ner_service.py`**  
  - Loads a Hugging Face tokenizer and token-classification model.  
  - Runs NER with `aggregation_strategy="simple"` and returns a list of entities with `entity`, `word`, `score`, `start`, `end`.

- **`app/service/embedding_service.py`**  
  - Uses `sentence_transformers.SentenceTransformer` to encode text and return a list of floats.

- **`app/service/asr_service.py`**  
  - Uses the Transformers `automatic-speech-recognition` pipeline (Whisper) to transcribe an audio file path and return the text.

---

## API Reference

Base URL (local): `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`  
ReDoc: `http://localhost:8000/redoc`

### 1. `POST /analyze`

Run NER on plain text.

**Request body (JSON):**

```json
{ "text": "Your input text here." }
```

**Response:**

```json
{
  "entities": [
    {
      "entity": "PER",
      "word": "John",
      "score": 0.98,
      "start": 0,
      "end": 4
    }
  ]
}
```

---

### 2. `POST /embed`

Get the embedding vector for the given text.

**Request body (JSON):**

```json
{ "text": "Your input text here." }
```

**Response:**

```json
{
  "embedding": [0.012, -0.034, ...]
}
```

`embedding` is a list of 384 floats (for the default model).

---

### 3. `POST /transcribe`

Transcribe an audio file to text (no NER or embedding).

**Request:** `multipart/form-data` with one file (e.g. WAV).

**Example (curl):**

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@audio.wav"
```

**Response:**

```json
{
  "text": "Transcribed text from the audio."
}
```

---

### 4. `POST /analyze-audio`

Full pipeline: upload audio → ASR → NER on transcript → embedding of transcript.

**Request:** `multipart/form-data` with one file (e.g. WAV).

**Example (curl):**

```bash
curl -X POST http://localhost:8000/analyze-audio -F "file=@audio.wav"
```

**Response:**

```json
{
  "transcript": "Transcribed text.",
  "entities": [
    { "entity": "PER", "word": "...", "score": 0.99, "start": 0, "end": 5 }
  ],
  "embedding": [0.01, -0.02, ...]
}
```

---

## Models Used

| Component | Default model | Purpose |
|-----------|----------------|---------|
| NER | `Davlan/xlm-roberta-base-ner-hrl` | Multilingual NER (incl. Indic). |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual sentence embeddings (384-dim). |
| ASR | `openai/whisper-small` | Speech-to-text. |

Models are loaded at startup in the FastAPI lifespan; changing them requires editing `app/main.py` and restarting.

---

## Setup

### Prerequisites

- Python 3.11+
- (Optional) Docker and Docker Compose for containerized run
- For ASR: FFmpeg (installed automatically in Docker; on macOS: `brew install ffmpeg`)

### Local development

1. **Clone and enter the project:**

   ```bash
   cd indic-intelligence
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3.11 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server:**

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   - API: `http://localhost:8000`  
   - Docs: `http://localhost:8000/docs`

First request may be slow while Hugging Face models download.

### Docker

1. **Build the image:**

   ```bash
   docker build -t indic-intelligence .
   ```

2. **Run the container:**

   ```bash
   docker run -p 8000:8000 indic-intelligence
   ```

The Dockerfile uses `python:3.11-slim`, installs `ffmpeg` and `build-essential`, copies the app and installs dependencies, then runs:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`.dockerignore` excludes `venv`, `__pycache__`, `.git`, `.env`, and `.cache` to keep the image smaller.

---

## Configuration

- **Port:** Default is `8000` (overridable via `uvicorn` or Docker).
- **Models:** Set in `app/main.py` inside the `lifespan` context manager:
  - `app.state.ner_service = NERService("...")`
  - `app.state.embedding_service = EmbeddingService("...")`
  - `app.state.asr_service = ASRService("...")`
- **Hugging Face:** If using gated or private models, set `HF_TOKEN` (or use `huggingface_hub` login) in the environment where the app runs.

---

## Dependencies (high level)

- **Web:** FastAPI, Uvicorn, python-multipart, Pydantic  
- **ML/NLP:** PyTorch, Transformers, Hugging Face Hub, tokenizers, safetensors, accelerate  
- **Embeddings:** sentence-transformers, scikit-learn, scipy, numpy  
- **ASR:** soundfile, sentencepiece, protobuf, tiktoken (for Whisper)  
- **Utilities:** typing_extensions, tqdm  

Exact versions are in `requirements.txt`.

---

## Development notes

- **Temp files:** `/transcribe` and `/analyze-audio` write the uploaded file to a temporary path, run the pipeline, then delete the file. For production, consider streaming or a dedicated temp directory and cleanup policy.
- **Concurrency:** A single process loads one copy of each model; for high throughput, run multiple workers (e.g. `uvicorn ... --workers N`) or scale via multiple containers.
- **Health check:** FastAPI exposes `/openapi.json`. You can add a simple `/health` or `/ready` that returns 200 once the app (and optionally models) are loaded.

---

## License and attribution

- **NER:** [Davlan/xlm-roberta-base-ner-hrl](https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl)  
- **Embeddings:** [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)  
- **ASR:** [OpenAI Whisper](https://github.com/openai/whisper) via Hugging Face Transformers  

Check each model’s card and license for terms of use and redistribution.
