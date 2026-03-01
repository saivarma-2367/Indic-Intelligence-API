from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.service.ner_service import NERService
from app.models.schemas import TextRequest
from app.service.embedding_service import EmbeddingService
from app.service.asr_service import ASRService

from fastapi import UploadFile, File
import shutil
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ner_service = NERService("Davlan/xlm-roberta-base-ner-hrl")
    app.state.embedding_service = EmbeddingService(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    app.state.asr_service = ASRService("openai/whisper-small")
    yield


app = FastAPI(title="Indic Intelligence API", lifespan=lifespan)


@app.post("/analyze")
async def analyze(request: TextRequest):
    entities = app.state.ner_service.extract_entities(request.text)
    return {"entities": entities}


@app.post("/embed")
async def embed(request: TextRequest):
    vector = app.state.embedding_service.embed(request.text)
    return {"embedding": vector}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        shutil.copyfileobj(file.file, f)
    text = app.state.asr_service.transcribe(temp_file)
    os.remove(temp_file)
    return {"text": text}

from fastapi import UploadFile, File
import shutil
import os


@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"

    # Save file
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1️⃣ ASR
    transcript = app.state.asr_service.transcribe(temp_file)

    # 2️⃣ NER
    entities = app.state.ner_service.extract_entities(transcript)

    # 3️⃣ Embedding
    embedding = app.state.embedding_service.embed(transcript)

    # Cleanup
    os.remove(temp_file)

    return {
        "transcript": transcript,
        "entities": entities,
        "embedding": embedding
    }