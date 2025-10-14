# src/app.py
"""
FastAPI сервис для генерации диалогов с использованием RAG.
Содержит только API endpoints и конфигурацию.
"""
import os
import json
from typing import List, Dict, Literal, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from src.generate_dialogue import (
    build_dialogue_prompt,
    retrieve,
    build_rag_context,
    call_openai,
)

# ===================== Настройки =====================
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_store")
COLLECTION   = os.getenv("QDRANT_COLLECTION", "spravedlivo_ru")
EMB_MODEL    = os.getenv("EMB_MODEL", "BAAI/bge-m3")
TOP_K        = int(os.getenv("TOP_K", "6"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Получаем из .env файла
LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMP     = float(os.getenv("LLM_TEMP", "0.7"))

# ===================== FastAPI =====================
app = FastAPI(title="RAG Dialogue API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Жизненный цикл ----
@app.on_event("startup")
async def startup():
    """Инициализация ресурсов при запуске сервера."""
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI

    # Qdrant
    if QDRANT_PATH:
        qdrant = QdrantClient(path=QDRANT_PATH)
    else:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Модель эмбеддингов
    emb_model = SentenceTransformer(EMB_MODEL)

    # OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY не задан в окружении.")
    oa_client = OpenAI(api_key=OPENAI_API_KEY)

    # Сохраняем в app.state
    app.state.qdrant = qdrant
    app.state.emb_model = emb_model
    app.state.oa_client = oa_client

@app.on_event("shutdown")
async def shutdown():
    """Корректное закрытие ресурсов при остановке сервера."""
    qdrant = getattr(app.state, "qdrant", None)
    try:
        if qdrant and hasattr(qdrant, "close"):
            qdrant.close()
    except Exception:
        pass

# ---- Входные модели ----
Mode = Literal["dialogue"]

class GenerateRequest(BaseModel):
    message: str = Field(..., description="Описание темы/кейса/поста/диалога — свободный текст.")
    mode: Mode = Field("dialogue", description="Что генерируем")
    top_k: int = Field(TOP_K, ge=1, le=20)
    # Параметры диалога
    turns: int = Field(6, ge=2, description="Чётное число ходов (A начинает)")
    chars_per_turn: int = Field(400, ge=120, le=1200)
    stance_A: Optional[str] = Field(None, description="Позиция A (опционально)")
    stance_B: Optional[str] = Field(None, description="Позиция B (опционально)")

class Health(BaseModel):
    status: str

# ===================== API Endpoints =====================

@app.get("/health", response_model=Health)
def health():
    """Проверка состояния сервиса."""
    return {"status": "ok"}

@app.post("/retrieve")
def retrieve_endpoint(req: GenerateRequest, request: Request):
    """Поиск релевантных документов в Qdrant."""
    hits = retrieve(
        request.app.state.qdrant,
        request.app.state.emb_model,
        req.message,
        collection=COLLECTION,
        k=req.top_k
    )
    return {"query": req.message, "results": hits}

@app.post("/rag_generate")
def rag_generate(req: GenerateRequest, request: Request):
    """Генерация диалога с использованием RAG."""
    qdrant = request.app.state.qdrant
    emb_model = request.app.state.emb_model
    oa_client = request.app.state.oa_client

    # 1) Поиск релевантных чанков
    hits = retrieve(qdrant, emb_model, req.message, collection=COLLECTION, k=req.top_k)
    rag_ctx = build_rag_context(hits) if hits else "(нет подходящих отрывков)"

    # 2) Построение промпта
    if req.mode == "dialogue":
        prompt = build_dialogue_prompt(
            message=req.message,
            rag_ctx=rag_ctx,
            turns=req.turns,
            chars_per_turn=req.chars_per_turn,
            stance_A=req.stance_A,
            stance_B=req.stance_B,
        )
        expect_json = True
    else:
        return {"error": f"Unknown mode: {req.mode}"}

    # 3) Вызов LLM
    content = call_openai(
        oa_client,
        prompt["system"],
        prompt["user"],
        model=LLM_MODEL,
        temperature=LLM_TEMP,
        expect_json=expect_json
    )

    # 4) Парсинг результата
    if expect_json:
        try:
            out = json.loads(content)
        except Exception:
            out = {"raw": content, "note": "Модель вернула не-JSON. Проверь промпт/response_format."}
    else:
        out = {"text": content}

    return {
        "mode": req.mode,
        "query": req.message,
        "rag_used": hits,
        "result": out
    }

# ===================== Запуск =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)

