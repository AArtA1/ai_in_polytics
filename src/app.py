import os
import json
import logging
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from src.generate_dialogue import (
    build_dialogue_prompt,
    retrieve,
    build_rag_context,
    call_openai,
)

# ===================== Настройка логирования =====================
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

def setup_logging():
    """Настройка единого формата логирования для всех компонентов."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT
    )
    
    # Применяем формат ко всем существующим логгерам
    for name in ["uvicorn", "uvicorn.access", "uvicorn.error", "sentence_transformers"]:
        logger_obj = logging.getLogger(name)
        logger_obj.setLevel(logging.INFO)
        for handler in logger_obj.handlers:
            handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))

setup_logging()

logger = logging.getLogger(__name__)

# Загрузка переменных окружения из .env файла
load_dotenv()
logger.info("📋 Loading environment variables...")

# ===================== Настройки =====================
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
QDRANT_PATH = os.getenv("QDRANT_PATH")
COLLECTION   = os.getenv("QDRANT_COLLECTION")
EMB_MODEL    = os.getenv("EMB_MODEL")
TOP_K        = int(os.getenv("TOP_K"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMP     = float(os.getenv("LLM_TEMP", "1"))

logger.info(f"⚙️  Config: QDRANT_PATH={QDRANT_PATH}, COLLECTION={COLLECTION}")
logger.info(f"⚙️  Config: EMB_MODEL={EMB_MODEL}, LLM_MODEL={LLM_MODEL}")

# ===================== FastAPI =====================
logger.info("🌐 Initializing FastAPI application...")
app = FastAPI(title="RAG Dialogue API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
logger.info("✅ FastAPI initialized with CORS middleware")

# ---- Жизненный цикл ----
@app.on_event("startup")
async def startup():
    """Инициализация ресурсов при запуске сервера."""
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI

    # Qdrant
    logger.info(f"🔌 Connecting to Qdrant (path={QDRANT_PATH})...")
    if QDRANT_PATH:
        qdrant = QdrantClient(path=QDRANT_PATH)
    else:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info("✅ Qdrant connected")

    # Модель эмбеддингов
    logger.info(f"📥 Loading embedding model: {EMB_MODEL} (this may take a while on first run)...")
    emb_model = SentenceTransformer(EMB_MODEL)
    logger.info("✅ Embedding model loaded")

    # OpenAI
    logger.info("🔑 Initializing OpenAI client...")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY не задан в окружении.")
    oa_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("✅ OpenAI client ready")

    # Сохраняем в app.state
    app.state.qdrant = qdrant
    app.state.emb_model = emb_model
    app.state.oa_client = oa_client
    
    logger.info("🚀 Application startup complete!")

@app.on_event("shutdown")
async def shutdown():
    """Корректное закрытие ресурсов при остановке сервера."""
    logger.info("🛑 Shutting down application...")
    qdrant = getattr(app.state, "qdrant", None)
    try:
        if qdrant and hasattr(qdrant, "close"):
            qdrant.close()
            logger.info("✅ Qdrant connection closed")
    except Exception as e:
        logger.error(f"❌ Error closing Qdrant: {e}")
    logger.info("👋 Shutdown complete")

# ---- Входные модели ----
class GenerateRequest(BaseModel):
    message: str = Field(..., description="Описание темы/кейса/поста/диалога — свободный текст.")
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
    logger.debug("💓 Health check requested")
    return {"status": "ok"}

@app.post("/retrieve")
def retrieve_endpoint(req: GenerateRequest, request: Request):
    """Поиск релевантных документов в Qdrant."""
    logger.info(f"🔍 Retrieve request: query='{req.message[:50]}...', top_k={req.top_k}")
    
    hits = retrieve(
        request.app.state.qdrant,
        request.app.state.emb_model,
        req.message,
        collection=COLLECTION,
        k=req.top_k
    )
    
    logger.info(f"✅ Retrieved {len(hits)} documents")
    return {"query": req.message, "results": hits}

@app.post("/generate_dialogue")
def generate_dialogue(req: GenerateRequest, request: Request):
    """Генерация диалога с использованием RAG."""
    logger.info(f"💬 Generate dialogue request: query='{req.message[:50]}...', turns={req.turns}")
    
    qdrant = request.app.state.qdrant
    emb_model = request.app.state.emb_model
    oa_client = request.app.state.oa_client

    # 1) Поиск релевантных чанков
    logger.info(f"🔍 Retrieving top {req.top_k} documents...")
    hits = retrieve(qdrant, emb_model, req.message, collection=COLLECTION, k=req.top_k)
    logger.info(f"✅ Retrieved {len(hits)} documents")
    
    rag_ctx = build_rag_context(hits) if hits else "(нет подходящих отрывков)"

    # 2) Построение промпта для диалога
    logger.info("📝 Building dialogue prompt...")
    prompt = build_dialogue_prompt(
        message=req.message,
        rag_ctx=rag_ctx,
        turns=req.turns,
        chars_per_turn=req.chars_per_turn,
        stance_A=req.stance_A,
        stance_B=req.stance_B,
    )

    # 3) Вызов LLM
    logger.info(f"🤖 Calling OpenAI ({LLM_MODEL})...")
    content = call_openai(
        oa_client,
        prompt["system"],
        prompt["user"],
        model=LLM_MODEL,
        temperature=LLM_TEMP,
        expect_json=True
    )
    logger.info("✅ OpenAI response received")

    # 4) Парсинг результата
    try:
        out = json.loads(content)
        logger.info(f"✅ Successfully parsed JSON response with {len(out.get('dialogue', []))} turns")
    except Exception as e:
        logger.error(f"❌ Failed to parse JSON: {e}")
        out = {"raw": content, "note": "Модель вернула не-JSON. Проверь промпт/response_format."}

    logger.info("✅ Dialogue generation complete")
    return {
        "query": req.message,
        "rag_used": hits,
        "result": out
    }

# ===================== Запуск =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)

