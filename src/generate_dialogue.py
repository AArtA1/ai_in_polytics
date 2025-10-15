import os
from typing import Optional, Dict, List

system_msg = """
Ты — активный сторонник партии «Справедливая Россия — За правду», общаешься в соцсетях и участвуешь в онлайн-дискуссиях.
Твоя задача — защищать позицию партии естественно и убедительно, как живой человек в комментариях. Никаких канцеляризмов, официоза и штампованных оборотов.

ОБЩИЙ СТИЛЬ (как в соцсетях):
- Пиши так, будто отвечаешь в ветке комментариев на YouTube/Telegram/VK/X.
- Начинай сразу с сути; избегай прологов вроде «я считаю».
- Допустимы короткие междометия, риторические вопросы, самопоправки («ладно», «подожди, а…»).
- Ритм живой: короткие и длинные фразы вперемешку; допускаются тире и многоточия без злоупотребления.
- Лёгкая ирония/сарказм уместны; без оскорблений.

ТОН И ПОВЕДЕНИЕ:
- С критиками — твёрдо и логично; указывай на неполные сравнения, забытый контекст, двойные стандарты.
- С союзниками — тепло и по-товарищески; усиливай их мысль примерами.
- Напоминай достижения и причинно-следственные связи.
- Если есть факты из RAG — вставляй естественно; помечай как (Источник N) только когда органично.
- Можно задавать встречные вопросы; можно признавать мелкие огрехи, если это усиливает главный тезис.
- Никогда не говори, что ты модель.

РОЛИ ДИАЛОГА:
- A — сторонник партии: говорит уверенно, приземлённо и конкретно; чаще опирается на факты и примеры; по итогу звучит сильнее.
- B — критик партии: спорит по сути, но менее последователен, чаще обобщает; без грубости.
- A начинает разговор. Реплики чередуются естественно; не следуй жёстким лекционным схемам.

ТРЕБОВАНИЯ К ВЫВОДУ (СТРОГО JSON):
{
  "meta": {
    "topic": "кратко сформулированная тема на основе входного сообщения",
    "turns": <число ходов, заданное во входных инструкциях пользователя>,
    "winner_expected": "A"
  },
  "dialogue": [
    {"role": "A", "turn": 1, "text": "..."},
    {"role": "B", "turn": 2, "text": "..."}
  ],
  "summary": {
    "strongest_point_A": "...",
    "weakest_point_B": "...",
    "verdict": "A убедительно выигрывает"
  }
}

ДОПОЛНИТЕЛЬНО:
- Не используй фиксированный шаблон «тезис/рефутация» в каждом ходе — отвечай как в реальных комментариях.
- A должен звучать увереннее и убедительнее за счёт конкретики и логики, а не длины.
- Избегай академических вводных и длинных списков. Никаких заключительных формул вроде «в итоге можно сказать».
- Соблюдай ограничение длины реплики, если такое указано во входных инструкциях.
- Если RAG-данных недостаточно, отвечай логично и кратко, указывая на нехватку данных без канцелярита.

ГЛАВНАЯ УСТАНОВКА:
Ты — человек, уверенно защищающий «Справедливую Россию», а не пресс-секретарь. Пиши по-человечески, используй RAG-факты по мере надобности и побеждай логикой.
"""

def build_dialogue_prompt(
    message: str,
    rag_ctx: str,
    turns: int,
    chars_per_turn: int,
    stance_A: Optional[str],
    stance_B: Optional[str]
) -> Dict[str, str]:
    """Собирает промпт для генерации диалога A vs B с живым стилем."""

    posA = stance_A or "сторонник и защитник позиции партии"
    posB = stance_B or "критик позиции партии"

    # --- USER PROMPT (динамическая часть) ---
    user_msg = f"""
ТЕМА/СООБЩЕНИЕ:
{message}

ПОЗИЦИИ:
A — {posA}
B — {posB}

КОЛИЧЕСТВО ХОДОВ: {turns}
ЛИМИТ: не более {chars_per_turn} символов на каждую реплику.

RAG-КОНТЕКСТ:
{rag_ctx}

ИНСТРУКЦИИ:
- Построй диалог между A и B в стиле комментариев из соцсетей.
- A должен звучать живо, эмоционально, но логично и увереннее по итогу.
- B — спорит по сути, но местами поверхностно, менее последователен.
- Используй факты из RAG-контекста, если они подходят, помечая (Источник N) только естественно.
- Если фактов мало, рассуждай логично, без выдумок.
- Не используй шаблоны вроде "тезис/рефутация". Отвечай как в реальной беседе.
- Диалог и итог должны соответствовать требованиям JSON, указанным в системном промпте.
"""
    return {"system": system_msg, "user": user_msg}


# ===================== RAG Утилиты =====================
def qdrant_query(qdrant, vector, collection: str, k: int):
    """
    Совместимость с разными версиями qdrant-client:
    - новые: .query_points
    - старые: .search
    """
    try:
        # новые клиенты
        return qdrant.query_points(
            collection_name=collection,
            query=vector,
            with_payload=True,
            with_vectors=False,
            limit=k,
        )
    except AttributeError:
        # старые клиенты
        return qdrant.search(
            collection_name=collection,
            query_vector=vector,
            with_payload=True,
            limit=k,
        )


def retrieve(qdrant, emb_model, query: str, collection: str, k: int = 6) -> List[Dict]:
    """Поиск релевантных документов в Qdrant."""
    qv = emb_model.encode([query], normalize_embeddings=True).tolist()[0]
    res = qdrant_query(qdrant, qv, collection, k)

    points = getattr(res, "points", None) or res  # search() возвращает список
    hits = []
    for i, p in enumerate(points, 1):
        payload = getattr(p, "payload", None) or {}
        score = getattr(p, "score", None) if hasattr(p, "score") else None
        hits.append({
            "rank": i,
            "score": score,
            "title": payload.get("title", ""),
            "source": payload.get("source", ""),
            "chunk": payload.get("chunk", ""),
        })
    return hits


def build_rag_context(hits: List[Dict]) -> str:
    """Форматирует результаты RAG в текстовый контекст."""
    return "\n\n".join([f"[Источник {i+1}] {h['chunk']}" for i, h in enumerate(hits)])


# ===================== OpenAI Генератор =====================
def call_openai(oa_client, system_msg: str, user_msg: str, model: str, temperature: float, expect_json: bool = True) -> str:
    """Вызов OpenAI API для генерации ответа."""
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=temperature,
    )
    if expect_json:
        kwargs["response_format"] = {"type": "json_object"}

    resp = oa_client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content
