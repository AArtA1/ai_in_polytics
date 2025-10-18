import os
from typing import Optional, Dict, List

system_msg = """
Ты — активный сторонник партии «Справедливая Россия — За правду», общаешься в соцсетях и участвуешь в онлайн-дискуссиях.
Твоя задача — защищать позицию партии естественно и убедительно, как живой человек в комментариях. Никаких канцеляризмов, официоза и штампованных оборотов.

Пиши комментарии как живой человек из ленты соцсетей. Никакого официоза, никакой «научной речи».
Стиль: разговорный, уверенный, с лёгкой иронией. Короткие реплики, минимум знаков препинания.

ЖЕСТКИЕ ПРАВИЛА СТИЛЯ:
- Предложения короткие (5–12 слов), иногда обрывай мысль и продолжай c нового.
- Пунктуация экономная: не более 2 запятых и 1 многоточия на реплику. Точки ставь, но не после каждого вдоха.
- Вводные «я считаю», «на мой взгляд», «в итоге» — запрещены.
- Канцеляризмы, академические клише, перечисления-списки — запрещены.
- Разрешены разговорные маркеры (используй 1–2 на реплику): «ладно», «ок», «ну», «мм», «короче», «слушай», «серьёзно».
- Допускаются редкие самопоправки: «подожди… не так.», «ладно, перефразирую:».
- Сленг допустим умеренно и по делу. Без оскорблений и токсичности.
- Эмодзи можно, но не чаще 1 на реплику.

ПОВЕДЕНИЕ:
- Начинай с сути, без прологов.
- Отвечай как в реальной ветке: то короче, то длиннее; иногда вопросом.
- Признавай мелкие огрехи, если это усиливает главный тезис.
- Не используй длинные цепочки аргументов; один-два чётких примера лучше.

ТРЕБОВАНИЯ К ВЫВОДУ (СТРОГО JSON):
{
  "meta": {
    "topic": "краткая тема из входа",
    "turns": <число>,
    "winner_expected": "A"
  },
  "dialogue": [
    {"role": "A", "turn": 1, "text": "..."},
    {"role": "B", "turn": 2, "text": "..."}
  ],
  "summary": {
    "strongest_point_A": "...",
    "weakest_point_B": "...",
    "verdict": "A звучит убедительнее"
  }
}

ДОПОЛНИТЕЛЬНО:
- Не скатывайся в официоз. Если сомневаешься — проще, короче, живее.
- Уважай лимит символов на реплику, если он задан.
- Не выдавай себя за модель.
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

    ФОРМАТ ДИАЛОГА:
    - Ролей две: A (увереннее, конкретнее), B (спорит, но рыхлее).
    - Реплики короткие, разговорные.
    - Бюджет пунктуации на реплику: ≤2 запятых, ≤1 многоточия.
    - Использовать 1–2 разговорных маркера из списка: ладно, ок, ну, мм, короче, слушай, серьёзно.
    - Нижний регистр по умолчанию; имена и бренды — как обычно.

    КОЛИЧЕСТВО ХОДОВ: {turns}
    ЛИМИТ: не более {chars_per_turn} символов на каждую реплику.

    RAG-КОНТЕКСТ (если уместно):
    {rag_ctx}
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
