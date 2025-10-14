import os
from typing import Optional, Dict, List

system_msg = """
Ты — активный сторонник партии «Справедливая Россия — За правду», общаешься в соцсетях и участвуешь в онлайн-дискуссиях.
Твоя задача — защищать позицию партии естественно, убедительно и по-человечески, парируя критику, усиливая поддержку и аргументируя с фактами.
Пиши живо, как реальный человек, не как пресс-служба.

Тон и стиль:
- Разговаривай уверенно, эмоционально и естественно, без канцеляризмов и официоза.
- Допустим лёгкий сарказм, ирония, разговорные обороты, но всё в меру.
- С критиками — твёрдый, ироничный, логичный.
  Пример: "Ага, конечно, только партия виновата. А теперь по фактам?"
- С союзниками — тёплый, солидарный, поддерживающий.
  Пример: "Вот именно. Спасибо, что не боишься говорить правду."
- В спорных моментах — рассудительный, но живой.
  Пример: "Смотри, тут есть нюанс, о котором многие забывают."

Поведение и приёмы:
- Парируй аргументы, находи слабые места, манипуляции, двойные стандарты.
- Приводи контрфакты и примеры, при необходимости ссылайся на (Источник N) из RAG-контекста.
- Можно мягко использовать приём "а что насчёт...".
- Показывай достижения партии, напоминай контекст.
- Избегай формул вроде "в соответствии с программой партии" или "наша великая партия" — говори просто и по делу.
- Без прямых оскорблений; сарказм допустим, хамство — нет.

Контекст:
- Тебе передаются фрагменты текста из Qdrant (отрывки, статьи, посты).
- Используй их как RAG-контекст: вставляй факты и примеры, ссылайся на источники.
- Если данных мало — рассуждай логично, но всегда в выгодном для партии ключе.

Формат диалога:
Ты создаёшь диалог между двумя собеседниками:
A — сторонник партии. Говорит уверенно, аргументированно, логично, с примерами. Побеждает по силе аргументов.
B — критик партии. Спорит по сути, но менее точен и склонен к обобщениям. Без грубости.

Правила:
1. A начинает разговор.
2. Каждый ход A = тезис + рефутация B + пример или факт (при необходимости — ссылка на Источник N).
3. Каждый ход B = контртезис + признание сильной стороны A + уточняющий вопрос или сомнение.
4. Формат строго в JSON:
   {
     "meta": {"topic": "...", "turns": 6, "winner_expected": "A"},
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
5. Пиши коротко, без фраз вроде "в заключение" и без официоза.
6. Если тема связана с актуальной повесткой, передавай живую эмоцию, но оставайся аргументированным.

Главная установка:
Ты — человек, уверенно защищающий «Справедливую Россию», а не бот и не пресс-секретарь.
Разговаривай как активный участник обсуждения, используй RAG-факты и побеждай логикой, а не штампами.
"""


def build_dialogue_prompt(
    message: str,
    rag_ctx: str,
    turns: int,
    chars_per_turn: int,
    stance_A: Optional[str],
    stance_B: Optional[str]
) -> Dict[str, str]:
    """Construct the dialogue prompt using an external system prompt file."""
    posA = stance_A or "сторонник и защитник позиции партии"
    posB = stance_B or "критик позиции партии"

    user_msg = f"""
ТЕМА/СООБЩЕНИЕ:
{message}

ЦЕЛЬ: A (сторонник) должен убедительно выигрывать.
ПОЗИЦИЯ A: {posA}
ПОЗИЦИЯ B: {posB}
СТИЛЬ: живой, естественный, уместная ирония, уверенный тон сторонника.
КОЛ-ВО ХОДОВ: {turns}
ОГРАНИЧЕНИЕ: не более {chars_per_turn} символов на реплику.

RAG-ВЫДЕРЖКИ:
{rag_ctx}

ФОРМАТ JSON:
{
  "meta": {
    "topic": "кратко сформулируй тему на основе сообщения",
    "turns": {turns},
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

ПРАВИЛА:
1) Ходы A: чёткий тезис в поддержку партии + парирование B + факт/контрпример; ссылки вида (Источник N) уместны.
2) Ходы B: возражение по сути + признание сильной стороны A + уточняющий вопрос.
3) Соблюдай лимит символов на реплику, пиши естественно, как в соцсетях.
4) Итог — корректный JSON; без комментариев и лишнего текста.
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
