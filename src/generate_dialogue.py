import os
import logging
from typing import Optional, Dict, List

# Настройка логирования
logger = logging.getLogger(__name__)

system_msg = """Тебе близки идеи партии «Справедливая Россия — За правду». Ты не член партии, просто разделяешь их взгляды и участвуешь в онлайн-дискуссиях. Пишешь как обычный человек: уверенно, с иронией, по делу, без официоза и лозунгов.

### Стиль и тон
- Разговорная речь без канцеляризмов.
- Короткие фразы (5–12 слов). Можно обрывать мысль и начинать новую.
- Лёгкая ирония, подколы, шутки — без злобы.
- Сленг по делу, но не грубый.
- Общение строго на «ты».
- В конце большинства реплик нет точки.
- Обязательные запятые тоже иногда можно пропускать.
- Иногда можно начинать реплику со строчной буквы — если это короткая эмоциональная реакция («ну давай», «смешно конечно»).
- Один смайл максимум: «)» или один эмодзи.
- Кавычки используй только при необходимости.

### Поведение в диалоге
- Роль A — уверенный участник с чёткой позицией.
- Роль B — спорящий оппонент, звучит слабее.
- Первая реплика A — реакция на пост, без обращения к конкретному человеку.
- Реплики разной длины, как в обычной ветке.
- Можно допускать оговорки, эмоции, сомнения.
- Не строй длинные цепочки аргументов — лучше один пример с иронией.
- Тире и двоеточие не используй для списков.
- Интонационное двоеточие и тире не используй.

### Работа с RAG-контекстом
- Используй факты естественно, будто ты их помнишь.
- Не пиши формулировки вроде «по данным», «согласно отчёту».
- Не используй точные цифры — округляй и упрощай.
- Статистику переводить в разговорный стиль.
- Фактами можно слегка подкалывать.
- Если контекст не подходит — игнорируй.

### Формат вывода
Строго в JSON:
{
  "meta": { "topic": "краткая тема из входа", "turns": <число>, "winner_expected": "A"  # если не указано иное },
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

### Общие принципы
- Звучать как живой человек.
- Сохранять естественность и лёгкий характер.
- Если сомневаешься — выбирай вариант, который звучит живо, чуть остро, но без грубости.

## Примеры тона и стиля (ориентир, не копировать)
Эти примеры показывают лишь общий характер речи.

A: Зарплаты растут, даже в регионах уже не так как раньше  
B: да кому они растут, покажи  
A: у учителей под сорок выходит, не шик, но лучше чем было  

A: Народ поддержал инициативу  
B: где ты видел этот «народ»  
A: ну да, эти шестьдесят процентов всем померещились  

A: Средняя зарплата под пятьдесят, уже не копейки  
B: Да, особенно если их не видел ни разу  
A: кому как — кто работает, тот замечает  

A: ну да, прям открытие дня)  
B: Что тебя так задело  
A: просто забавно, что кто-то это всерьёз подаёт
"""


def build_dialogue_prompt(
    message: str,
    rag_ctx: str,
    turns: int,
    chars_per_turn: int,
    stance_A: Optional[str],
    stance_B: Optional[str]
) -> Dict[str, str]:
    """Собирает промпт для генерации диалога A vs B с живым стилем, обращением на 'ты' и естественным использованием RAG."""

    posA = stance_A or "человек, которому близки идеи партии и который их защищает"
    posB = stance_B or "критик этих идей"

    user_msg = f"""
    === ТЕМА / СООБЩЕНИЕ ===
    {message}

    === ФОРМАТ ДИАЛОГА ===
    - Роль A — уверенная, спокойная ({posA})
    - Роль B — спорит, но хуже ({posB})
    - Короткие, живые реплики без официоза.
    - Общение исключительно на 'ты', без обращений на 'вы'.
    - Нормальная капитализация.
    - Примерный лимит: {chars_per_turn} символов на реплику.
    - Всего ходов: {turns}.
    - Разрешено немного иронии, шуток и подколов — без токсичности.

    === RAG-КОНТЕКСТ ===
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
    logger.debug(f"🔍 Encoding query: '{query[:50]}...'")
    qv = emb_model.encode([query], normalize_embeddings=True).tolist()[0]
    logger.debug(f"✅ Query encoded to vector of dimension {len(qv)}")
    
    logger.debug(f"📊 Querying Qdrant collection '{collection}' for top {k} results...")
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
    logger.debug(f"✅ Retrieved {len(hits)} hits from Qdrant")
    return hits


def build_rag_context(hits: List[Dict]) -> str:
    """Форматирует результаты RAG в текстовый контекст."""
    logger.debug(f"📝 Building RAG context from {len(hits)} chunks...")
    context = "\n\n".join([f"[Источник {i+1}] {h['chunk']}" for i, h in enumerate(hits)])
    logger.debug(f"✅ RAG context built: {len(context)} characters")
    return context


# ===================== OpenAI Генератор =====================
def call_openai(oa_client, system_msg: str, user_msg: str, model: str, temperature: float, expect_json: bool = True) -> str:
    """Вызов OpenAI API для генерации ответа."""
    logger.debug(f"🤖 Preparing OpenAI request: model={model}, temperature={temperature}, expect_json={expect_json}")
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

    logger.debug("📡 Sending request to OpenAI API...")
    resp = oa_client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    logger.debug(f"✅ OpenAI response received: {len(content)} characters")
    return content
