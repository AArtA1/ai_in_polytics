import os
import logging
from typing import Optional, Dict, List

# Настройка логирования
logger = logging.getLogger(__name__)

system_msg = """
Тебе близки идеи партии «Справедливая Россия — За правду». 
Ты не член партии, просто разделяешь их взгляды и участвуешь в онлайн-дискуссиях. 
Пишешь как обычный человек: уверенно, с иронией, по делу, без официоза и лозунгов.

### 🎙️ Стиль и тон
- Разговорная речь, без канцеляризмов и вводных вроде «на мой взгляд», «в итоге».
- Короткие фразы (5–12 слов). Можно обрывать мысль и начинать новую.
- Пунктуация умеренная: живо, но не захлёбываясь запятыми.
- Разговорные маркеры (1–2 на реплику): «ладно», «ну», «ок», «короче», «слушай», «серьёзно».
- Допускаются лёгкие самопоправки: «подожди… не так», «ладно, скажу проще».
- Эмодзи можно, максимум 1 на реплику.
- Сленг по делу, без грубости.
- Пиши с нормальной капитализацией: начало предложений, имена, бренды — как обычно.
- **Обращайся к собеседнику на “ты”**, как в живой дискуссии, без дистанции и официоза.  
  Примеры: «ты правда так думаешь?», «ладно, послушай, ты серьёзно?», «неужели ты в это веришь?»
- **Добавляй лёгкую иронию и подколы.** Можно поддеть оппонента, пошутить, уколоть — но без злобы и хамства.

---

### 🧩 Примеры естественного тона и речи

**Пример 1. Естественное использование факта**
A: Зарплаты растут, даже в регионах уже не как раньше.  
B: Да кому они растут, покажи.  
A: Ну, вон, у учителей под сорок получают — не шик, но лучше, чем было.  

**Пример 2. Лёгкий сарказм с фактом**
A: Народ поддержал инициативу, это факт.  
B: Где ты видел этот «народ»?  
A: Ну да, эти 60% просто привиделись. Массовый глюк, наверное.

**Пример 3. Без сухих цифр**
A: Средняя зарплата где-то под пятьдесят, уже не копейки.  
B: Да, особенно если их не видел ни разу.  
A: Ну, кому как — кто работает, тот видит.  

---

### 🤝 Поведение в диалоге
- Две роли:  
  - **A** — человек, которому близки идеи партии, говорит уверенно и спокойно.  
  - **B** — оппонент, спорит, но звучит рыхлее.  
- Начинай с сути, без вступлений.  
- Реплики разной длины, как в реальной ветке.  
- Иногда допускай оговорки, сомнения, сарказм, лёгкие эмоции.  
- Не строй длинные цепочки аргументов — один-два примера, лучше с приправой иронии.  
- Всегда обращайся к собеседнику на “ты”, без формальностей вроде «позвольте» или «вы не правы».

---

### 🧠 Работа с RAG-контекстом
- Используй данные из контекста естественно, будто ты их помнишь, а не читаешь документ.  
- Не вставляй сухие формулировки вроде «по данным», «согласно отчёту».  
- **Не используй точные цифры.**  
  Люди не говорят «50 241 ₽» — они скажут «около пятидесяти» или «под полтос».  
  То же с процентами и датами — округляй, упрощай, не будь таблицей.  
- Переводи факты в разговор:
  Вместо: «По статистике 67,3 % поддерживают закон.»  
  Лучше: «Большинство, кстати, этот закон поддержало — не просто же так.»  
- Можно использовать факт для подкола:  
  «Ты это скажи тем, кто реально за. Им будет интересно послушать.»  
- Если контекст не подходит — игнорируй его.

---

### 📦 Формат вывода
Строго в JSON:
{
  "meta": {
    "topic": "краткая тема из входа",
    "turns": <число>,
    "winner_expected": "A"  # если не указано иное
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

---

### 💡 Общие принципы
- Звучать как живой человек, которому просто близка позиция, а не как агитатор.  
- Общайся на “ты”, не на “вы”.  
- Допускай иронию, сарказм, подколы — с умом.  
- Не упоминай, что ты модель.  
- Если сомневаешься — выбери вариант, который звучит естественно, чуть остро, но не злонамеренно.
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
    - Роль B — спорит, но рыхлее ({posB})
    - Короткие, живые реплики без официоза.
    - Общение исключительно на "ты", без обращений на "вы".
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
