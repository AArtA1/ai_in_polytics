# AI in Politics - RAG System

Система для генерации диалогов и анализа политических текстов с использованием RAG (Retrieval-Augmented Generation).

## Установка

```bash
# Установка зависимостей через uv
uv sync
```

## 🐳 Docker деплой

### Сборка и запуск

```bash
# Сборка образа
docker build -t ai-in-politics .

# Запуск контейнера
docker run -d \
  --name ai-politics \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-api-key-here \
  ai-in-politics

# Просмотр логов (с детальной диагностикой)
docker logs -f ai-politics

# Остановка
docker stop ai-politics

# Удаление контейнера
docker rm ai-politics
```

### Диагностика проблем

Если контейнер зависает на "Waiting for application startup":
- Проверьте логи: `docker logs -f <container_id>`
- При первом запуске модель эмбеддингов (~2GB) кэшируется в образе, это занимает время
- После сборки образа последующие запуски будут мгновенными

Интерактивная отладка:
```bash
# Запуск bash в контейнере
docker run -it --rm ai-in-politics bash

# Проверка структуры
ls -la /app
ls -la /app/qdrant_store

# Ручной запуск приложения
uv run python -c "from src.app import app; print('Import OK')"
```

## Настройка переменных окружения

1. Скопируйте файл `env.example` в `.env`:
```bash
cp env.example .env
```

2. Отредактируйте `.env` и добавьте ваш OpenAI API ключ:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```


Получить ключ можно на https://platform.openai.com/api-keys

### Системный промпт диалогов

Системный промпт вынесен в один файл и может быть переопределён через переменную окружения:

- `dialogue_system_prompt.txt` — файл по умолчанию (можете изменить под свой стиль)

Переменная окружения:

```bash
DIALOGUE_SYSTEM_PROMPT_PATH=dialogue_system_prompt.txt
```

В диалоге участник A — сторонник/защитник позиции партии, участник B — критик. Стиль — живой, уместная ирония, без оскорблений.

## Запуск

### FastAPI сервер для генерации диалогов

```bash
# С автоперезагрузкой для разработки
uv run uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

# Или напрямую
uv run python src/app.py
```

API будет доступен на http://localhost:8000

- Документация: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Другие скрипты

```bash
# Индексация корпуса в Qdrant
uv run python src/index_corpus.py

# Тестирование retriever
uv run python test_retriever.py

# RAG MVP
uv run python src/rag_mvp.py
```

## Структура проекта

- `src/app.py` - FastAPI API сервер (только endpoints)
- `src/dialogue_prompts.py` - RAG логика и промпты для диалогов
- `src/index_corpus.py` - Индексация текстов в Qdrant
- `src/rag_mvp.py` - MVP RAG системы
- `src/scrape_data.py` - Скрапинг данных
- `test_retriever.py` - Тесты retriever
- `data_spravedlivo/` - Данные corpus
- `qdrant_store/` - Локальное хранилище Qdrant

## API Endpoints

### POST /rag_generate

Генерация контента на основе RAG.

**Параметры:**
- `message` - описание темы/кейса
- `mode` - тип генерации: `dialogue`, `post`, `answer`, `summary`
- `top_k` - количество релевантных документов (по умолчанию 6)
- `turns` - количество ходов в диалоге (для mode=dialogue)
- `chars_per_turn` - лимит символов на реплику (для mode=dialogue)
- `stance_A`, `stance_B` - позиции участников (опционально)

**Пример запроса:**
```json
{
  "message": "Обсудите влияние социальных сетей на политику",
  "mode": "dialogue",
  "top_k": 6,
  "turns": 6,
  "chars_per_turn": 400
}
```

### POST /retrieve

Поиск релевантных документов.

### GET /health

Проверка статуса сервера.

