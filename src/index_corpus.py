# index_corpus.py
import glob
import json
import os
import re
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === Настройки ===
DATA_DIR = "data_spravedlivo/text"
COLL = "spravedlivo_ru"
EMB_NAME = "BAAI/bge-m3"       # RU/EN мультиязычная модель эмбеддингов
CHUNK_TOKENS = 900             # размер чанка (по словам)
OVERLAP_TOKENS = 180           # перекрытие между чанками
SAVE_META = "index_meta.json"  # сохранение краткой статистики


# === Утилиты ===
def chunk_text(txt: str, max_tokens=CHUNK_TOKENS, overlap=OVERLAP_TOKENS):
    """Разбиваем текст на перекрывающиеся чанки."""
    words = re.findall(r"\S+", txt)
    out, i = [], 0
    while i < len(words):
        j = min(i + max_tokens, len(words))
        out.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i = max(0, j - overlap)
    return out


def iter_docs():
    """Проходит по всем txt файлам и отдаёт тексты."""
    files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    for p in tqdm(files, desc="Сканирование файлов", ncols=100):
        with open(p, encoding="utf-8") as f:
            txt = f.read()
        title = os.path.basename(p)
        yield {"path": p, "title": title, "text": txt, "source": "spravedlivo.ru"}


# === Индексация ===
def main():
    start_time = time.time()
    print(f"🚀 Начинаем индексацию из {DATA_DIR}")

    qdrant = QdrantClient(path="./qdrant_store")  # для MVP; в проде — docker qdrant
    emb = SentenceTransformer(EMB_NAME)
    dim = emb.get_sentence_embedding_dimension()

    qdrant.recreate_collection(
        COLL,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    total_chunks = 0
    all_points = []

    docs = list(iter_docs())

    for doc in tqdm(docs, desc="Индексация документов", ncols=100):
        chunks = chunk_text(doc["text"])
        total_chunks += len(chunks)
        vecs = emb.encode(chunks, normalize_embeddings=True, show_progress_bar=False).tolist()

        for ch, v in zip(chunks, vecs, strict=False):
            all_points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=v,
                payload={
                    "title": doc["title"],
                    "source": doc["source"],
                    "chunk": ch
                }
            ))

        # батчево пушим каждые 500 чанков
        if len(all_points) >= 500:
            qdrant.upsert(COLL, points=all_points)
            all_points = []

    if all_points:
        qdrant.upsert(COLL, points=all_points)

    elapsed = time.time() - start_time
    print(f"\n✅ Индексация завершена: {total_chunks} чанков, {len(docs)} файлов.")
    print(f"⏱ Время: {elapsed:.1f} сек")

    meta = {
        "collection": COLL,
        "embedding_model": EMB_NAME,
        "num_docs": len(docs),
        "num_chunks": total_chunks,
        "build_time_sec": elapsed
    }
    with open(SAVE_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"📄 Сохранена статистика → {SAVE_META}")


if __name__ == "__main__":
    main()
