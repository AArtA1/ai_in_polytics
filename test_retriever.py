import os
import re
import sys
import csv
import json
import argparse
from typing import List, Dict, Optional

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer


def color(txt, c):
    COLORS = {
        "grey": "\033[90m", "red": "\033[91m", "green": "\033[92m",
        "yellow": "\033[93m", "blue": "\033[94m", "magenta": "\033[95m",
        "cyan": "\033[96m", "white": "\033[97m", "reset": "\033[0m",
        "bold": "\033[1m",
    }
    return f"{COLORS.get(c,'')}{txt}{COLORS['reset']}"

def highlight_snippet(text: str, query: str, width: int = 260) -> str:
    """Сделать короткий сниппет и подсветить совпадения слов из запроса."""
    # берём пару ключевых слов из запроса
    terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2][:6]
    tset = set(terms)

    # найдём первое вхождение любого терма
    pos = min([text.lower().find(t) for t in tset if t in text.lower()] + [0])
    start = max(0, pos - width // 2)
    end = min(len(text), start + width)
    chunk = text[start:end]

    # подсветка совпадений
    def repl(m):
        return color(m.group(0), "yellow") + color("", "reset")

    for t in sorted(tset, key=len, reverse=True):
        chunk = re.sub(fr"(?i)\b{re.escape(t)}\b", repl, chunk)

    # добавим многоточия если обрезали
    if start > 0:
        chunk = "..." + chunk
    if end < len(text):
        chunk = chunk + "..."
    return chunk

def connect_qdrant(args) -> QdrantClient:
    if args.qdrant_path:
        return QdrantClient(path=args.qdrant_path)
    return QdrantClient(host=args.qdrant_host, port=args.qdrant_port)

def retrieve(qdrant: QdrantClient, emb_model: SentenceTransformer, collection: str, query: str, k: int = 8):
    qv = emb_model.encode([query], normalize_embeddings=True).tolist()[0]
    res = qdrant.query_points(
        collection_name=collection,
        query=qv,
        with_payload=True,
        with_vectors=False,
        limit=k,
    )
    hits = []
    for i, p in enumerate(res.points, 1):
        payload = p.payload or {}
        hits.append({
            "rank": i,
            "score": p.score,
            "title": payload.get("title", ""),
            "source": payload.get("source", ""),
            "chunk": payload.get("chunk", ""),
        })
    return hits

def save_csv(rows: List[Dict], path: str):
    fields = ["rank", "score", "title", "source", "chunk"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k, "") or "").replace("\n", " ") for k in fields})

def print_hits(hits: List[Dict], query: str, show_full: bool = False):
    if not hits:
        print(color("Ничего не найдено.", "red"))
        return
    for h in hits:
        print(color(f"\n#{h['rank']}  score={h['score']:.4f}", "cyan"))
        if h["title"]:
            print(color("title: ", "grey") + h["title"])
        if h["source"]:
            print(color("source:", "grey"), h["source"])
        text = h["chunk"] or ""
        if show_full:
            print(color("text  :", "grey"), text.strip())
        else:
            print(color("snippet:", "grey"), highlight_snippet(text, query))

def main():
    ap = argparse.ArgumentParser(description="Test RAG retriever over Qdrant")
    ap.add_argument("--collection", "-c", default="spravedlivo_ru", help="Имя коллекции в Qdrant")
    ap.add_argument("--emb", default="BAAI/bge-m3", help="HF модель эмбеддингов")
    ap.add_argument("--topk", type=int, default=8, help="Сколько документов возвращать")
    ap.add_argument("--qdrant-host", default="127.0.0.1")
    ap.add_argument("--qdrant-port", type=int, default=6333)
    ap.add_argument("--qdrant-path", default="", help="Альтернатива host/port: путь к локальному Qdrant хранилищу (например, ./qdrant_store)")
    ap.add_argument("--query", "-q", default="", help="Один запрос (если не указан — интерактивный REPL)")
    ap.add_argument("--full", action="store_true", help="Печатать полный текст чанка")
    ap.add_argument("--save-csv", default="", help="Сохранить результаты в CSV")
    args = ap.parse_args()

    # подключение к Qdrant
    try:
        qdrant = connect_qdrant(args)
        cols = qdrant.get_collections().collections
        names = {c.name for c in cols}
        if args.collection not in names:
            print(color(f"Коллекция '{args.collection}' не найдена. Доступные: {sorted(names)}", "red"))
            sys.exit(1)
    except Exception as e:
        print(color(f"Не удалось подключиться к Qdrant: {e}", "red"))
        sys.exit(1)

    # загрузка модели эмбеддингов
    print(color(f"Загружаю эмбеддинги: {args.emb}", "grey"))
    emb = SentenceTransformer(args.emb)

    def run_one(q: str):
        hits = retrieve(qdrant, emb, args.collection, q, k=args.topk)
        print(color(f"\n🔎 Запрос: ", "bold") + q)
        print_hits(hits, q, show_full=args.full)
        if args.save_csv:
            save_csv(hits, args.save_csv)
            print(color(f"\nCSV сохранён → {args.save_csv}", "green"))

    if args.query:
        run_one(args.query)
    else:
        # REPL
        print(color("Интерактивный режим. Введите запрос (пустая строка — выход).", "grey"))
        while True:
            try:
                q = input(color("\n> ", "bold"))
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q.strip():
                break
            run_one(q.strip())

if __name__ == "__main__":
    main()
