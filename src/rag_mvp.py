# rag_mvp.py
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchParams, Query
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import os

COLL = "spravedlivo_ru"
EMB_NAME = "BAAI/bge-m3"
RERANKER = None  # "BAAI/bge-reranker-large"  # включи при желании
LLM_NAME = "sambanovasystems/SambaLingo-Russian-Base"  # поставь доступную модель

SYS = (
"Ты отвечаешь ТОЛЬКО на основе переданных фрагментов. "
"Любой факт подтверждай короткой цитатой в кавычках и ссылкой вида [idN: источник]. "
"Если фактов нет, скажи, что данных недостаточно. Язык: русский."
)

def connect_qdrant():
    return QdrantClient(path="./qdrant_store")  # тот же инстанс → персистентность нужна? используй docker

def retrieve(qdrant, emb_model, query: str, k=8):
    qv = emb_model.encode([query], normalize_embeddings=True).tolist()[0]
    res = qdrant.query_points(
        collection_name=COLL,
        query=qv,              # вектор запроса
        limit=k,
        with_payload=True,
        with_vectors=False,
    )
    hits = [{"score": r.score, **r.payload} for r in res.points]
    for i, h in enumerate(hits, 1):
        h["cite_id"] = f"id{i}"
        h["cite_str"] = f"[{h.get('source','source')} | {h.get('title','')} | {h['cite_id']}]"
    return hits

def rerank(query: str, hits: list[dict], topk=8) -> list[dict]:
    if RERANKER is None:
        return hits[:8]
    ce = CrossEncoder(RERANKER)
    pairs = [(query, h["chunk"]) for h in hits]
    scores = ce.predict(pairs).tolist()
    for h, s in zip(hits, scores, strict=False): h["rr_score"] = float(s)
    hits = sorted(hits, key=lambda x: x["rr_score"], reverse=True)
    return hits[:topk]

def build_context(passages: list[dict], max_chars=3000) -> str:
    out, cur = [], 0
    for p in passages:
        head = f"### {p['cite_str']}"
        body = p["chunk"].strip()
        sec = head + "\n" + body
        if cur + len(sec) > max_chars: break
        out.append(sec)
        cur += len(sec)
    return "\n\n".join(out)

def load_llm():
    offload_dir = os.path.abspath("./offload")
    os.makedirs(offload_dir, exist_ok=True)

    # 1) Конфиг с return_dict сразу в фабрике
    config = AutoConfig.from_pretrained(LLM_NAME, return_dict=True, use_cache=True)

    tok = AutoTokenizer.from_pretrained(LLM_NAME, use_fast=True)
    # 2) На LLaMA-подобных pad_token часто отсутствует
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        config=config,
        dtype=torch.float16,          # вместо torch_dtype=...
        device_map="auto",
        offload_folder=offload_dir,   # если используется оффлоад на диск
        low_cpu_mem_usage=True,
    )

    # 3) Генерации нужен корректный pad_token_id
    llm.generation_config.pad_token_id = tok.pad_token_id
    return tok, llm

def answer(query: str, qdrant: QdrantClient, emb_model, tok, llm):
    hits = retrieve(qdrant, emb_model, query, k=24)
    hits = rerank(query, hits, topk=8)
    ctx = build_context(hits)

    user = (
        f"Вопрос: {query}\n"
        "Сначала дай краткий «Итог» (2–3 предложения). "
        "Затем разделы «Факты» (с цитатами и [idX]) и «Комментарий». "
        "Не добавляй сведений вне фрагментов."
    )

    prompt = f"[SYSTEM]\n{SYS}\n\n[CONTEXT]\n{ctx}\n\n[USER]\n{user}"
    inputs = tok(
        prompt,
        return_tensors="pt",
        padding=True,          # 4) обязательно, чтобы появился attention_mask
        truncation=True        # на всякий случай, если промпт очень длинный
    ).to(llm.device)
    out = llm.generate(**inputs, max_new_tokens=450, do_sample=True, temperature=0.5, top_p=0.9)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text, hits

if __name__ == "__main__":
    # Для персистентности замени на подключение к docker-Qdrant и НЕ пересоздавай коллекцию при индексации
    qdrant = connect_qdrant()
    # Лайфхак: если индекс строили в другом процессе/узле — просто перестрой его там же; для демо можно
    emb = SentenceTransformer(EMB_NAME)
    tok, llm = load_llm()

    q = "Какие пункты программы касаются поддержки семей и соцполитики?"
    resp, used = answer(q, qdrant, emb, tok, llm)
    print(resp)