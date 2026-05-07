# -*- coding: utf-8 -*-
import os
import json
import time
import hashlib
from typing import Any, Dict, List
import requests
import numpy as np
import faiss
from tqdm import tqdm


SILICONFLOW_TOKEN = "sk-owpvsumadfmgklabdqoyyesnagsjzofcbrhfgowzoxuuifth"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"
FORCE_DIMENSIONS = None
IN_PATH = "chunks_with_embedding_text.jsonl"
OUT_DIR = "rag_store"
DOCS_JSONL = os.path.join(OUT_DIR, "docs.jsonl")
IDMAP_JSON = os.path.join(OUT_DIR, "idmap.json")
FAISS_INDEX = os.path.join(OUT_DIR, "faiss.index")
STATS_JSON = os.path.join(OUT_DIR, "stats.json")
BATCH_SIZE = 64
MAX_CHARS = 2200
MAX_RETRIES = 5
BACKOFF_BASE_SEC = 2.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_text(s: Any, max_chars: int = MAX_CHARS) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    if len(s) > max_chars:
        s = s[:max_chars]
    return s

def fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_docs_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("chunk_id")
                if cid:
                    done.add(cid)
            except Exception:
                continue
    return done

def read_idmap(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_idmap(path: str, idmap: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(idmap, f, ensure_ascii=False, indent=2)

def write_stats(path: str, stats: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype("float32", copy=False)
    faiss.normalize_L2(mat)
    return mat


def embed_texts(texts: List[str]) -> np.ndarray:
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
        "encoding_format": "float",
    }
    if FORCE_DIMENSIONS is not None:
        payload["dimensions"] = int(FORCE_DIMENSIONS)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(EMBED_URL, json=payload, headers=headers, timeout=60)
            if r.status_code in (401, 403):
                raise RuntimeError(f"Auth failed ({r.status_code}): {r.text}")
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
            data = r.json()
            vecs = [item["embedding"] for item in data["data"]]
            return np.array(vecs, dtype="float32")
        except Exception as e:
            last_err = e
            # auth errors: no retry
            if "Auth failed" in str(e):
                raise
            sleep_s = BACKOFF_BASE_SEC * (attempt + 1)
            print(f"[embed] attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            print(f"[embed] sleep {sleep_s:.1f}s then retry...")
            time.sleep(sleep_s)
    raise RuntimeError(f"Embedding failed after retries. Last error: {last_err}")


def build_store() -> None:
    ensure_dir(OUT_DIR)
    done_ids = read_docs_done_ids(DOCS_JSONL)
    idmap = read_idmap(IDMAP_JSON)
    index = None
    next_internal_id = 0
    if os.path.exists(FAISS_INDEX) and idmap:
        index = faiss.read_index(FAISS_INDEX)
        next_internal_id = (max(int(k) for k in idmap.keys()) + 1) if idmap else 0
    new_items: List[Dict[str, Any]] = []
    type_counter: Dict[str, int] = {}
    for obj in load_jsonl(IN_PATH):
        cid = obj.get("chunk_id")
        if not cid or cid in done_ids:
            continue
        meta = obj.get("meta", {}) or {}
        t = meta.get("type", "unknown")
        type_counter[t] = type_counter.get(t, 0) + 1
        emb_txt = safe_text(obj.get("embedding_text", ""))
        if not emb_txt:
            continue
        new_items.append({
            "chunk_id": cid,
            "embedding_text": emb_txt,
            "text": obj.get("text", ""),
            "meta": meta,
            "fp": fingerprint(emb_txt),
        })
    if not new_items:
        print("[build] No new items. Store is up-to-date.")
        return
    print(f"[build] New items to process: {len(new_items)}")
    print(f"[build] New items by meta.type: {type_counter}")
    new_vecs: List[np.ndarray] = []
    for i in tqdm(range(0, len(new_items), BATCH_SIZE), desc="[build] embedding"):
        batch = new_items[i:i+BATCH_SIZE]
        texts = [x["embedding_text"] for x in batch]
        mat = embed_texts(texts)
        mat = l2_normalize(mat)
        if index is None:
            dim = mat.shape[1]
            index = faiss.IndexFlatIP(dim)
        for j, x in enumerate(batch):
            internal_id = next_internal_id
            next_internal_id += 1
            idmap[str(internal_id)] = x["chunk_id"]
            new_vecs.append(mat[j:j+1])
            append_jsonl(DOCS_JSONL, {
                "chunk_id": x["chunk_id"],
                "fp": x["fp"],
                "text": x["text"],
                "meta": x["meta"],
            })
    add_mat = np.vstack(new_vecs).astype("float32")
    index.add(add_mat)
    faiss.write_index(index, FAISS_INDEX)
    write_idmap(IDMAP_JSON, idmap)
    stats = {
        "model": EMBED_MODEL,
        "embed_url": EMBED_URL,
        "input_jsonl": IN_PATH,
        "store_dir": OUT_DIR,
        "new_added": len(new_items),
        "total_vectors": int(index.ntotal),
        "total_docs": len(read_docs_done_ids(DOCS_JSONL)),
        "index_type": "IndexFlatIP (cosine via L2 normalize)",
        "batch_size": BATCH_SIZE,
        "force_dimensions": FORCE_DIMENSIONS,
    }
    write_stats(STATS_JSON, stats)
    print("[build] Done.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build_store()
