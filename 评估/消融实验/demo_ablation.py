# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SILICONFLOW_TOKEN = "sk-owpvsumadfmgklabdqoyyesnagsjzofcbrhfgowzoxuuifth"
EMBED_URL = os.getenv("EMBED_URL", "https://api.siliconflow.cn/v1/embeddings").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3").strip()
STORE_DIR = os.getenv("RAG_STORE_DIR", "rag_store_baseline")
FAISS_INDEX = os.path.join(STORE_DIR, "faiss.index")
IDMAP_JSON = os.path.join(STORE_DIR, "idmap.json")
DOCS_JSONL = os.path.join(STORE_DIR, "docs.jsonl")
DEFAULT_RETRIEVE_N = 50
DEFAULT_TOPK = 8
EMBED_CACHE_DIR = Path(os.getenv("EMBED_CACHE_DIR", ".embed_cache"))
EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5

_INDEX_CACHE: Optional[faiss.Index] = None
_IDMAP_CACHE: Optional[Dict[int, str]] = None
_DOCS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def build_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


SESSION = build_session()


def ensure_token() -> None:
    if not SILICONFLOW_TOKEN:
        raise RuntimeError("未检测到 SILICONFLOW_API_KEY / API_KEY 环境变量")


def load_idmap(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_docs_map(path: str) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("chunk_id")
            if cid:
                mapping[cid] = obj
    return mapping


def load_store(force_reload: bool = False) -> Tuple[faiss.Index, Dict[int, str], Dict[str, Dict[str, Any]]]:
    global _INDEX_CACHE, _IDMAP_CACHE, _DOCS_CACHE
    if not force_reload and _INDEX_CACHE is not None and _IDMAP_CACHE is not None and _DOCS_CACHE is not None:
        return _INDEX_CACHE, _IDMAP_CACHE, _DOCS_CACHE
    if not os.path.exists(FAISS_INDEX):
        raise FileNotFoundError(f"Missing: {FAISS_INDEX}")
    if not os.path.exists(IDMAP_JSON):
        raise FileNotFoundError(f"Missing: {IDMAP_JSON}")
    if not os.path.exists(DOCS_JSONL):
        raise FileNotFoundError(f"Missing: {DOCS_JSONL}")
    _INDEX_CACHE = faiss.read_index(FAISS_INDEX)
    _IDMAP_CACHE = load_idmap(IDMAP_JSON)
    _DOCS_CACHE = load_docs_map(DOCS_JSONL)
    return _INDEX_CACHE, _IDMAP_CACHE, _DOCS_CACHE


def cache_path_for_text(text: str) -> Path:
    key = hashlib.sha1(f"{EMBED_MODEL}::{text}".encode("utf-8")).hexdigest()
    return EMBED_CACHE_DIR / f"{key}.json"


def embed_query(text: str) -> np.ndarray:
    ensure_token()
    text = (text or "").strip()
    if not text:
        raise ValueError("query 为空")
    cp = cache_path_for_text(text)
    if cp.exists():
        arr = np.array(json.loads(cp.read_text(encoding="utf-8")), dtype="float32")
        return arr.reshape(1, -1)
    payload = {"model": EMBED_MODEL, "input": [text], "encoding_format": "float"}
    headers = {"Authorization": f"Bearer {SILICONFLOW_TOKEN}", "Content-Type": "application/json"}
    resp = SESSION.post(EMBED_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    vec = np.array(resp.json()["data"][0]["embedding"], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    cp.write_text(json.dumps(vec.flatten().tolist(), ensure_ascii=False), encoding="utf-8")
    return vec


def keyword_filter(text: str, query: str) -> float:
    q_terms = [x for x in query.lower().split() if x]
    t = text.lower()
    return float(sum(1 for x in q_terms if x in t))


def retrieve(query: str, intent: str = "general", retrieve_n: int = DEFAULT_RETRIEVE_N, topk: int = DEFAULT_TOPK) -> List[Dict[str, Any]]:
    index, idmap, docs = load_store()
    qv = embed_query(query)
    scores, idxs = index.search(qv, max(retrieve_n, topk))
    out: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx < 0:
            continue
        cid = idmap.get(int(idx))
        if not cid or cid not in docs:
            continue
        row = docs[cid]
        text = row.get("text", "") or ""
        meta = row.get("meta", {}) or {}
        rerank = float(score) + 0.03 * keyword_filter(text + " " + json.dumps(meta, ensure_ascii=False), query)
        if intent != "general":
            rerank += 0.01 if intent in (json.dumps(meta, ensure_ascii=False)) else 0.0
        out.append({"chunk_id": cid, "text": text, "meta": meta, "score": float(score), "rerank_score": rerank})
    out.sort(key=lambda x: x["rerank_score"], reverse=True)
    return out[:topk]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--intent", default="general")
    ap.add_argument("--retrieve_n", type=int, default=DEFAULT_RETRIEVE_N)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    args = ap.parse_args()
    rows = retrieve(args.query, intent=args.intent, retrieve_n=args.retrieve_n, topk=args.topk)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
