# -*- coding: utf-8 -*-
import os
import json
import time
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import faiss
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================
# 基础配置
# =========================
SILICONFLOW_TOKEN = "sk-owpvsumadfmgklabdqoyyesnagsjzofcbrhfgowzoxuuifth"
EMBED_URL = os.getenv("EMBED_URL", "https://api.siliconflow.cn/v1/embeddings").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3").strip()
FORCE_DIMENSIONS = None

STORE_DIR = os.getenv("RAG_STORE_DIR", "rag_store")
FAISS_INDEX = os.path.join(STORE_DIR, "faiss.index")
IDMAP_JSON = os.path.join(STORE_DIR, "idmap.json")
DOCS_JSONL = os.path.join(STORE_DIR, "docs.jsonl")

DEFAULT_RETRIEVE_N = 50
DEFAULT_TOPK = 8

EMBED_CACHE_DIR = Path(os.getenv("EMBED_CACHE_DIR", ".embed_cache"))
EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT = 60
MAX_RETRIES = 5


# =========================
# 全局缓存：避免重复加载大文件
# =========================
_INDEX_CACHE: Optional[faiss.Index] = None
_IDMAP_CACHE: Optional[Dict[int, str]] = None
_DOCS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


# =========================
# Requests Session：禁用系统代理 + 自动重试
# =========================
def build_session() -> requests.Session:
    session = requests.Session()

    # 关键：不读取系统代理 / 环境变量代理
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


# =========================
# 工具函数
# =========================
def ensure_token() -> None:
    if not SILICONFLOW_TOKEN:
        raise RuntimeError(
            "未检测到 SILICONFLOW_API_KEY 环境变量。"
            "请先在系统环境变量或运行环境中设置它。"
        )


def load_idmap(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_docs_map(path: str) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("chunk_id")
            if cid:
                m[cid] = obj
    return m


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


def _cache_key(query: str) -> str:
    raw = f"{EMBED_MODEL}|||{query.strip()}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def _cache_path(query: str) -> Path:
    return EMBED_CACHE_DIR / f"{_cache_key(query)}.json"


def _load_embed_cache(query: str) -> Optional[np.ndarray]:
    path = _cache_path(query)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        vec = np.array(obj["embedding"], dtype="float32")[None, :]
        faiss.normalize_L2(vec)
        return vec
    except Exception:
        return None


def _save_embed_cache(query: str, vec_2d: np.ndarray) -> None:
    path = _cache_path(query)
    obj = {
        "model": EMBED_MODEL,
        "query": query,
        "embedding": vec_2d[0].tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


# =========================
# Embedding
# =========================
def embed_query(query: str, use_cache: bool = True) -> np.ndarray:
    query = (query or "").strip()
    if not query:
        raise ValueError("embed_query 输入为空")

    if use_cache:
        cached = _load_embed_cache(query)
        if cached is not None:
            return cached

    ensure_token()

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": [query],
        "encoding_format": "float",
    }
    if FORCE_DIMENSIONS is not None:
        payload["dimensions"] = int(FORCE_DIMENSIONS)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = SESSION.post(
                EMBED_URL,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                proxies={"http": None, "https": None},
            )

            if r.status_code in (401, 403):
                raise RuntimeError(f"Auth failed ({r.status_code}): {r.text}")
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

            data = r.json()
            if "data" not in data or not data["data"]:
                raise RuntimeError(f"接口返回缺少 data 字段：{data}")

            vec = np.array(data["data"][0]["embedding"], dtype="float32")[None, :]
            faiss.normalize_L2(vec)

            if use_cache:
                _save_embed_cache(query, vec)
            return vec

        except Exception as e:
            last_err = e
            wait_s = min(2 ** (attempt - 1), 10)
            print(f"[embed_query] 第 {attempt}/{MAX_RETRIES} 次请求失败：{e}，{wait_s}s 后重试...")
            time.sleep(wait_s)

    raise RuntimeError(f"embedding 请求连续失败：{last_err}")


# =========================
# 检索类型偏好
# =========================
def preferred_types(intent: str) -> Optional[List[str]]:
    intent = (intent or "").strip().lower()

    if intent in ("permission", "perm", "android_permission"):
        return ["table_row", "table_note", "table_footnote", "table_summary"]

    if intent in ("profile", "appendixa", "necessary_pii", "category"):
        return ["appendixA_profile", "appendixA_table_item", "text"]

    if intent in ("definition", "term", "clause", "standard_text"):
        return ["text", "table_note", "table_footnote"]

    if intent in ("general", "any", ""):
        return None

    return None


# =========================
# 检索主函数
# =========================
def retrieve(
    query: str,
    intent: str = "general",
    retrieve_n: int = DEFAULT_RETRIEVE_N,
    topk: int = DEFAULT_TOPK,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []

    index, idmap, docs = load_store()
    vq = embed_query(query, use_cache=use_cache)
    scores, ids = index.search(vq, retrieve_n)

    pref = preferred_types(intent)
    results: List[Dict[str, Any]] = []

    def add_one(internal_id: int, score: float) -> None:
        chunk_id = idmap.get(internal_id)
        if not chunk_id:
            return
        doc = docs.get(chunk_id)
        if not doc:
            return

        meta = doc.get("meta", {}) or {}
        t = meta.get("type", "unknown")

        results.append({
            "chunk_id": chunk_id,
            "score": float(score),
            "type": t,
            "meta": meta,
            "text": doc.get("text", ""),
        })

    # 第一轮：按 intent 偏好筛选
    for internal_id, score in zip(ids[0], scores[0]):
        if internal_id == -1:
            continue

        internal_id = int(internal_id)
        chunk_id = idmap.get(internal_id)
        if not chunk_id:
            continue

        doc = docs.get(chunk_id)
        if not doc:
            continue

        meta = doc.get("meta", {}) or {}
        t = meta.get("type", "unknown")

        if pref is not None and t not in pref:
            continue

        add_one(internal_id, score)
        if len(results) >= topk:
            return results

    # 第二轮：若偏好筛选后不够 topk，则补充其余结果
    if pref is not None and len(results) < topk:
        seen = {r["chunk_id"] for r in results}
        for internal_id, score in zip(ids[0], scores[0]):
            if internal_id == -1:
                continue

            internal_id = int(internal_id)
            chunk_id = idmap.get(internal_id)
            if not chunk_id or chunk_id in seen:
                continue

            add_one(internal_id, score)
            if len(results) >= topk:
                break

    return results


# =========================
# 展示函数
# =========================
def pretty_print(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("\n[无检索结果]")
        return

    for i, r in enumerate(results, 1):
        meta = r.get("meta", {}) or {}
        sec = meta.get("section_path") or meta.get("section_title") or meta.get("section_num") or ""
        table_id = meta.get("table_id") or ""
        appendix = meta.get("appendix_letter") or ""
        snippet = (r.get("text", "") or "").replace("\n", " ").strip()
        snippet = snippet[:260] + ("…" if len(snippet) > 260 else "")

        print(f"\n#{i} score={r['score']:.4f} type={r['type']} chunk_id={r['chunk_id']}")
        if sec or table_id or appendix:
            print(f"  section: {sec}  table_id: {table_id}  appendix: {appendix}")
        print(f"  snippet: {snippet}")


# =========================
# 命令行入口
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="", help="query text")
    ap.add_argument("--intent", default="general", help="permission/profile/definition/general")
    ap.add_argument("--retrieve_n", type=int, default=DEFAULT_RETRIEVE_N)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--no_cache", action="store_true", help="disable embedding cache")
    ap.add_argument("--reload_store", action="store_true", help="force reload faiss/docs/idmap")
    ap.add_argument("--interactive", action="store_true", help="interactive mode")
    args = ap.parse_args()

    if args.reload_store:
        load_store(force_reload=True)

    if args.interactive:
        print("Interactive retrieval demo. Type 'exit' to quit.")
        while True:
            q = input("\nQuery> ").strip()
            if not q or q.lower() in ("exit", "quit", "q"):
                break

            intent = input("Intent (permission/profile/definition/general)> ").strip() or "general"

            try:
                res = retrieve(
                    q,
                    intent=intent,
                    retrieve_n=args.retrieve_n,
                    topk=args.topk,
                    use_cache=not args.no_cache,
                )
                pretty_print(res)
            except Exception as e:
                print(f"[ERROR] 检索失败：{e}")
    else:
        if not args.q.strip():
            raise SystemExit("Please provide --q or use --interactive")

        try:
            res = retrieve(
                args.q.strip(),
                intent=args.intent,
                retrieve_n=args.retrieve_n,
                topk=args.topk,
                use_cache=not args.no_cache,
            )
            pretty_print(res)
        except Exception as e:
            raise SystemExit(f"检索失败：{e}")


if __name__ == "__main__":
    main()