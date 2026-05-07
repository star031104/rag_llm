# -*- coding: utf-8 -*-
"""
vectorize_ablation.py

用途：
1. 为 baseline / e3 / e4 构建独立的 RAG 向量库
2. 从 chunks jsonl 中读取文本，调用 embedding 接口生成向量
3. 写出：
   - faiss.index
   - docs.jsonl
   - idmap.json

兼容调用方式：
    python vectorize_ablation.py --reset
    python vectorize_ablation.py --reset --experiment baseline
    python vectorize_ablation.py --reset --experiment e3
    python vectorize_ablation.py --reset --experiment e4
    python vectorize_ablation.py --reset --base_dir E:\\exper

说明：
- E3 使用 BAAI/bge-large-zh-v1.5，官方最大输入长度为 512 tokens。
- 因为中文字符与 token 不完全等价，这里采用“保守字符裁剪 + 413 自动降长重试”的方式。
"""

from __future__ import annotations

import os
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Iterable, Optional

import requests
import numpy as np
import faiss
from tqdm import tqdm


# =========================
# 基础配置
# =========================

DEFAULT_EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
DEFAULT_API_KEY = "sk-owpvsumadfmgklabdqoyyesnagsjzofcbrhfgowzoxuuifth"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR

# 模型配置
MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    # baseline
    "BAAI/bge-m3": {
        "safe_max_chars": 1200,   # baseline 相对宽松
        "default_batch_size": 64,
        "min_chars": 200,
    },
    # E3：官方最大 512 tokens，必须保守
    "BAAI/bge-large-zh-v1.5": {
        "safe_max_chars": 420,    # 对中文做保守近似，避免 512 token 上限炸裂
        "default_batch_size": 1,  # 大模型接口更稳妥用单条
        "min_chars": 120,
    },
}

DEFAULT_EMBED_MODEL = "BAAI/bge-m3"


# =========================
# 实验配置读取
# =========================

def try_get_experiment_config(exp_name: str):
    try:
        from exp_config import get_experiment
        return get_experiment(exp_name)
    except Exception as exc:
        print(f"[WARN] 读取 exp_config 失败，使用兜底配置: {exc}")
        return None


def resolve_experiment_settings(base_dir: Path, experiment: str) -> Dict[str, Any]:
    exp_name = (experiment or "baseline").strip().lower()
    exp_cfg = try_get_experiment_config(exp_name)

    embed_model = DEFAULT_EMBED_MODEL
    chunk_input_name = "chunks_with_embedding_text.jsonl"
    rag_store_dirname = "rag_store_baseline"

    if exp_cfg is not None:
        embed_model = getattr(exp_cfg, "embed_model", DEFAULT_EMBED_MODEL)
        chunk_input_name = getattr(exp_cfg, "chunk_output_jsonl", "chunks_with_embedding_text.jsonl")
        rag_store_dirname = getattr(exp_cfg, "rag_store_dirname", "rag_store_baseline")
    else:
        if exp_name == "e3":
            embed_model = "BAAI/bge-large-zh-v1.5"
            rag_store_dirname = "rag_store_e3_embedding"
        elif exp_name == "e4":
            rag_store_dirname = "rag_store_e4_chunk"
            chunk_input_name = "chunks_with_embedding_text_e4.jsonl"
        else:
            rag_store_dirname = "rag_store_baseline"

    input_jsonl = base_dir / chunk_input_name
    store_dir = base_dir / rag_store_dirname

    profile = MODEL_PROFILES.get(
        embed_model,
        {
            "safe_max_chars": 800,
            "default_batch_size": 8,
            "min_chars": 120,
        },
    )

    return {
        "experiment": exp_name,
        "embed_model": embed_model,
        "input_jsonl": input_jsonl,
        "store_dir": store_dir,
        "docs_path": store_dir / "docs.jsonl",
        "idmap_path": store_dir / "idmap.json",
        "faiss_index": store_dir / "faiss.index",
        "safe_max_chars": profile["safe_max_chars"],
        "default_batch_size": profile["default_batch_size"],
        "min_chars": profile["min_chars"],
    }


# =========================
# IO
# =========================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"输入 chunk 文件不存在: {path}")

    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as exc:
                print(f"[WARN] 跳过非法 JSONL 行 {lineno}: {exc}")
    return items


def batched(seq: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


# =========================
# 文本预处理
# =========================

def raw_text_from_item(item: Dict[str, Any]) -> str:
    text = (
        item.get("embedding_text")
        or item.get("text")
        or item.get("content")
        or item.get("chunk_text")
        or ""
    )
    if not isinstance(text, str):
        text = str(text)
    return text.strip()


def clip_text_for_embedding(text: str, max_chars: int) -> str:
    """
    对单条文本做保守裁剪。
    用“头部 + 尾部”方式尽量保留关键信息。
    """
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text

    marker = "\n...\n"
    remain = max_chars - len(marker)
    if remain <= 20:
        return text[:max_chars]

    head_len = remain // 2
    tail_len = remain - head_len
    return text[:head_len] + marker + text[-tail_len:]


def choose_embedding_text(item: Dict[str, Any], max_chars: int) -> str:
    return clip_text_for_embedding(raw_text_from_item(item), max_chars=max_chars)


# =========================
# Embedding API
# =========================

def normalize_vecs(vecs: np.ndarray) -> np.ndarray:
    vecs = np.asarray(vecs, dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


def call_embedding_api(
    texts: List[str],
    model: str,
    api_key: str,
    embed_url: str = DEFAULT_EMBED_URL,
    timeout: int = 180,
    max_retry: int = 5,
) -> List[List[float]]:
    if not api_key:
        raise ValueError(
            "未检测到 SILICONFLOW_API_KEY。请先设置环境变量，"
            "或直接在代码里写入 DEFAULT_API_KEY。"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
    }

    last_err = None
    for attempt in range(1, max_retry + 1):
        try:
            resp = requests.post(embed_url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            arr = data.get("data", [])
            if not arr:
                raise ValueError(f"embedding 接口返回为空: {data}")
            arr = sorted(arr, key=lambda x: x.get("index", 0))
            return [x["embedding"] for x in arr]
        except Exception as exc:
            last_err = exc
            print(f"[WARN] Embedding 调用失败，第 {attempt}/{max_retry} 次: {exc}")
            if attempt < max_retry:
                time.sleep(min(2 * attempt, 8))

    raise RuntimeError(f"Embedding 调用连续失败: {last_err}")


def embed_single_text_with_fallback(
    text: str,
    model: str,
    api_key: str,
    start_chars: int,
    min_chars: int,
) -> List[float]:
    """
    单条文本 embedding。
    如果触发 413，就自动继续缩短。
    """
    original = (text or "").strip()
    current_limit = min(start_chars, max(len(original), min_chars))
    candidate = clip_text_for_embedding(original, max_chars=current_limit)

    while True:
        try:
            embs = call_embedding_api([candidate], model=model, api_key=api_key)
            return embs[0]
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code == 413:
                if current_limit <= min_chars:
                    raise RuntimeError(
                        f"单条文本即使裁剪到 {current_limit} 字符仍触发 413，无法继续压缩。"
                    ) from exc
                new_limit = max(min_chars, current_limit // 2)
                print(f"[WARN] 单条文本过长，触发 413，裁剪长度 {current_limit} -> {new_limit}")
                current_limit = new_limit
                candidate = clip_text_for_embedding(original, max_chars=current_limit)
                continue
            raise


def embed_texts(
    texts: List[str],
    model: str,
    api_key: str,
    batch_size: int,
    safe_max_chars: int,
    min_chars: int,
) -> List[List[float]]:
    """
    - batch_size == 1：逐条 embedding，并支持 413 自动降长
    - batch_size > 1：按批处理，适合 baseline
    """
    all_embs: List[List[float]] = []

    if batch_size == 1:
        for idx, text in enumerate(tqdm(texts, desc="[build] embedding"), 1):
            try:
                emb = embed_single_text_with_fallback(
                    text=text,
                    model=model,
                    api_key=api_key,
                    start_chars=safe_max_chars,
                    min_chars=min_chars,
                )
                all_embs.append(emb)
            except Exception as exc:
                print(f"[ERROR] 第 {idx} 条文本 embedding 失败: {exc}")
                raise
        return all_embs

    batches = list(batched(texts, batch_size))
    for batch in tqdm(batches, desc="[build] embedding"):
        embs = call_embedding_api(batch, model=model, api_key=api_key)
        all_embs.extend(embs)
    return all_embs


# =========================
# 建库
# =========================

def build_store(
    base_dir: Path,
    experiment: str,
    reset: bool = False,
    batch_size_override: Optional[int] = None,
):
    settings = resolve_experiment_settings(base_dir, experiment)

    exp_name = settings["experiment"]
    embed_model = settings["embed_model"]
    input_jsonl: Path = settings["input_jsonl"]
    store_dir: Path = settings["store_dir"]
    docs_path: Path = settings["docs_path"]
    idmap_path: Path = settings["idmap_path"]
    faiss_index_path: Path = settings["faiss_index"]
    safe_max_chars: int = settings["safe_max_chars"]
    default_batch_size: int = settings["default_batch_size"]
    min_chars: int = settings["min_chars"]

    batch_size = batch_size_override if batch_size_override is not None else default_batch_size

    print(f"[INFO] 实验组: {exp_name}")
    print(f"[INFO] BASE_DIR = {base_dir}")
    print(f"[INFO] INPUT_JSONL = {input_jsonl}")
    print(f"[INFO] STORE_DIR = {store_dir}")
    print(f"[INFO] FAISS_INDEX = {faiss_index_path}")
    print(f"[INFO] EMBED_MODEL = {embed_model}")
    print(f"[INFO] SAFE_MAX_CHARS = {safe_max_chars}")
    print(f"[INFO] BATCH_SIZE = {batch_size}")

    store_dir.mkdir(parents=True, exist_ok=True)

    if reset:
        for p in [docs_path, idmap_path, faiss_index_path]:
            if p.exists():
                p.unlink()

    items = load_jsonl(input_jsonl)
    if not items:
        raise ValueError(f"输入 chunk 为空: {input_jsonl}")

    print(f"[build] New items to process: {len(items)}")

    type_stats: Dict[str, int] = {}
    raw_lengths: List[int] = []

    for x in items:
        meta = x.get("meta") or {}
        t = meta.get("type", "unknown")
        type_stats[t] = type_stats.get(t, 0) + 1
        raw_lengths.append(len(raw_text_from_item(x)))

    print(f"[build] New items by meta.type: {type_stats}")
    print(f"[build] 超过 {safe_max_chars} 字符的 chunk 数量: "
          f"{sum(1 for n in raw_lengths if n > safe_max_chars)}/{len(raw_lengths)}")
    print(f"[build] 最长 chunk 字符数: {max(raw_lengths) if raw_lengths else 0}")

    texts = [choose_embedding_text(x, max_chars=safe_max_chars) for x in items]

    valid_pairs = [(i, t) for i, t in enumerate(texts) if t]
    if not valid_pairs:
        raise ValueError("所有 chunk 的 embedding_text/text 都为空，无法建库。")

    valid_indices = [i for i, _ in valid_pairs]
    valid_texts = [t for _, t in valid_pairs]

    api_key = DEFAULT_API_KEY
    embs = embed_texts(
        valid_texts,
        model=embed_model,
        api_key=api_key,
        batch_size=batch_size,
        safe_max_chars=safe_max_chars,
        min_chars=min_chars,
    )

    if not embs:
        raise ValueError("没有生成任何 embedding。")

    mat = np.asarray(embs, dtype="float32")
    mat = normalize_vecs(mat)

    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    docs: List[Dict[str, Any]] = []
    idmap: Dict[str, Any] = {}

    for local_idx, global_idx in enumerate(valid_indices):
        item = items[global_idx]
        docs.append(item)
        idmap[str(local_idx)] = item.get("id", str(global_idx))

    docs_path.parent.mkdir(parents=True, exist_ok=True)
    idmap_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(faiss_index_path))

    with docs_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with idmap_path.open("w", encoding="utf-8") as f:
        json.dump(idmap, f, ensure_ascii=False, indent=2)

    print(f"[build] done -> {store_dir}")
    print(f"[build] faiss vectors: {index.ntotal}")
    print(f"[build] docs saved: {len(docs)}")


# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--experiment", type=str, default="baseline")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="可选。若不提供，则按模型默认值。E3 推荐 1。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    build_store(
        base_dir=base_dir,
        experiment=args.experiment,
        reset=args.reset,
        batch_size_override=args.batch_size,
    )


if __name__ == "__main__":
    main()