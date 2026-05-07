# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


IGNORE_META_KEYS = {"chunk_index", "chunk_id", "offset", "span_start", "span_end"}


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_group_key(meta: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    cleaned = {k: v for k, v in (meta or {}).items() if k not in IGNORE_META_KEYS}
    return tuple(sorted((str(k), json.dumps(v, ensure_ascii=False, sort_keys=True)) for k, v in cleaned.items()))


def make_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(text):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def build_variant(input_jsonl: Path, output_jsonl: Path, mode: str) -> None:
    rows = list(load_jsonl(input_jsonl))
    if mode == "baseline":
        dump_jsonl(output_jsonl, rows)
        print(f"[chunk] baseline copy done -> {output_jsonl}")
        return

    if mode != "alt_small_overlap":
        raise ValueError(f"未知 chunk mode: {mode}")

    grouped: Dict[Tuple[Tuple[str, str], ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[stable_group_key(row.get("meta", {}) or {})].append(row)

    out_rows: List[Dict[str, Any]] = []
    for _, items in grouped.items():
        items = sorted(items, key=lambda x: str(x.get("chunk_id", "")))
        merged_text = "\n".join((x.get("text", "") or "").strip() for x in items if (x.get("text", "") or "").strip())
        if not merged_text:
            continue
        base_meta = copy.deepcopy(items[0].get("meta", {}) or {})
        new_chunks = make_chunks(merged_text, chunk_size=500, overlap=150)
        base_chunk_id = str(items[0].get("chunk_id", "group"))
        for idx, chunk_text in enumerate(new_chunks, 1):
            meta = copy.deepcopy(base_meta)
            meta["alt_chunk_mode"] = mode
            meta["alt_chunk_index"] = idx
            out_rows.append({
                "chunk_id": f"{base_chunk_id}__{mode}__{idx}",
                "text": chunk_text,
                "embedding_text": chunk_text,
                "meta": meta,
            })
    dump_jsonl(output_jsonl, out_rows)
    print(f"[chunk] mode={mode} done. input={len(rows)} output={len(out_rows)} -> {output_jsonl}")


def main() -> None:
    ap = argparse.ArgumentParser(description="为 E4 生成不同 Chunk 版本的 jsonl")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--mode", default="alt_small_overlap")
    args = ap.parse_args()
    build_variant(Path(args.input_jsonl), Path(args.output_jsonl), args.mode)


if __name__ == "__main__":
    main()
