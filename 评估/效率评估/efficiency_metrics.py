# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


class EfficiencyRecorder:
    """
    统一记录：
    - latency_total_s
    - rag_latency_s
    - llm_inference_time_s
    - token cost
    - throughput（汇总时计算）
    """

    def __init__(
        self,
        app_name: str,
        pipeline_name: str,
        pricing: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.app_name = app_name
        self.pipeline_name = pipeline_name
        self.pricing = pricing or {
            "currency": "¥",
            "input_per_million": 0.0,
            "output_per_million": 0.0,
        }
        self.meta = meta or {}

        self.started_at = time.perf_counter()
        self.finished_at: Optional[float] = None

        self.stage_times: Dict[str, float] = {}
        self.rag_calls: List[Dict[str, Any]] = []
        self.llm_calls: List[Dict[str, Any]] = []
        self.extra: Dict[str, Any] = {}

    def add_stage_time(self, stage_name: str, seconds: float) -> None:
        self.stage_times[stage_name] = self.stage_times.get(stage_name, 0.0) + safe_float(seconds)

    def add_rag_call(
        self,
        query: str,
        latency_s: float,
        topk: Optional[int] = None,
        hit_count: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rag_calls.append({
            "query": query,
            "latency_s": safe_float(latency_s),
            "topk": topk,
            "hit_count": hit_count,
            "extra": extra or {},
        })

    def add_llm_call(
        self,
        model: str,
        latency_s: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
        provider: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        total_tokens = int(total_tokens)

        in_price = safe_float(self.pricing.get("input_per_million", 0.0))
        out_price = safe_float(self.pricing.get("output_per_million", 0.0))

        input_cost = prompt_tokens / 1_000_000.0 * in_price
        output_cost = completion_tokens / 1_000_000.0 * out_price
        total_cost = input_cost + output_cost

        self.llm_calls.append({
            "provider": provider,
            "model": model,
            "latency_s": safe_float(latency_s),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "extra": extra or {},
        })

    def set_extra(self, **kwargs: Any) -> None:
        self.extra.update(kwargs)

    def finalize(self) -> Dict[str, Any]:
        if self.finished_at is None:
            self.finished_at = time.perf_counter()

        total_latency = self.finished_at - self.started_at
        rag_latency = sum(x["latency_s"] for x in self.rag_calls)
        llm_latency = sum(x["latency_s"] for x in self.llm_calls)

        prompt_tokens = sum(x["prompt_tokens"] for x in self.llm_calls)
        completion_tokens = sum(x["completion_tokens"] for x in self.llm_calls)
        total_tokens = sum(x["total_tokens"] for x in self.llm_calls)
        total_cost = sum(x["total_cost"] for x in self.llm_calls)

        data = {
            "app_name": self.app_name,
            "pipeline_name": self.pipeline_name,
            "meta": self.meta,
            "stage_times": self.stage_times,
            "rag_calls": self.rag_calls,
            "llm_calls": self.llm_calls,
            "summary": {
                "latency_total_s": total_latency,
                "rag_latency_s": rag_latency,
                "llm_inference_time_s": llm_latency,
                "other_time_s": max(0.0, total_latency - rag_latency - llm_latency),
                "rag_call_count": len(self.rag_calls),
                "llm_call_count": len(self.llm_calls),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "token_cost": total_cost,
                "currency": self.pricing.get("currency", "¥"),
            },
            "extra": self.extra,
        }
        return data

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        ensure_parent(path)
        data = self.finalize()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def load_efficiency_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))