# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from exp_config import ExperimentConfig


DEFAULT_CHAT_URL = os.getenv("CHAT_URL", "https://api.siliconflow.cn/v1/chat/completions")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_API_KEY = "sk-owpvsumadfmgklabdqoyyesnagsjzofcbrhfgowzoxuuifth"


class ExperimentContext:
    def __init__(self, experiment: ExperimentConfig, base_dir: Optional[Path] = None):
        self.experiment = experiment
        self.base_dir = Path(base_dir or ".").resolve()

    def experiment_output_root(self) -> Path:
        return self.base_dir / "output_ablation" / self.experiment.name

    def app_output_dir(self, app_name: str) -> Path:
        p = self.experiment_output_root() / app_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_manifest(self, app_name: str) -> None:
        if not self.experiment.save_experiment_manifest:
            return
        out = self.app_output_dir(app_name) / "experiment_manifest.json"
        out.write_text(json.dumps(asdict(self.experiment), ensure_ascii=False, indent=2), encoding="utf-8")


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_url: str = DEFAULT_CHAT_URL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        timeout: int = 180,
        max_retries: int = 5,
        sleep_base: float = 2.0,
        system_prompt: str = "你是一名严谨的隐私合规与国家标准分析专家。",
    ):
        self.api_key = api_key or DEFAULT_API_KEY
        self.chat_url = chat_url
        self.chat_model = chat_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_base = sleep_base
        self.system_prompt = system_prompt

    def chat(self, user_prompt: str, temperature: float = 0.1, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            raise RuntimeError("未设置 SILICONFLOW_API_KEY / API_KEY 环境变量")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
        }
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(self.chat_url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 429:
                    wait_s = self.sleep_base * attempt * 2
                    print(f"[WARN] 429 限流，等待 {wait_s:.1f}s 后重试...")
                    time.sleep(wait_s)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return (data["choices"][0]["message"]["content"] or "").strip()
            except Exception as exc:
                last_err = exc
                wait_s = min(self.sleep_base * attempt, 15)
                print(f"[WARN] LLM 调用失败，第 {attempt}/{self.max_retries} 次：{exc}")
                time.sleep(wait_s)
        raise RuntimeError(f"LLM 调用失败：{last_err}")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return ""


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def try_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(safe_read_text(path))
    except Exception:
        return default


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dedup_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def split_sentences(text: str) -> List[str]:
    text = (text or "").replace("\r", "\n")
    parts = re.split(r"(?<=[。！？；\n])|(?<=[.!?;])\s+", text)
    return [normalize_space(p) for p in parts if normalize_space(p)]


def sliding_windows(text: str, window_size: int = 1600, step: int = 1000) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= window_size:
        return [text]
    out: List[str] = []
    start = 0
    while start < len(text):
        chunk = text[start:start + window_size].strip()
        if chunk:
            out.append(chunk)
        if start + window_size >= len(text):
            break
        start += step
    return out


def shorten(text: str, limit: int = 180) -> str:
    text = normalize_space(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def extract_json_payload(raw: str) -> Any:
    if not raw:
        return None
    text = raw.strip()
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    candidates.extend(fenced)
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        first = text.find(open_ch)
        last = text.rfind(close_ch)
        if first != -1 and last != -1 and last > first:
            candidates.append(text[first:last + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def normalize_evidence_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return dedup_keep_order(normalize_space(str(x)) for x in value if normalize_space(str(x)))
    if isinstance(value, str):
        s = normalize_space(value)
        return [s] if s else []
    return []


def try_import_retrieve(experiment: ExperimentConfig):
    if not experiment.use_rag_retrieval:
        return None
    try:
        from demo_ablation import retrieve  # type: ignore
        return retrieve
    except Exception:
        try:
            from demo import retrieve  # type: ignore
            return retrieve
        except Exception:
            return None
