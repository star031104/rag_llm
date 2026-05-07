# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from efficiency_metrics import EfficiencyRecorder


def import_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本：{file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="国标和权限分析 - 效率评估包装器")
    parser.add_argument("--script_path", required=True, help="原始脚本路径：国标和权限分析.py")
    parser.add_argument("--project_dir", required=True, help="项目根目录")
    parser.add_argument("--app_name", required=True, help="应用名称")
    parser.add_argument("--timing_json", required=True, help="效率指标输出 JSON")
    parser.add_argument("--currency", default="¥")
    parser.add_argument("--input_price_per_million", type=float, default=0.0)
    parser.add_argument("--output_price_per_million", type=float, default=0.0)
    args = parser.parse_args()

    script_path = Path(args.script_path).resolve()
    project_dir = Path(args.project_dir).resolve()

    mod = import_module_from_path("std_perm_module", script_path)

    # 重绑路径
    if hasattr(mod, "BASE_DIR"):
        mod.BASE_DIR = project_dir
    if hasattr(mod, "CATEGORY_DIR"):
        mod.CATEGORY_DIR = project_dir / "dataset" / "分类"
    if hasattr(mod, "POLICY_DIR"):
        mod.POLICY_DIR = project_dir / "dataset" / "隐私政策"
    if hasattr(mod, "APK_MAPPING_DIR"):
        mod.APK_MAPPING_DIR = project_dir / "dataset" / "apk映射"
    if hasattr(mod, "SEMANTIC_CACHE_FILE"):
        mod.SEMANTIC_CACHE_FILE = project_dir / "dataset" / "权限语义缓存" / "semantic_cache.json"
    if hasattr(mod, "OUTPUT_ROOT_DIR"):
        mod.OUTPUT_ROOT_DIR = project_dir / "output"
        mod.OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    recorder = EfficiencyRecorder(
        app_name=args.app_name,
        pipeline_name="国标和权限分析",
        pricing={
            "currency": args.currency,
            "input_per_million": args.input_price_per_million,
            "output_per_million": args.output_price_per_million,
        },
        meta={
            "script_path": str(script_path),
            "project_dir": str(project_dir),
        }
    )

    chat_success_records: List[Dict[str, Any]] = []

    # patch requests.post
    if not hasattr(mod, "requests"):
        raise RuntimeError("原脚本未导入 requests，无法埋点。")

    original_post = mod.requests.post

    def patched_post(*post_args, **post_kwargs):
        resp = original_post(*post_args, **post_kwargs)
        try:
            url = ""
            if post_args:
                url = str(post_args[0])
            if "chat/completions" in url and getattr(resp, "status_code", None) == 200:
                payload = resp.json()
                usage = payload.get("usage", {}) or {}
                model = payload.get("model") or (
                    post_kwargs.get("json", {}) or {}
                ).get("model", "")
                chat_success_records.append({
                    "model": model,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                })
        except Exception:
            pass
        return resp

    mod.requests.post = patched_post

    # patch call_llm
    if not hasattr(mod, "call_llm"):
        raise RuntimeError("原脚本未找到 call_llm。")

    original_call_llm = mod.call_llm

    def patched_call_llm(*call_args, **call_kwargs):
        before = len(chat_success_records)
        t0 = time.perf_counter()
        result = original_call_llm(*call_args, **call_kwargs)
        dt = time.perf_counter() - t0

        new_records = chat_success_records[before:]
        usage = new_records[-1] if new_records else {}

        recorder.add_llm_call(
            model=str(usage.get("model") or getattr(mod, "CHAT_MODEL", "")),
            provider="OpenAI-Compatible",
            latency_s=dt,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
        )
        return result

    mod.call_llm = patched_call_llm

    # patch retrieve
    if hasattr(mod, "retrieve"):
        original_retrieve = mod.retrieve

        def patched_retrieve(*ret_args, **ret_kwargs):
            query = ""
            if ret_args:
                query = str(ret_args[0])
            elif "query" in ret_kwargs:
                query = str(ret_kwargs["query"])

            t0 = time.perf_counter()
            result = original_retrieve(*ret_args, **ret_kwargs)
            dt = time.perf_counter() - t0

            hit_count = len(result) if isinstance(result, list) else None
            recorder.add_rag_call(
                query=query,
                latency_s=dt,
                topk=ret_kwargs.get("topk"),
                hit_count=hit_count,
                extra={"intent": ret_kwargs.get("intent")},
            )
            return result

        mod.retrieve = patched_retrieve

    # 找到目标 app 记录
    if not hasattr(mod, "load_all_category_records"):
        raise RuntimeError("原脚本未找到 load_all_category_records。")
    if not hasattr(mod, "load_semantic_cache"):
        raise RuntimeError("原脚本未找到 load_semantic_cache。")
    if not hasattr(mod, "analyze_app"):
        raise RuntimeError("原脚本未找到 analyze_app。")

    app_records = mod.load_all_category_records()
    target = None
    for rec in app_records:
        if str(rec.get("app_name", "")).strip() == args.app_name:
            target = rec
            break

    if target is None:
        raise RuntimeError(f"未在分类数据中找到应用：{args.app_name}")

    cache = mod.load_semantic_cache()

    t0 = time.perf_counter()
    ok = True
    err = ""

    try:
        mod.analyze_app(target, cache)
    except Exception as e:
        ok = False
        err = repr(e)
        raise
    finally:
        recorder.add_stage_time("analysis", time.perf_counter() - t0)
        recorder.set_extra(success=ok, error=err)
        recorder.save(args.timing_json)


if __name__ == "__main__":
    main()