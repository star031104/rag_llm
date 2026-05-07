# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="向量化建库 - 效率评估包装器")
    parser.add_argument("--script_path", required=True, help="原始脚本路径：向量化.py")
    parser.add_argument("--project_dir", required=True, help="项目根目录")
    parser.add_argument("--timing_json", required=True, help="效率指标输出 JSON")
    parser.add_argument("--in_path", default="", help="可选，覆盖 IN_PATH")
    parser.add_argument("--out_dir", default="", help="可选，覆盖 OUT_DIR")
    args = parser.parse_args()

    script_path = Path(args.script_path).resolve()
    project_dir = Path(args.project_dir).resolve()

    mod = import_module_from_path("vectorize_module", script_path)

    # 路径重绑
    if args.in_path:
        mod.IN_PATH = args.in_path
    else:
        # 这里默认保持原脚本配置；如需覆盖请传 --in_path
        pass

    if args.out_dir:
        mod.OUT_DIR = args.out_dir
        mod.DOCS_JSONL = str(Path(mod.OUT_DIR) / "docs.jsonl")
        mod.IDMAP_JSON = str(Path(mod.OUT_DIR) / "idmap.json")
        mod.FAISS_INDEX = str(Path(mod.OUT_DIR) / "faiss.index")
        mod.STATS_JSON = str(Path(mod.OUT_DIR) / "stats.json")

    recorder = EfficiencyRecorder(
        app_name="KB_BUILD",
        pipeline_name="向量化建库",
        pricing={"currency": "¥", "input_per_million": 0.0, "output_per_million": 0.0},
        meta={
            "script_path": str(script_path),
            "project_dir": str(project_dir),
            "in_path": str(getattr(mod, "IN_PATH", "")),
            "out_dir": str(getattr(mod, "OUT_DIR", "")),
        }
    )

    t0 = time.perf_counter()
    ok = True
    err = ""

    try:
        mod.build_store()
    except Exception as e:
        ok = False
        err = repr(e)
        raise
    finally:
        recorder.add_stage_time("build_store", time.perf_counter() - t0)
        recorder.set_extra(success=ok, error=err)
        recorder.save(args.timing_json)


if __name__ == "__main__":
    main()