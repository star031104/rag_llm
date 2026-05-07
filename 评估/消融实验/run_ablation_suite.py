# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from exp_config import EXPERIMENTS, get_experiment, list_experiment_names


THIS_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str] | None = None) -> int:
    print("\n" + "=" * 100)
    print("[CMD]", " ".join(cmd))
    print("=" * 100)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    return proc.returncode


def iter_experiments(mode: str) -> Iterable[str]:
    mode = (mode or "all").strip().lower()
    if mode == "all":
        return [x for x in list_experiment_names() if x != "baseline"]
    if mode == "baseline":
        return ["baseline"]
    if mode in EXPERIMENTS:
        return [mode]
    if mode == "all_with_baseline":
        return list_experiment_names()
    raise SystemExit("未知 --mode，可选: baseline / e1 / e2 / e3 / e4 / e5 / e6 / all / all_with_baseline")


def ensure_default_store_alias(base_dir: Path, dirname: str) -> None:
    src = base_dir / dirname
    dst = base_dir / "rag_store"
    if not src.exists() or dst.exists():
        return
    try:
        if os.name == "nt":
            shutil.copytree(src, dst)
        else:
            os.symlink(src, dst, target_is_directory=True)
    except Exception:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def prepare_rag_for_experiment(base_dir, exp_name, force_rebuild=False):
    env = os.environ.copy()

    # E4 先构建不同 chunk
    if exp_name == "e4":
        code = run_cmd(
            [
                sys.executable,
                str(THIS_DIR / "chunk_builder_ablation.py"),
                "--input_jsonl", str(base_dir / "chunks_with_embedding_text.jsonl"),
                "--output_jsonl", str(base_dir / "chunks_with_embedding_text_e4.jsonl"),
                "--mode", "alt_small_overlap",
            ],
            cwd=base_dir,
            env=env,
        )
        if code != 0:
            raise RuntimeError("E4 chunk 构建失败")

    # E3 / E4 / baseline 等都应显式传 experiment
    if force_rebuild:
        code = run_cmd(
            [
                sys.executable,
                str(THIS_DIR / "vectorize_ablation.py"),
                "--base_dir", str(base_dir),
                "--experiment", exp_name,
                "--reset",
            ],
            cwd=base_dir,
            env=env,
        )
    else:
        code = run_cmd(
            [
                sys.executable,
                str(THIS_DIR / "vectorize_ablation.py"),
                "--base_dir", str(base_dir),
                "--experiment", exp_name,
            ],
            cwd=base_dir,
            env=env,
        )

    if code != 0:
        raise RuntimeError(f"{exp_name} 向量库构建失败")


def main() -> None:
    ap = argparse.ArgumentParser(description="E1-E6 消融实验总控脚本（只跑分析，不做评估）")
    ap.add_argument("--base_dir", default=".", help="你的项目根目录")
    ap.add_argument("--mode", default="all", help="baseline/e1/e2/e3/e4/e5/e6/all/all_with_baseline")
    ap.add_argument("--app", default="", help="只跑某个 app；为空则跑全部")
    ap.add_argument("--prepare_rag", action="store_true", help="按实验自动构建对应的 RAG 库")
    ap.add_argument("--force_rebuild_rag", action="store_true", help="强制重建对应实验的 RAG 库")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    py = sys.executable
    exps = list(iter_experiments(args.mode))
    print("将运行实验:", ", ".join(exps))

    for exp_name in exps:
        exp = get_experiment(exp_name)
        if args.prepare_rag or exp_name in {"baseline", "e3", "e4"}:
            prepare_rag_for_experiment(base_dir, exp_name, force_rebuild=args.force_rebuild_rag)

        env = os.environ.copy()
        env["CHAT_MODEL"] = exp.chat_model
        env["EMBED_MODEL"] = exp.embed_model
        env["RAG_STORE_DIR"] = str(base_dir / exp.rag_store_dirname)
        if exp.chunk_mode != "baseline":
            env["RAG_INPUT_JSONL"] = str(base_dir / exp.chunk_output_jsonl)

        common = ["--base_dir", str(base_dir), "--experiment", exp_name]
        if args.app.strip():
            common += ["--app", args.app.strip()]

        for script in [
            "policy_permission_ablation.py",
            "standard_permission_ablation.py",
            "standard_policy_ablation.py",
        ]:
            code = run_cmd([py, str(THIS_DIR / script), *common], cwd=base_dir, env=env)
            if code != 0:
                raise SystemExit(code)

    print("\n全部分析已完成。输出目录默认在: output_ablation/<experiment>/<app_name>/")


if __name__ == "__main__":
    main()
