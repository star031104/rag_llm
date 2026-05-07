# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt

from efficiency_metrics import load_efficiency_json


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_apps(project_dir: Path) -> List[str]:
    policy_dir = project_dir / "dataset" / "隐私政策"
    if not policy_dir.exists():
        return []
    return sorted([p.stem for p in policy_dir.glob("*.txt")])


def run_cmd(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def write_csv(rows: List[Dict], csv_path: Path) -> None:
    if not rows:
        return
    ensure_dir(csv_path.parent)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values: List[float]):
    vals = [float(x) for x in values if x is not None]
    return mean(vals) if vals else None


def draw_bar(values: Dict[str, float], save_path: Path, title: str, ylabel: str = "Seconds") -> None:
    ensure_dir(save_path.parent)
    labels = list(values.keys())
    nums = list(values.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, nums)
    plt.xticks(rotation=20)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def draw_pie(values: Dict[str, float], save_path: Path, title: str) -> None:
    ensure_dir(save_path.parent)
    labels = []
    nums = []
    for k, v in values.items():
        if v > 0:
            labels.append(k)
            nums.append(v)
    if not nums:
        return
    plt.figure(figsize=(7, 7))
    plt.pie(nums, labels=labels, autopct="%1.1f%%")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def load_summary_value(path: Path, key: str, default=None):
    if not path.exists():
        return default
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary", {}).get(key, default)


def main():
    parser = argparse.ArgumentParser(description="系统效率评估总控")
    parser.add_argument("--project_dir", required=True, help="项目根目录")
    parser.add_argument("--eff_dir", default="", help="效率评估脚本目录，默认当前脚本所在目录")
    parser.add_argument("--python_exe", default=sys.executable, help="Python 解释器路径")
    parser.add_argument("--output_dir", required=True, help="结果输出目录")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个 app，0 表示全部")
    parser.add_argument("--app_keyword", default="", help="只跑名称包含该关键字的 app")
    parser.add_argument("--pipelines", default="pp_perm,std_policy,std_perm", help="逗号分隔：pp_perm,std_policy,std_perm")
    parser.add_argument("--do_kb_build", action="store_true", help="是否测知识库构建时间")
    parser.add_argument("--currency", default="¥")
    parser.add_argument("--input_price_per_million", type=float, default=0.0)
    parser.add_argument("--output_price_per_million", type=float, default=0.0)

    # 原始脚本路径，可覆盖
    parser.add_argument("--script_pp_perm", default="", help="隐私政策和权限分析.py 路径")
    parser.add_argument("--script_std_policy", default="", help="国标和隐私政策分析.py 路径")
    parser.add_argument("--script_std_perm", default="", help="国标和权限分析.py 路径")
    parser.add_argument("--script_vectorize", default="", help="向量化.py 路径")

    # 建库脚本可选覆盖参数
    parser.add_argument("--kb_in_path", default="", help="向量化输入 jsonl 路径")
    parser.add_argument("--kb_out_dir", default="", help="向量化输出目录")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    eff_dir = Path(args.eff_dir).resolve() if args.eff_dir else Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    raw_dir = output_dir / "raw"
    plot_dir = output_dir / "plots"
    ensure_dir(raw_dir)
    ensure_dir(plot_dir)

    wrapper_pp_perm = eff_dir / "wrap_隐私政策和权限分析.py"
    wrapper_std_policy = eff_dir / "wrap_国标和隐私政策分析.py"
    wrapper_std_perm = eff_dir / "wrap_国标和权限分析.py"
    wrapper_vectorize = eff_dir / "wrap_向量化.py"

    script_pp_perm = Path(args.script_pp_perm).resolve() if args.script_pp_perm else (project_dir / "隐私政策和权限分析.py")
    script_std_policy = Path(args.script_std_policy).resolve() if args.script_std_policy else (project_dir / "国标和隐私政策分析.py")
    script_std_perm = Path(args.script_std_perm).resolve() if args.script_std_perm else (project_dir / "国标和权限分析.py")
    script_vectorize = Path(args.script_vectorize).resolve() if args.script_vectorize else (project_dir / "向量化.py")

    apps = discover_apps(project_dir)
    if args.app_keyword:
        apps = [x for x in apps if args.app_keyword in x]
    if args.limit > 0:
        apps = apps[:args.limit]

    if not apps:
        raise RuntimeError("没有发现可评估的 app。请检查 dataset/隐私政策 目录。")

    selected_pipelines = [x.strip() for x in args.pipelines.split(",") if x.strip()]
    pipeline_defs = []
    if "pp_perm" in selected_pipelines:
        pipeline_defs.append(("隐私政策和权限分析", wrapper_pp_perm, script_pp_perm))
    if "std_policy" in selected_pipelines:
        pipeline_defs.append(("国标和隐私政策分析", wrapper_std_policy, script_std_policy))
    if "std_perm" in selected_pipelines:
        pipeline_defs.append(("国标和权限分析", wrapper_std_perm, script_std_perm))

    per_run_rows: List[Dict] = []

    # 1) 可选：知识库构建时间
    if args.do_kb_build:
        kb_dir = raw_dir / "KB_BUILD"
        ensure_dir(kb_dir)
        timing_json = kb_dir / "timing.json"
        stdout_txt = kb_dir / "stdout.txt"
        stderr_txt = kb_dir / "stderr.txt"

        cmd = [
            args.python_exe,
            str(wrapper_vectorize),
            "--script_path", str(script_vectorize),
            "--project_dir", str(project_dir),
            "--timing_json", str(timing_json),
        ]
        if args.kb_in_path:
            cmd.extend(["--in_path", args.kb_in_path])
        if args.kb_out_dir:
            cmd.extend(["--out_dir", args.kb_out_dir])

        t0 = time.perf_counter()
        cp = run_cmd(cmd, cwd=project_dir)
        wall = time.perf_counter() - t0

        stdout_txt.write_text(cp.stdout, encoding="utf-8")
        stderr_txt.write_text(cp.stderr, encoding="utf-8")

        if timing_json.exists():
            data = load_efficiency_json(timing_json)
            summary = data.get("summary", {})
            per_run_rows.append({
                "app_name": "KB_BUILD",
                "pipeline_name": "向量化建库",
                "success": 1 if cp.returncode == 0 else 0,
                "return_code": cp.returncode,
                "latency_total_s": round(float(summary.get("latency_total_s", wall)), 6),
                "rag_latency_s": round(float(summary.get("rag_latency_s", 0.0)), 6),
                "llm_inference_time_s": round(float(summary.get("llm_inference_time_s", 0.0)), 6),
                "prompt_tokens": int(summary.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(summary.get("completion_tokens", 0) or 0),
                "total_tokens": int(summary.get("total_tokens", 0) or 0),
                "token_cost": round(float(summary.get("token_cost", 0.0)), 8),
                "throughput_app_per_hour": None,
                "timing_json": str(timing_json),
            })
        else:
            per_run_rows.append({
                "app_name": "KB_BUILD",
                "pipeline_name": "向量化建库",
                "success": 1 if cp.returncode == 0 else 0,
                "return_code": cp.returncode,
                "latency_total_s": round(wall, 6),
                "rag_latency_s": 0.0,
                "llm_inference_time_s": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "token_cost": 0.0,
                "throughput_app_per_hour": None,
                "timing_json": "",
            })

    # 2) 三条主分析链路
    bench_t0 = time.perf_counter()

    for app_name in apps:
        for pipeline_name, wrapper_path, script_path in pipeline_defs:
            app_dir = raw_dir / pipeline_name / app_name
            ensure_dir(app_dir)
            timing_json = app_dir / "timing.json"
            stdout_txt = app_dir / "stdout.txt"
            stderr_txt = app_dir / "stderr.txt"

            cmd = [
                args.python_exe,
                str(wrapper_path),
                "--script_path", str(script_path),
                "--project_dir", str(project_dir),
                "--app_name", app_name,
                "--timing_json", str(timing_json),
                "--currency", args.currency,
                "--input_price_per_million", str(args.input_price_per_million),
                "--output_price_per_million", str(args.output_price_per_million),
            ]

            print(f"[RUN] {pipeline_name} | {app_name}")
            t0 = time.perf_counter()
            cp = run_cmd(cmd, cwd=project_dir)
            wall = time.perf_counter() - t0

            stdout_txt.write_text(cp.stdout, encoding="utf-8")
            stderr_txt.write_text(cp.stderr, encoding="utf-8")

            if timing_json.exists():
                data = load_efficiency_json(timing_json)
                summary = data.get("summary", {})
                total_latency = float(summary.get("latency_total_s", wall))
                per_run_rows.append({
                    "app_name": app_name,
                    "pipeline_name": pipeline_name,
                    "success": 1 if cp.returncode == 0 else 0,
                    "return_code": cp.returncode,
                    "latency_total_s": round(total_latency, 6),
                    "rag_latency_s": round(float(summary.get("rag_latency_s", 0.0)), 6),
                    "llm_inference_time_s": round(float(summary.get("llm_inference_time_s", 0.0)), 6),
                    "prompt_tokens": int(summary.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(summary.get("completion_tokens", 0) or 0),
                    "total_tokens": int(summary.get("total_tokens", 0) or 0),
                    "token_cost": round(float(summary.get("token_cost", 0.0)), 8),
                    "throughput_app_per_hour": round(3600.0 / total_latency, 6) if total_latency > 0 else None,
                    "timing_json": str(timing_json),
                })
            else:
                per_run_rows.append({
                    "app_name": app_name,
                    "pipeline_name": pipeline_name,
                    "success": 1 if cp.returncode == 0 else 0,
                    "return_code": cp.returncode,
                    "latency_total_s": round(wall, 6),
                    "rag_latency_s": 0.0,
                    "llm_inference_time_s": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "token_cost": 0.0,
                    "throughput_app_per_hour": round(3600.0 / wall, 6) if wall > 0 else None,
                    "timing_json": "",
                })

    bench_wall = time.perf_counter() - bench_t0

    # 3) 导出逐条结果
    per_run_csv = output_dir / "per_run_efficiency.csv"
    write_csv(per_run_rows, per_run_csv)

    # 4) 分 pipeline 汇总
    pipeline_summary_rows: List[Dict] = []
    pipeline_names = sorted(set(r["pipeline_name"] for r in per_run_rows if r["pipeline_name"] != "向量化建库"))

    for pname in pipeline_names:
        rows = [r for r in per_run_rows if r["pipeline_name"] == pname and r["success"] == 1]
        if not rows:
            continue
        avg_latency = safe_mean([r["latency_total_s"] for r in rows])
        avg_rag = safe_mean([r["rag_latency_s"] for r in rows])
        avg_llm = safe_mean([r["llm_inference_time_s"] for r in rows])
        avg_prompt_tokens = safe_mean([r["prompt_tokens"] for r in rows])
        avg_completion_tokens = safe_mean([r["completion_tokens"] for r in rows])
        avg_total_tokens = safe_mean([r["total_tokens"] for r in rows])
        avg_cost = safe_mean([r["token_cost"] for r in rows])

        pipeline_summary_rows.append({
            "pipeline_name": pname,
            "app_count": len(rows),
            "avg_latency_total_s": round(avg_latency, 6) if avg_latency is not None else None,
            "avg_rag_latency_s": round(avg_rag, 6) if avg_rag is not None else None,
            "avg_llm_inference_time_s": round(avg_llm, 6) if avg_llm is not None else None,
            "avg_prompt_tokens": round(avg_prompt_tokens, 3) if avg_prompt_tokens is not None else None,
            "avg_completion_tokens": round(avg_completion_tokens, 3) if avg_completion_tokens is not None else None,
            "avg_total_tokens": round(avg_total_tokens, 3) if avg_total_tokens is not None else None,
            "avg_token_cost": round(avg_cost, 8) if avg_cost is not None else None,
            "avg_cost_per_app": round(avg_cost, 8) if avg_cost is not None else None,
            "throughput_app_per_hour": round(len(rows) / bench_wall * 3600.0, 6) if bench_wall > 0 else None,
        })

    pipeline_summary_csv = output_dir / "pipeline_summary.csv"
    write_csv(pipeline_summary_rows, pipeline_summary_csv)

    # 5) overall summary
    all_valid = [r for r in per_run_rows if r["pipeline_name"] != "向量化建库" and r["success"] == 1]

    overall_summary = {
        "total_runs": len([r for r in per_run_rows if r["pipeline_name"] != "向量化建库"]),
        "success_runs": len(all_valid),
        "avg_latency_total_s": safe_mean([r["latency_total_s"] for r in all_valid]),
        "avg_rag_latency_s": safe_mean([r["rag_latency_s"] for r in all_valid]),
        "avg_llm_inference_time_s": safe_mean([r["llm_inference_time_s"] for r in all_valid]),
        "avg_prompt_tokens": safe_mean([r["prompt_tokens"] for r in all_valid]),
        "avg_completion_tokens": safe_mean([r["completion_tokens"] for r in all_valid]),
        "avg_total_tokens": safe_mean([r["total_tokens"] for r in all_valid]),
        "avg_token_cost": safe_mean([r["token_cost"] for r in all_valid]),
        "avg_cost_per_app": safe_mean([r["token_cost"] for r in all_valid]),
        "throughput_overall_app_per_hour": len(all_valid) / bench_wall * 3600.0 if bench_wall > 0 else None,
        "benchmark_wallclock_s": bench_wall,
    }

    kb_rows = [r for r in per_run_rows if r["pipeline_name"] == "向量化建库"]
    if kb_rows:
        overall_summary["kb_build_time_s"] = kb_rows[0]["latency_total_s"]

    summary_json = output_dir / "overall_summary.json"
    summary_json.write_text(json.dumps(overall_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 6) 出图
    # 6.1 各 pipeline 平均总时延
    if pipeline_summary_rows:
        draw_bar(
            {r["pipeline_name"]: float(r["avg_latency_total_s"] or 0.0) for r in pipeline_summary_rows},
            plot_dir / "avg_latency_by_pipeline.png",
            "Average Latency by Pipeline"
        )

        draw_bar(
            {r["pipeline_name"]: float(r["avg_token_cost"] or 0.0) for r in pipeline_summary_rows},
            plot_dir / "avg_cost_by_pipeline.png",
            "Average Cost by Pipeline",
            ylabel="Currency"
        )

    # 6.2 总体时间分布
    time_distribution = {
        "RAG": float(overall_summary.get("avg_rag_latency_s") or 0.0),
        "LLM": float(overall_summary.get("avg_llm_inference_time_s") or 0.0),
        "Other": max(
            0.0,
            float(overall_summary.get("avg_latency_total_s") or 0.0)
            - float(overall_summary.get("avg_rag_latency_s") or 0.0)
            - float(overall_summary.get("avg_llm_inference_time_s") or 0.0)
        ),
    }
    draw_pie(time_distribution, plot_dir / "overall_time_distribution.png", "Overall Time Distribution")

    print("\n================ 效率评估完成 ================\n")
    print(json.dumps(overall_summary, ensure_ascii=False, indent=2))
    print(f"\n逐条结果：{per_run_csv}")
    print(f"分链路汇总：{pipeline_summary_csv}")
    print(f"总体汇总：{summary_json}")
    print(f"图表目录：{plot_dir}")


if __name__ == "__main__":
    main()