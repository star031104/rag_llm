# -*- coding: utf-8 -*-
"""
专业版 RAG 检索评估脚本
适配当前项目的 demo.py -> retrieve() 接口。
"""

import json
import math
import argparse
import importlib.util
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Set


def normalize_text(text: str) -> str:
    return (text or "").strip()


def load_demo_module(demo_py: str):
    demo_path = Path(demo_py)
    if not demo_path.exists():
        raise FileNotFoundError(f"未找到 demo.py: {demo_py}")
    spec = importlib.util.spec_from_file_location("project_demo", str(demo_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "retrieve"):
        raise AttributeError("demo.py 中未找到 retrieve() 函数")
    return module


def load_dataset(qa_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_docs_map(docs_jsonl: str) -> Dict[str, Dict[str, Any]]:
    docs = {}
    with open(docs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                cid = obj.get("chunk_id")
                if cid:
                    docs[cid] = obj
    return docs


def validate_dataset(dataset: List[Dict[str, Any]], docs_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    missing_gold = []
    bad_rows = []
    duplicated_ids = set()
    seen_ids = set()

    for row in dataset:
        rid = row.get("id", "")
        if rid in seen_ids:
            duplicated_ids.add(rid)
        seen_ids.add(rid)

        if not row.get("question") or not row.get("gold_chunks"):
            bad_rows.append(rid)
            continue

        for cid in row["gold_chunks"]:
            if cid not in docs_map:
                missing_gold.append({"id": rid, "missing_chunk": cid})

    return {
        "total_rows": len(dataset),
        "bad_rows": bad_rows,
        "duplicated_ids": sorted(duplicated_ids),
        "missing_gold_chunks": missing_gold,
    }


def match_chunk(retrieved: Dict[str, Any], gold_chunks: Set[str]) -> bool:
    return normalize_text(retrieved.get("chunk_id")) in gold_chunks


def relevance_list(results: List[Dict[str, Any]], gold_chunks: Set[str], k: int) -> List[int]:
    return [1 if match_chunk(item, gold_chunks) else 0 for item in results[:k]]


def hit_at_k(rel: List[int]) -> float:
    return 1.0 if any(rel) else 0.0


def recall_at_k(rel: List[int], total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(rel) / total_relevant


def mrr(results: List[Dict[str, Any]], gold_chunks: Set[str]) -> float:
    for rank, item in enumerate(results, start=1):
        if match_chunk(item, gold_chunks):
            return 1.0 / rank
    return 0.0


def dcg(rel: List[int]) -> float:
    score = 0.0
    for i, r in enumerate(rel, start=1):
        if r > 0:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(rel: List[int], total_relevant: int) -> float:
    ideal_len = min(total_relevant, len(rel))
    if ideal_len == 0:
        return 0.0
    ideal_rel = [1] * ideal_len + [0] * (len(rel) - ideal_len)
    ideal_score = dcg(ideal_rel)
    if ideal_score == 0:
        return 0.0
    return dcg(rel) / ideal_score


def first_hit_rank(results: List[Dict[str, Any]], gold_chunks: Set[str]) -> int:
    for rank, item in enumerate(results, start=1):
        if match_chunk(item, gold_chunks):
            return rank
    return -1


def init_metric_dict(topk_list: List[int]) -> Dict[str, float]:
    metrics = {"MRR": 0.0}
    for k in topk_list:
        metrics[f"Hit@{k}"] = 0.0
        metrics[f"Recall@{k}"] = 0.0
        metrics[f"nDCG@{k}"] = 0.0
    return metrics


def average_metric_dict(metric_dict: Dict[str, float], count: int) -> Dict[str, float]:
    if count == 0:
        return {k: 0.0 for k in metric_dict.keys()}
    return {k: v / count for k, v in metric_dict.items()}


def evaluate(
    qa_path: str,
    demo_py: str,
    docs_jsonl: str,
    topk_list: List[int],
    retrieve_n: int = 50,
    use_intent: bool = True,
    validate_only: bool = False,
    output: str = "retrieval_report_professional.json",
    error_output: str = "retrieval_errors_professional.json",
) -> None:
    dataset = load_dataset(qa_path)
    docs_map = load_docs_map(docs_jsonl)
    validation = validate_dataset(dataset, docs_map)

    if validate_only:
        result = {"validation": validation}
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if validation["bad_rows"] or validation["duplicated_ids"] or validation["missing_gold_chunks"]:
        raise ValueError(f"评估集校验失败：{json.dumps(validation, ensure_ascii=False)}")

    demo_module = load_demo_module(demo_py)
    retrieve = demo_module.retrieve

    max_k = max(topk_list)
    total = len(dataset)

    overall_acc = init_metric_dict(topk_list)
    by_type_acc = defaultdict(lambda: {"count": 0, **init_metric_dict(topk_list)})
    by_standard_acc = defaultdict(lambda: {"count": 0, **init_metric_dict(topk_list)})
    error_cases = []

    for row in dataset:
        question = row["question"]
        gold_chunks = set(row["gold_chunks"])
        qtype = row.get("type", "unknown")
        standard = row.get("standard", "unknown")
        intent = row.get("intent", "general")
        used_intent = intent if use_intent else "general"
        total_relevant = max(1, len(gold_chunks))

        results = retrieve(
            question,
            intent=used_intent,
            retrieve_n=retrieve_n,
            topk=max_k
        )

        mrr_score = mrr(results, gold_chunks)
        overall_acc["MRR"] += mrr_score
        by_type_acc[qtype]["MRR"] += mrr_score
        by_standard_acc[standard]["MRR"] += mrr_score

        hit_rank = first_hit_rank(results, gold_chunks)

        for k in topk_list:
            rel = relevance_list(results, gold_chunks, k)
            hk = hit_at_k(rel)
            rk = recall_at_k(rel, total_relevant)
            nk = ndcg_at_k(rel, total_relevant)

            overall_acc[f"Hit@{k}"] += hk
            overall_acc[f"Recall@{k}"] += rk
            overall_acc[f"nDCG@{k}"] += nk

            by_type_acc[qtype][f"Hit@{k}"] += hk
            by_type_acc[qtype][f"Recall@{k}"] += rk
            by_type_acc[qtype][f"nDCG@{k}"] += nk

            by_standard_acc[standard][f"Hit@{k}"] += hk
            by_standard_acc[standard][f"Recall@{k}"] += rk
            by_standard_acc[standard][f"nDCG@{k}"] += nk

        by_type_acc[qtype]["count"] += 1
        by_standard_acc[standard]["count"] += 1

        if hit_rank == -1 or hit_rank > 3:
            error_cases.append({
                "id": row["id"],
                "question": question,
                "intent_used": used_intent,
                "type": qtype,
                "standard": standard,
                "gold_chunks": list(gold_chunks),
                "first_hit_rank": hit_rank,
                "top_results": [
                    {
                        "rank": i + 1,
                        "chunk_id": item.get("chunk_id", ""),
                        "score": item.get("score", 0.0),
                        "type": item.get("type", ""),
                        "text_preview": normalize_text(item.get("text", ""))[:180],
                    }
                    for i, item in enumerate(results[:max_k])
                ]
            })

    overall = average_metric_dict(overall_acc, total)

    by_type = {}
    for key, acc in by_type_acc.items():
        count = acc.pop("count")
        by_type[key] = {
            "count": count,
            **average_metric_dict(acc, count)
        }

    by_standard = {}
    for key, acc in by_standard_acc.items():
        count = acc.pop("count")
        by_standard[key] = {
            "count": count,
            **average_metric_dict(acc, count)
        }

    report = {
        "dataset_size": total,
        "use_intent_filter": use_intent,
        "retrieve_n": retrieve_n,
        "topk_list": topk_list,
        "validation": validation,
        "overall": overall,
        "by_type": by_type,
        "by_standard": by_standard,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(error_output, "w", encoding="utf-8") as f:
        json.dump(error_cases, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="专业版 RAG 检索评估脚本")
    parser.add_argument("--qa_path", required=True, help="评估集 jsonl 文件路径")
    parser.add_argument("--demo_py", required=True, help="项目 demo.py 路径")
    parser.add_argument("--docs_jsonl", required=True, help="知识库 docs.jsonl 路径")
    parser.add_argument("--output", default="retrieval_report_professional.json", help="结果报告输出路径")
    parser.add_argument("--error_output", default="retrieval_errors_professional.json", help="错误样本输出路径")
    parser.add_argument("--retrieve_n", type=int, default=50, help="FAISS 召回候选数")
    parser.add_argument("--topk", default="1,3,5,8", help="评估的 topk 列表，例如 1,3,5,8")
    parser.add_argument("--no_intent", action="store_true", help="关闭 intent filter 做消融实验")
    parser.add_argument("--validate_only", action="store_true", help="仅校验数据集与 gold chunk，不跑检索")
    args = parser.parse_args()

    topk_list = [int(x) for x in args.topk.split(",") if x.strip()]
    evaluate(
        qa_path=args.qa_path,
        demo_py=args.demo_py,
        docs_jsonl=args.docs_jsonl,
        topk_list=topk_list,
        retrieve_n=args.retrieve_n,
        use_intent=not args.no_intent,
        validate_only=args.validate_only,
        output=args.output,
        error_output=args.error_output,
    )
