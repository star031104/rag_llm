#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国标-隐私政策一致性分析评估脚本（控制点级，固定路径版）

说明：
1. 不使用命令行参数，直接在代码顶部修改 INPUT_FILE / OUTPUT_DIR 即可运行。
2. 适用于控制点级数据集。
3. 标签仅支持：满足 / 部分满足 / 不满足。
4. 输入文件至少包含以下列：
   - app_name
   - control_point
   - pred_label
   - gold_label
5. 支持 xlsx / xls / csv。

输出内容：
- overall_metrics.json                  总体指标
- per_control_point_metrics.csv         各控制点指标
- confusion_matrix.csv                  总体混淆矩阵
- prediction_errors.csv                 错误样本
- label_distribution.csv                标签分布
- normalized_input.csv                  归一化后的输入数据
- per_control_confusion_matrices/       各控制点混淆矩阵
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ============================================================
# 需要修改的配置
# ============================================================
INPUT_FILE = "data.xlsx"
SHEET_NAME = 0  # Excel 第一个 sheet；如果是 csv 会自动忽略
OUTPUT_DIR = "eval_outputs_simple"

# 标签顺序：论文里建议固定这一顺序
LABEL_ORDER: List[str] = ["满足", "部分满足", "不满足"]

CONTROL_POINT_NAME_MAP: Dict[str, str] = {
    "A1": "隐私政策适用范围",
    "B1": "个人信息类型与必要性",
    "C2": "敏感个人信息单独同意",
    "D2": "第三方SDK合规披露",
    "F1": "个人信息安全保护措施",
}

REQUIRED_COLUMNS = ["app_name", "control_point", "pred_label", "gold_label"]


# ============================================================
# 基础工具函数
# ============================================================
def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()



def normalize_app_name(x: Any) -> str:
    s = normalize_text(x)
    s = s.replace("（", "(").replace("）", ")")
    return s



def normalize_control_point(x: Any) -> str:
    s = normalize_text(x).upper()
    s = s.replace(" ", "")
    return s



def normalize_label(x: Any) -> str:
    s = normalize_text(x)
    if s not in LABEL_ORDER:
        raise ValueError(f"发现非法标签：{x}。当前脚本只支持 {LABEL_ORDER}")
    return s



def detect_file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return "excel"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"暂不支持的文件类型：{path.suffix}")



def load_input_table(path: Path, sheet_name=0) -> pd.DataFrame:
    file_type = detect_file_type(path)
    if file_type == "excel":
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path, encoding="utf-8-sig")



def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"输入数据缺少必要列：{missing}。当前列为：{list(df.columns)}")



def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)
    out = df.copy()

    out["app_name"] = out["app_name"].apply(normalize_app_name)
    out["control_point"] = out["control_point"].apply(normalize_control_point)
    out["pred_label"] = out["pred_label"].apply(normalize_label)
    out["gold_label"] = out["gold_label"].apply(normalize_label)

    if "control_point_name" not in out.columns:
        out["control_point_name"] = out["control_point"].map(CONTROL_POINT_NAME_MAP).fillna("")
    else:
        out["control_point_name"] = out["control_point_name"].apply(normalize_text)
        out.loc[out["control_point_name"] == "", "control_point_name"] = (
            out["control_point"].map(CONTROL_POINT_NAME_MAP).fillna("")
        )

    optional_text_cols = [
        "pred_reason",
        "gold_reason",
        "evidence_span",
        "source_file",
        "notes",
    ]
    for col in optional_text_cols:
        if col in out.columns:
            out[col] = out[col].apply(normalize_text)

    out["is_correct"] = (out["pred_label"] == out["gold_label"]).astype(int)
    return out


# ============================================================
# 指标计算
# ============================================================
def compute_summary_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "evaluated_samples": int(len(y_true)),
        "accuracy": round(float(acc), 6),
        "macro_precision": round(float(p_macro), 6),
        "macro_recall": round(float(r_macro), 6),
        "macro_f1": round(float(f1_macro), 6),
        "micro_precision": round(float(p_micro), 6),
        "micro_recall": round(float(r_micro), 6),
        "micro_f1": round(float(f1_micro), 6),
        "weighted_precision": round(float(p_weighted), 6),
        "weighted_recall": round(float(r_weighted), 6),
        "weighted_f1": round(float(f1_weighted), 6),
        "classification_report": report,
    }



def build_confusion_df(y_true: List[str], y_pred: List[str], labels: List[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.index.name = "gold\\pred"
    return df.reset_index()



def evaluate_per_control_point(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cp, g in df.groupby("control_point", sort=True):
        y_true = g["gold_label"].tolist()
        y_pred = g["pred_label"].tolist()
        metrics = compute_summary_metrics(y_true, y_pred, labels)
        rows.append({
            "control_point": cp,
            "control_point_name": g["control_point_name"].iloc[0] if "control_point_name" in g.columns else "",
            "sample_count": len(g),
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "micro_precision": metrics["micro_precision"],
            "micro_recall": metrics["micro_recall"],
            "micro_f1": metrics["micro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
        })
    return pd.DataFrame(rows).sort_values(by=["control_point"]).reset_index(drop=True)



def build_error_table(df: pd.DataFrame) -> pd.DataFrame:
    error_df = df[df["is_correct"] == 0].copy()
    preferred_cols = [
        "app_name",
        "control_point",
        "control_point_name",
        "pred_label",
        "gold_label",
        "pred_reason",
        "gold_reason",
        "evidence_span",
        "source_file",
        "notes",
    ]
    keep_cols = [c for c in preferred_cols if c in error_df.columns]
    return error_df[keep_cols] if keep_cols else error_df



def build_label_distribution(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    gold_counts = df["gold_label"].value_counts().reindex(labels, fill_value=0)
    pred_counts = df["pred_label"].value_counts().reindex(labels, fill_value=0)
    return pd.DataFrame({
        "label": labels,
        "gold_count": [int(gold_counts[l]) for l in labels],
        "pred_count": [int(pred_counts[l]) for l in labels],
    })


# ============================================================
# 导出
# ============================================================
def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def save_control_confusion_matrices(df: pd.DataFrame, output_dir: Path, labels: List[str]) -> None:
    sub_dir = output_dir / "per_control_confusion_matrices"
    sub_dir.mkdir(parents=True, exist_ok=True)
    for cp, g in df.groupby("control_point", sort=True):
        cm_df = build_confusion_df(g["gold_label"].tolist(), g["pred_label"].tolist(), labels)
        cm_df.to_csv(sub_dir / f"{cp}_confusion_matrix.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    input_path = Path(INPUT_FILE)
    output_dir = Path(OUTPUT_DIR)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    raw_df = load_input_table(input_path, sheet_name=SHEET_NAME)
    df = prepare_dataframe(raw_df)

    y_true = df["gold_label"].tolist()
    y_pred = df["pred_label"].tolist()

    overall_metrics = compute_summary_metrics(y_true, y_pred, LABEL_ORDER)
    overall_metrics.update({
        "input_file": str(input_path),
        "total_rows": int(len(df)),
        "unique_apps": int(df["app_name"].nunique()),
        "unique_control_points": int(df["control_point"].nunique()),
        "control_points": sorted(df["control_point"].unique().tolist()),
        "label_order": LABEL_ORDER,
    })

    per_control_metrics = evaluate_per_control_point(df, LABEL_ORDER)
    confusion_df = build_confusion_df(y_true, y_pred, LABEL_ORDER)
    errors_df = build_error_table(df)
    label_distribution_df = build_label_distribution(df, LABEL_ORDER)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(overall_metrics, output_dir / "overall_metrics.json")
    df.to_csv(output_dir / "normalized_input.csv", index=False, encoding="utf-8-sig")
    per_control_metrics.to_csv(output_dir / "per_control_point_metrics.csv", index=False, encoding="utf-8-sig")
    confusion_df.to_csv(output_dir / "confusion_matrix.csv", index=False, encoding="utf-8-sig")
    errors_df.to_csv(output_dir / "prediction_errors.csv", index=False, encoding="utf-8-sig")
    label_distribution_df.to_csv(output_dir / "label_distribution.csv", index=False, encoding="utf-8-sig")
    save_control_confusion_matrices(df, output_dir, LABEL_ORDER)

    print("=" * 80)
    print("国标-隐私政策一致性分析评估结果（控制点级）")
    print("=" * 80)
    print(f"输入文件: {input_path}")
    print(f"样本数: {len(df)}")
    print(f"APP 数: {df['app_name'].nunique()}")
    print(f"控制点数: {df['control_point'].nunique()}")
    print(f"控制点列表: {', '.join(sorted(df['control_point'].unique().tolist()))}")
    print("-" * 80)
    print(f"Accuracy      : {overall_metrics['accuracy']:.6f}")
    print(f"Macro-F1      : {overall_metrics['macro_f1']:.6f}")
    print(f"Micro-F1      : {overall_metrics['micro_f1']:.6f}")
    print(f"Weighted-F1   : {overall_metrics['weighted_f1']:.6f}")
    print(f"Macro-Prec    : {overall_metrics['macro_precision']:.6f}")
    print(f"Macro-Recall  : {overall_metrics['macro_recall']:.6f}")
    print("-" * 80)
    print("各控制点指标：")
    if not per_control_metrics.empty:
        print(per_control_metrics.to_string(index=False))
    print("=" * 80)
    print(f"结果已输出到：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
