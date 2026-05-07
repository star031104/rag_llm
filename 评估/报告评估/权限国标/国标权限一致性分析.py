# -*- coding: utf-8 -*-
"""
国标-权限合规性分析评估（五分类）

适用输入表：
app_name,permission,pred_label,gold_label

输出：
1. overall_metrics.json
2. per_label_metrics.csv
3. per_app_metrics.csv
4. confusion_matrix.csv
5. prediction_errors.csv
6. label_distribution.csv
7. normalized_input.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# =========================================================
# 直接改这里
# =========================================================
INPUT_FILE = Path(r"standard_permission_dataset.csv")
OUTPUT_DIR = Path(r"standard_permission_outputs")
SHEET_NAME = 0   # 仅 xlsx 有效，csv 可忽略
# =========================================================

LABELS: List[str] = [
    "业务核心必要",
    "业务辅助合理",
    "国标未直接枚举但可解释",
    "业务弱相关需审查",
    "高风险异常权限",
]

REQUIRED_COLUMNS: List[str] = [
    "app_name",
    "permission",
    "pred_label",
    "gold_label",
]


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return " ".join(str(x).replace("\u3000", " ").split()).strip()


def load_table(input_file: Path, sheet_name=0) -> pd.DataFrame:
    suffix = input_file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_file, encoding="utf-8-sig")
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(input_file, sheet_name=sheet_name)
    raise ValueError(f"不支持的文件类型: {suffix}")


def validate_dataframe(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}，当前列: {list(df.columns)}")

    bad_pred = sorted(set(df["pred_label"].dropna().astype(str)) - set(LABELS))
    bad_gold = sorted(set(df["gold_label"].dropna().astype(str)) - set(LABELS))

    if bad_pred:
        raise ValueError(f"pred_label 中存在非法标签: {bad_pred}")
    if bad_gold:
        raise ValueError(f"gold_label 中存在非法标签: {bad_gold}")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in REQUIRED_COLUMNS:
        out[col] = out[col].apply(clean_text)

    out = out[
        (out["app_name"] != "")
        & (out["permission"] != "")
        & (out["pred_label"] != "")
        & (out["gold_label"] != "")
    ].reset_index(drop=True)

    validate_dataframe(out)

    out["is_correct"] = (out["pred_label"] == out["gold_label"]).astype(int)
    return out


def compute_overall_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    acc = accuracy_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="micro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
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


def build_confusion_df(y_true: List[str], y_pred: List[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    cm_df.index.name = "gold\\pred"
    return cm_df.reset_index()


def build_per_label_metrics(y_true: List[str], y_pred: List[str]) -> pd.DataFrame:
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    rows = []
    for i, label in enumerate(LABELS):
        rows.append({
            "label": label,
            "precision": round(float(p[i]), 6),
            "recall": round(float(r[i]), 6),
            "f1": round(float(f1[i]), 6),
            "support": int(support[i]),
        })
    return pd.DataFrame(rows)


def build_per_app_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for app_name, g in df.groupby("app_name"):
        y_true = g["gold_label"].tolist()
        y_pred = g["pred_label"].tolist()
        m = compute_overall_metrics(y_true, y_pred)
        rows.append({
            "app_name": app_name,
            "sample_count": len(g),
            "accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "micro_f1": m["micro_f1"],
            "weighted_f1": m["weighted_f1"],
        })
    return (
        pd.DataFrame(rows)
        .sort_values(by=["sample_count", "app_name"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    gold_counts = df["gold_label"].value_counts().reindex(LABELS, fill_value=0)
    pred_counts = df["pred_label"].value_counts().reindex(LABELS, fill_value=0)

    rows = []
    for label in LABELS:
        rows.append({
            "label": label,
            "gold_count": int(gold_counts[label]),
            "pred_count": int(pred_counts[label]),
        })
    return pd.DataFrame(rows)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_table(INPUT_FILE, sheet_name=SHEET_NAME)
    df = prepare_dataframe(raw_df)

    y_true = df["gold_label"].tolist()
    y_pred = df["pred_label"].tolist()

    overall_metrics = compute_overall_metrics(y_true, y_pred)
    overall_metrics.update({
        "input_file": str(INPUT_FILE),
        "sample_count": int(len(df)),
        "app_count": int(df["app_name"].nunique()),
        "label_set": LABELS,
    })

    confusion_df = build_confusion_df(y_true, y_pred)
    per_label_df = build_per_label_metrics(y_true, y_pred)
    per_app_df = build_per_app_metrics(df)
    errors_df = df[df["is_correct"] == 0][["app_name", "permission", "pred_label", "gold_label"]].copy()
    label_dist_df = build_label_distribution(df)

    df.to_csv(OUTPUT_DIR / "normalized_input.csv", index=False, encoding="utf-8-sig")
    confusion_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv", index=False, encoding="utf-8-sig")
    per_label_df.to_csv(OUTPUT_DIR / "per_label_metrics.csv", index=False, encoding="utf-8-sig")
    per_app_df.to_csv(OUTPUT_DIR / "per_app_metrics.csv", index=False, encoding="utf-8-sig")
    errors_df.to_csv(OUTPUT_DIR / "prediction_errors.csv", index=False, encoding="utf-8-sig")
    label_dist_df.to_csv(OUTPUT_DIR / "label_distribution.csv", index=False, encoding="utf-8-sig")
    save_json(overall_metrics, OUTPUT_DIR / "overall_metrics.json")

    print("=" * 80)
    print("国标-权限合规性分析评估结果（权限级，五分类）")
    print("=" * 80)
    print(f"输入文件: {INPUT_FILE}")
    print(f"样本数: {len(df)}")
    print(f"APP 数: {df['app_name'].nunique()}")
    print("-" * 80)
    print(f"Accuracy      : {overall_metrics['accuracy']:.6f}")
    print(f"Macro-F1      : {overall_metrics['macro_f1']:.6f}")
    print(f"Micro-F1      : {overall_metrics['micro_f1']:.6f}")
    print(f"Weighted-F1   : {overall_metrics['weighted_f1']:.6f}")
    print(f"Macro-Prec    : {overall_metrics['macro_precision']:.6f}")
    print(f"Macro-Recall  : {overall_metrics['macro_recall']:.6f}")
    print("-" * 80)
    print("各标签指标：")
    print(per_label_df.to_string(index=False))
    print("-" * 80)
    print("各 APP 指标（前 20 行）：")
    print(per_app_df.head(20).to_string(index=False))
    print("=" * 80)
    print(f"结果已输出到: {OUTPUT_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()