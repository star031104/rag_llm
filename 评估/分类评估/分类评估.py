# -*- coding: utf-8 -*-
"""
根据“金标准 CSV + 分类结果 JSON 文件夹”进行评估。

适配的分类 JSON 结构（已根据 DeepSeek.json / 168运友物流.json 定制）：
{
  "app_name": "...",
  "is_in_39": true/false,
  "gbt_39_category": "邮件快件寄递类" / "不属于39类",
  "alt_category": "...",
  "confidence": 0.9,
  ...
}

评估分两层：
1. 二分类评估：是否属于 39 类
2. 多分类评估：仅对“真实属于39类”的样本评估具体类别
   - 真实非39类样本不参与多分类
   - 若真实39类样本被预测为非39类，则在多分类中视为错误

运行示例：
python classification_eval_from_json_dir_final.py ^
  --gold_csv app_classification_eval_dataset_150_gbt39.csv ^
  --pred_dir D:\\桌面\\毕设\\code\\分类json结果 ^
  --output_dir classification_eval_outputs
"""

import json
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

GBT_39_CATEGORIES: List[str] = [
    "地图导航类", "网络约车类", "即时通信类", "网络社区类", "网络支付类",
    "网上购物类", "餐饮外卖类", "邮件快件寄递类", "交通票务类", "婚恋相亲类",
    "求职招聘类", "网络借贷类", "房屋租售类", "二手车交易类", "问诊挂号类",
    "旅游服务类", "酒店服务类", "网络游戏类", "学习教育类", "本地生活类",
    "女性健康类", "用车服务类", "投资理财类", "手机银行类", "邮箱云盘类",
    "远程会议类", "网络直播类", "在线影音类", "短视频类", "新闻资讯类",
    "运动健身类", "浏览器类", "输入法类", "安全管理类", "电子图书类",
    "拍摄美化类", "应用商店类", "实用工具类", "演出票务类",
]

NON39_ALIASES = {
    "非39类", "不属于39类", "非 39 类", "不属于 39 类",
    "none", "non39", "non_39", "other"
}
INVALID_MULTICLASS_PRED = "__INVALID_NON39_OR_UNKNOWN__"


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_binary(x: Any) -> int:
    if isinstance(x, bool):
        return 1 if x else 0

    s = normalize_text(x).lower()
    if s in {"1", "true", "yes", "y", "是", "属于39类", "属于 39 类"}:
        return 1
    if s in {"0", "false", "no", "n", "否", "非39类", "不属于39类", "不属于 39 类"}:
        return 0
    try:
        return 1 if int(float(s)) == 1 else 0
    except Exception:
        raise ValueError(f"无法解析二分类标签：{x}")


def normalize_label_39(x: Any) -> str:
    s = normalize_text(x)
    if s in NON39_ALIASES:
        return "非39类"
    return s


def canonical_name(name: str) -> str:
    """
    用于匹配 CSV 的 app_name 和 JSON 文件名 / JSON 内 app_name。
    """
    name = normalize_text(name)
    name = name.replace(".json", "")
    name = name.replace("（", "(").replace("）", ")")
    name = name.replace(" ", "")
    name = name.lower()
    return name


def load_gold_csv(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(csv_path, encoding="utf-8")

    required = ["app_name", "is_in_39", "label_39"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"金标准 CSV 缺少必要列：{missing}")

    df = df.copy()
    df["app_name"] = df["app_name"].apply(normalize_text)
    df["app_name_canonical"] = df["app_name"].apply(canonical_name)
    df["gold_is_in_39"] = df["is_in_39"].apply(normalize_binary)
    df["gold_label_39"] = df["label_39"].apply(normalize_label_39)
    return df


def extract_prediction_fields(obj: Dict[str, Any], fallback_file_stem: str) -> Dict[str, Any]:
    """
    已按你给的两份 JSON 适配：
    - app_name
    - is_in_39
    - gbt_39_category
    """
    app_name = normalize_text(obj.get("app_name", "")) or fallback_file_stem

    if "is_in_39" not in obj:
        raise ValueError("缺少字段 is_in_39")
    if "gbt_39_category" not in obj:
        raise ValueError("缺少字段 gbt_39_category")

    pred_is = normalize_binary(obj["is_in_39"])
    pred_label = normalize_label_39(obj["gbt_39_category"])

    # 做一层一致性修正
    if pred_is == 0:
        pred_label = "非39类"
    elif pred_label in NON39_ALIASES or pred_label == "":
        pred_label = INVALID_MULTICLASS_PRED

    return {
        "app_name": app_name,
        "app_name_canonical": canonical_name(app_name),
        "pred_is_in_39": pred_is,
        "pred_label_39": pred_label,
    }


def load_predictions_from_dir(pred_dir: str) -> pd.DataFrame:
    pred_dir = Path(pred_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"预测结果目录不存在：{pred_dir}")

    rows = []
    json_files = sorted(pred_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"目录下没有 JSON 文件：{pred_dir}")

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                obj = json.load(f)

            pred = extract_prediction_fields(obj, jf.stem)
            rows.append({
                "app_name_pred": pred["app_name"],
                "app_name_canonical": pred["app_name_canonical"],
                "pred_file": jf.name,
                "pred_is_in_39": pred["pred_is_in_39"],
                "pred_label_39": pred["pred_label_39"],
                "raw_json": json.dumps(obj, ensure_ascii=False),
            })
        except Exception as e:
            rows.append({
                "app_name_pred": jf.stem,
                "app_name_canonical": canonical_name(jf.stem),
                "pred_file": jf.name,
                "pred_is_in_39": None,
                "pred_label_39": None,
                "raw_json": f"JSON_PARSE_ERROR: {e}",
            })

    return pd.DataFrame(rows)


def merge_gold_and_pred(gold_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        gold_df,
        pred_df,
        on="app_name_canonical",
        how="left",
    )
    return merged


def binary_eval(df: pd.DataFrame) -> Dict[str, Any]:
    y_true = df["gold_is_in_39"].tolist()
    y_pred = df["pred_is_in_39"].astype(int).tolist()

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    return {
        "evaluated_samples": int(len(df)),
        "accuracy": round(float(acc), 6),
        "precision": round(float(p), 6),
        "recall": round(float(r), 6),
        "f1": round(float(f1), 6),
        "confusion_matrix": {
            "labels": ["非39类", "39类"],
            "matrix": cm.tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def multiclass_eval(df: pd.DataFrame) -> Dict[str, Any]:
    """
    仅对真实属于39类的样本做类别评估。
    真实非39类样本不纳入多分类统计。
    """
    pos_df = df[df["gold_is_in_39"] == 1].copy()

    y_true = pos_df["gold_label_39"].tolist()
    y_pred = [
        p if p in GBT_39_CATEGORIES else INVALID_MULTICLASS_PRED
        for p in pos_df["pred_label_39"].tolist()
    ]

    strict_acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=GBT_39_CATEGORIES,
        average="macro",
        zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=GBT_39_CATEGORIES,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_true, y_pred,
        labels=GBT_39_CATEGORIES,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=GBT_39_CATEGORIES)

    return {
        "evaluated_true39_samples": int(len(pos_df)),
        "accuracy_strict_on_true39": round(float(strict_acc), 6),
        "macro_precision": round(float(p_macro), 6),
        "macro_recall": round(float(r_macro), 6),
        "macro_f1": round(float(f1_macro), 6),
        "weighted_precision": round(float(p_weighted), 6),
        "weighted_recall": round(float(r_weighted), 6),
        "weighted_f1": round(float(f1_weighted), 6),
        "classification_report_by_label": report,
        "confusion_matrix_labels": GBT_39_CATEGORIES,
        "confusion_matrix": cm.tolist(),
    }


def build_error_tables(df: pd.DataFrame):
    out = df.copy()
    out["binary_correct"] = (out["gold_is_in_39"] == out["pred_is_in_39"]).astype(int)
    out["multiclass_correct"] = (
        (out["gold_is_in_39"] == 1) &
        (out["gold_label_39"] == out["pred_label_39"])
    ).astype(int)

    binary_errors = out[out["binary_correct"] == 0].copy()
    multiclass_errors = out[(out["gold_is_in_39"] == 1) & (out["multiclass_correct"] == 0)].copy()

    keep_cols = [
        c for c in [
            "id", "app_name", "description",
            "gold_is_in_39", "gold_label_39",
            "pred_file", "pred_is_in_39", "pred_label_39",
            "rationale", "evidence_span", "raw_json"
        ] if c in out.columns
    ]
    return binary_errors[keep_cols], multiclass_errors[keep_cols]


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv", required=True, help="金标准 CSV 路径")
    parser.add_argument("--pred_dir", required=True, help="分类 JSON 目录")
    parser.add_argument("--output_dir", default="classification_eval_outputs", help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_df = load_gold_csv(args.gold_csv)
    pred_df = load_predictions_from_dir(args.pred_dir)
    merged = merge_gold_and_pred(gold_df, pred_df)

    matched = merged["pred_file"].notna().sum()
    unmatched = len(merged) - matched

    # 只对成功匹配并成功解析的样本做评估
    valid_df = merged.dropna(subset=["pred_is_in_39", "pred_label_39"]).copy()

    binary_report = binary_eval(valid_df)
    multi_report = multiclass_eval(valid_df)
    binary_errors, multiclass_errors = build_error_tables(valid_df)

    summary = {
        "gold_csv": args.gold_csv,
        "pred_dir": args.pred_dir,
        "gold_total": int(len(gold_df)),
        "matched_json_count": int(matched),
        "unmatched_count": int(unmatched),
        "valid_eval_count": int(len(valid_df)),
        "binary_evaluation": binary_report,
        "multiclass_evaluation_true39_only": multi_report,
    }

    save_json(summary, output_dir / "classification_eval_report.json")

    valid_df.to_csv(output_dir / "classification_eval_predictions_full.csv", index=False, encoding="utf-8-sig")
    binary_errors.to_csv(output_dir / "binary_errors.csv", index=False, encoding="utf-8-sig")
    multiclass_errors.to_csv(output_dir / "multiclass_errors_true39_only.csv", index=False, encoding="utf-8-sig")

    merged[merged["pred_file"].isna()].to_csv(
        output_dir / "unmatched_gold_samples.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # confusion matrix 导出
    binary_cm_df = pd.DataFrame(
        binary_report["confusion_matrix"]["matrix"],
        index=binary_report["confusion_matrix"]["labels"],
        columns=binary_report["confusion_matrix"]["labels"],
    )
    binary_cm_df.reset_index().rename(columns={"index": "gold\\pred"}).to_csv(
        output_dir / "binary_confusion_matrix.csv",
        index=False,
        encoding="utf-8-sig"
    )

    multi_cm_df = pd.DataFrame(
        multi_report["confusion_matrix"],
        index=multi_report["confusion_matrix_labels"],
        columns=multi_report["confusion_matrix_labels"],
    )
    multi_cm_df.reset_index().rename(columns={"index": "gold\\pred"}).to_csv(
        output_dir / "multiclass_confusion_matrix_true39_only.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 72)
    print(f"金标准样本数: {len(gold_df)}")
    print(f"匹配到 JSON 的样本数: {matched}")
    print(f"未匹配样本数: {unmatched}")
    print(f"有效评估样本数: {len(valid_df)}")
    print("=" * 72)
    print("二分类评估结果")
    print(json.dumps(binary_report, ensure_ascii=False, indent=2))
    print("=" * 72)
    print("多分类评估结果（仅真实39类）")
    print(json.dumps({
        k: v for k, v in multi_report.items()
        if k not in {"classification_report_by_label", "confusion_matrix"}
    }, ensure_ascii=False, indent=2))
    print("=" * 72)
    print(f"结果已输出到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
