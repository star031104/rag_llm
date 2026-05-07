# -*- coding: utf-8 -*-
"""
三类报告自动化质量评估（仅自动化部分）

评估对象：
1. 国标-隐私政策分析报告
2. 隐私政策-权限分析报告
3. 国标-权限分析报告

目录结构：
output/
  app1/
    国标和隐私政策分析结果.md
    隐私政策和权限分析结果.md
    国标和权限分析结果.md
  app2/
    ...

自动化指标：
1. 结构完整率（Structure Completeness）
2. 证据存在率（Evidence Presence Rate）

说明：
- 不做结论一致率分析
- 只评估报告本体质量
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# =========================================================
# 直接改这里
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "output"
RESULT_DIR = BASE_DIR / "report_quality_outputs"

STD_POLICY_REPORT = "国标和隐私政策分析结果.md"
POLICY_PERMISSION_REPORT = "隐私政策和权限分析结果.md"
STD_PERMISSION_REPORT = "国标和权限分析结果.md"
# =========================================================


def read_text(path: Path) -> str:
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return path.read_text(errors="ignore")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u3000", " ")).strip()


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def find_app_dirs(output_root: Path) -> List[Path]:
    if not output_root.exists():
        return []
    return [p for p in sorted(output_root.iterdir()) if p.is_dir()]


# =========================================================
# 1. 国标-隐私政策分析报告
# =========================================================

STD_POLICY_REQUIRED_SECTIONS = [
    "## A1",
    "## B1",
    "## C2",
    "## D2",
    "## F1",
    "### 子义务判断结果",
    "### 总体合规结论",
    "### 高风险问题总结",
    "### 整改建议",
]

STD_POLICY_CONTROL_POINTS = ["A1", "B1", "C2", "D2", "F1"]


def parse_std_policy_metrics(app_name: str, report_path: Path) -> Dict:
    text = read_text(report_path)

    # 结构完整率
    structure_map = {sec: (sec in text) for sec in STD_POLICY_REQUIRED_SECTIONS}
    structure_total = len(structure_map)
    structure_hit = sum(structure_map.values())
    structure_rate = structure_hit / structure_total if structure_total else 0.0

    # 证据存在率：每个控制点是否有理由/风险/建议
    evidence_total = 0
    evidence_hit = 0

    for cp in STD_POLICY_CONTROL_POINTS:
        m_block = re.search(rf"(?ms)^##\s*{re.escape(cp)}\b(.*?)(?=^##\s*[A-Z]\d\b|\Z)", text)
        if not m_block:
            continue
        block = m_block.group(1)

        # 有总体结论才算一个有效控制点
        has_conclusion = bool(re.search(r"总体(?:合规)?结论[：:]\s*(满足|部分满足|不满足)", block))
        if not has_conclusion:
            continue

        evidence_total += 1

        has_reason = bool(re.search(r"理由[：:]", block))
        has_risk = "高风险问题总结" in block
        has_advice = "整改建议" in block

        if has_reason or has_risk or has_advice:
            evidence_hit += 1

    evidence_rate = evidence_hit / evidence_total if evidence_total else 0.0

    return {
        "app_name": app_name,
        "report_type": "国标-隐私政策",
        "structure_total": structure_total,
        "structure_hit": structure_hit,
        "structure_rate": round(structure_rate, 6),
        "evidence_total": evidence_total,
        "evidence_hit": evidence_hit,
        "evidence_rate": round(evidence_rate, 6),
        "report_path": str(report_path),
    }


# =========================================================
# 2. 隐私政策-权限分析报告
# =========================================================

POLICY_PERMISSION_REQUIRED_SECTIONS = [
    "## 一、分析说明",
    "## 二、总体统计",
    "## 三、分层统计",
    "## 四、隐私政策能力块提取结果",
    "## 五、逐权限分析结果",
]


def parse_policy_permission_metrics(app_name: str, report_path: Path) -> Dict:
    text = read_text(report_path)

    # 结构完整率
    structure_map = {sec: (sec in text) for sec in POLICY_PERMISSION_REQUIRED_SECTIONS}
    structure_total = len(structure_map)
    structure_hit = sum(structure_map.values())
    structure_rate = structure_hit / structure_total if structure_total else 0.0

    # 证据存在率：每个权限块是否带证据
    evidence_total = 0
    evidence_hit = 0

    matches = list(re.finditer(r"(?ms)^###\s*\d+\.\s*([^\n]+?)\s*$", text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        # 有合规状态才算一个有效权限条目
        has_status = bool(
            re.search(r"合规状态[：:]\s*\*\*(已明确告知|间接告知|未告知)\*\*", block)
            or re.search(r"合规状态[：:]\s*(已明确告知|间接告知|未告知)", block)
        )
        if not has_status:
            continue

        evidence_total += 1
        if "证据：" in block or "- 证据：" in block:
            evidence_hit += 1

    evidence_rate = evidence_hit / evidence_total if evidence_total else 0.0

    return {
        "app_name": app_name,
        "report_type": "隐私政策-权限",
        "structure_total": structure_total,
        "structure_hit": structure_hit,
        "structure_rate": round(structure_rate, 6),
        "evidence_total": evidence_total,
        "evidence_hit": evidence_hit,
        "evidence_rate": round(evidence_rate, 6),
        "report_path": str(report_path),
    }


# =========================================================
# 3. 国标-权限分析报告
# =========================================================

STD_PERMISSION_REQUIRED_SECTIONS = [
    "## 一、基本信息",
    "## 二、分析方法说明",
    "## 三、国标检索摘要",
    "## 四、分析结果",
    "### 4.1 总体判断",
    "### 4.2 权限匹配对比分析",
    "### 4.3 重点权限审查",
    "### 4.4 重总结性结论",
]

STD_PERMISSION_LABELS = {
    "业务核心必要",
    "业务辅助合理",
    "国标未直接枚举但可解释",
    "业务弱相关需审查",
    "高风险异常权限",
}


def parse_markdown_table(section_text: str) -> List[List[str]]:
    rows = []
    for line in section_text.splitlines():
        raw = line.strip()
        if not raw.startswith("|"):
            continue
        cells = [clean_text(x) for x in raw.strip("|").split("|")]
        if not cells:
            continue
        joined = "".join(cells)
        if re.fullmatch(r"[-: ]+", joined):
            continue
        rows.append(cells)
    return rows


def parse_std_permission_metrics(app_name: str, report_path: Path) -> Dict:
    text = read_text(report_path)

    # 结构完整率
    structure_map = {sec: (sec in text) for sec in STD_PERMISSION_REQUIRED_SECTIONS}
    structure_total = len(structure_map)
    structure_hit = sum(structure_map.values())
    structure_rate = structure_hit / structure_total if structure_total else 0.0

    # 证据存在率：4.2 表中每条权限判断是否具备“说明/依据”
    evidence_total = 0
    evidence_hit = 0

    m = re.search(
        r"(?ms)^###\s*4\.2\s*权限匹配对比分析\s*$"
        r"(.*?)"
        r"(?=^###\s*4\.3\b|^##\s*五[、.]|^##\s*5\b|^###\s*5\b|\Z)",
        text,
    )
    if not m:
        m = re.search(
            r"(?ms)^##\s*4\.2\s*权限匹配对比分析\s*$"
            r"(.*?)"
            r"(?=^##\s*4\.3\b|^##\s*五[、.]|^##\s*5\b|\Z)",
            text,
        )

    section = m.group(1) if m else ""
    if section:
        rows = parse_markdown_table(section)
        if len(rows) >= 2:
            header = rows[0]
            header_map = {clean_text(col): idx for idx, col in enumerate(header)}

            status_idx = -1
            reason_idx = -1

            for k in ["判断状态", "匹配状态", "状态"]:
                if k in header_map:
                    status_idx = header_map[k]
                    break

            for k in ["说明", "规范性说明", "依据说明", "理由"]:
                if k in header_map:
                    reason_idx = header_map[k]
                    break

            for row in rows[1:]:
                if status_idx == -1 or len(row) <= status_idx:
                    continue

                label = clean_text(row[status_idx])
                if label not in STD_PERMISSION_LABELS:
                    continue

                evidence_total += 1

                has_reason = False
                if reason_idx != -1 and len(row) > reason_idx:
                    reason_text = clean_text(row[reason_idx])
                    has_reason = bool(reason_text)

                if has_reason:
                    evidence_hit += 1

    evidence_rate = evidence_hit / evidence_total if evidence_total else 0.0

    return {
        "app_name": app_name,
        "report_type": "国标-权限",
        "structure_total": structure_total,
        "structure_hit": structure_hit,
        "structure_rate": round(structure_rate, 6),
        "evidence_total": evidence_total,
        "evidence_hit": evidence_hit,
        "evidence_rate": round(evidence_rate, 6),
        "report_path": str(report_path),
    }


# =========================================================
# 汇总函数
# =========================================================

def summarize(df: pd.DataFrame, report_type: str) -> Dict:
    if df.empty:
        return {
            "report_type": report_type,
            "report_count": 0,
            "avg_structure_rate": 0.0,
            "avg_evidence_rate": 0.0,
            "total_evidence_items": 0,
            "total_evidence_hits": 0,
            "global_evidence_rate": 0.0,
        }

    total_evidence_items = int(df["evidence_total"].sum())
    total_evidence_hits = int(df["evidence_hit"].sum())

    return {
        "report_type": report_type,
        "report_count": int(len(df)),
        "avg_structure_rate": round(float(df["structure_rate"].mean()), 6),
        "avg_evidence_rate": round(float(df["evidence_rate"].mean()), 6),
        "total_evidence_items": total_evidence_items,
        "total_evidence_hits": total_evidence_hits,
        "global_evidence_rate": round(total_evidence_hits / total_evidence_items, 6) if total_evidence_items else 0.0,
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    app_dirs = find_app_dirs(OUTPUT_ROOT)
    if not app_dirs:
        print(f"[ERROR] 未找到 output 目录或其中没有 app 子目录：{OUTPUT_ROOT}")
        return

    std_policy_rows = []
    policy_permission_rows = []
    std_permission_rows = []

    for app_dir in app_dirs:
        app_name = app_dir.name

        # 1. 国标-隐私政策
        std_policy_path = app_dir / STD_POLICY_REPORT
        if std_policy_path.exists():
            std_policy_rows.append(parse_std_policy_metrics(app_name, std_policy_path))

        # 2. 隐私政策-权限
        policy_perm_path = app_dir / POLICY_PERMISSION_REPORT
        if policy_perm_path.exists():
            policy_permission_rows.append(parse_policy_permission_metrics(app_name, policy_perm_path))

        # 3. 国标-权限
        std_perm_path = app_dir / STD_PERMISSION_REPORT
        if std_perm_path.exists():
            std_permission_rows.append(parse_std_permission_metrics(app_name, std_perm_path))

    std_policy_df = pd.DataFrame(std_policy_rows)
    policy_permission_df = pd.DataFrame(policy_permission_rows)
    std_permission_df = pd.DataFrame(std_permission_rows)

    std_policy_summary = summarize(std_policy_df, "国标-隐私政策")
    policy_permission_summary = summarize(policy_permission_df, "隐私政策-权限")
    std_permission_summary = summarize(std_permission_df, "国标-权限")

    # 导出逐报告明细
    std_policy_df.to_csv(RESULT_DIR / "std_policy_report_quality_detail.csv", index=False, encoding="utf-8-sig")
    policy_permission_df.to_csv(RESULT_DIR / "policy_permission_report_quality_detail.csv", index=False, encoding="utf-8-sig")
    std_permission_df.to_csv(RESULT_DIR / "std_permission_report_quality_detail.csv", index=False, encoding="utf-8-sig")

    # 导出 summary
    save_json(std_policy_summary, RESULT_DIR / "std_policy_report_quality_summary.json")
    save_json(policy_permission_summary, RESULT_DIR / "policy_permission_report_quality_summary.json")
    save_json(std_permission_summary, RESULT_DIR / "std_permission_report_quality_summary.json")

    all_summary_df = pd.DataFrame([
        std_policy_summary,
        policy_permission_summary,
        std_permission_summary,
    ])
    all_summary_df.to_csv(RESULT_DIR / "all_report_quality_summary.csv", index=False, encoding="utf-8-sig")

    print("=" * 90)
    print("三类报告自动化质量评估完成（仅结构完整率 + 证据存在率）")
    print("=" * 90)
    print(all_summary_df.to_string(index=False))
    print("=" * 90)
    print(f"输出目录: {RESULT_DIR.resolve()}")
    print("=" * 90)


if __name__ == "__main__":
    main()