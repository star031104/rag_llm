# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from common_ablation import (
    ExperimentContext,
    LLMClient,
    dedup_keep_order,
    extract_json_payload,
    normalize_space,
    safe_read_text,
    safe_write_text,
    shorten,
    split_sentences,
    try_import_retrieve,
    try_json_load,
    dump_json,
)
from exp_config import ExperimentConfig, get_experiment


ALLOWED_STATUSES = {
    "业务核心必要",
    "业务辅助合理",
    "国标未直接枚举但可解释",
    "业务弱相关需审查",
    "高风险异常权限",
}


class Analyzer:
    def __init__(self, base_dir: Path, experiment: ExperimentConfig):
        self.base_dir = Path(base_dir).resolve()
        self.exp = experiment
        self.ctx = ExperimentContext(experiment, self.base_dir)
        self.category_dir = self.base_dir / "dataset" / "分类"
        self.policy_dir = self.base_dir / "dataset" / "隐私政策"
        self.apk_mapping_dir = self.base_dir / "dataset" / "apk映射"
        self.semantic_cache_file = self.base_dir / "dataset" / "权限语义缓存" / f"semantic_cache_{experiment.name}.json"
        self.retrieve = try_import_retrieve(experiment)
        self.llm = LLMClient(system_prompt="你是一名移动应用国家标准与权限必要性联合分析专家。")

    def safe_json_object(self, text: str) -> Dict[str, Any]:
        obj = extract_json_payload(text)
        return obj if isinstance(obj, dict) else {}

    def normalize_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "是", "属于", "39类", "in_39"}
        return False

    def extract_app_records_from_json_data(self, data: Any) -> List[Dict]:
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for key in ["results", "apps", "data", "items", "app_list"]:
                if key in data and isinstance(data[key], list):
                    return [x for x in data[key] if isinstance(x, dict)]
            return [data]
        return []

    def normalize_app_record(self, record: Dict, fallback_name: str = "") -> Dict[str, Any]:
        app_name = record.get("app_name") or record.get("app") or record.get("name") or record.get("appName") or record.get("应用名称") or fallback_name
        is_in_39 = record.get("is_in_39", record.get("in_39", record.get("是否属于39类", record.get("属于39类", False))))
        category = record.get("gbt_39_category") or record.get("category") or record.get("app_category") or record.get("39_category") or record.get("类别") or record.get("39类类别") or "未分类"
        return {"app_name": str(app_name).strip(), "is_in_39": self.normalize_bool(is_in_39), "gbt_39_category": str(category).strip()}

    def load_all_category_records(self) -> List[Dict[str, Any]]:
        files = sorted(self.category_dir.glob("*.json"))
        merged: Dict[str, Dict[str, Any]] = {}
        for jf in files:
            try:
                data = json.loads(safe_read_text(jf))
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] 跳过无效分类文件 {jf.name}: {exc}")
                continue
            for raw in self.extract_app_records_from_json_data(data):
                rec = self.normalize_app_record(raw, fallback_name=jf.stem)
                if rec["app_name"]:
                    merged[rec["app_name"]] = rec
        return sorted(merged.values(), key=lambda x: x["app_name"])

    def find_permission_file(self, app_name: str) -> Optional[Path]:
        for p in [self.apk_mapping_dir / f"{app_name}.txt", self.apk_mapping_dir / f"{app_name}.apk.txt"]:
            if p.exists():
                return p
        return None

    def parse_permission_line(self, line: str) -> Dict[str, str]:
        raw = line.strip()
        if not raw:
            return {}
        if "\t" in raw:
            a, b = raw.split("\t", 1)
            return {"raw_line": raw, "permission_name": a.strip(), "permission_desc_en": b.strip()}
        m = re.match(r"^([A-Za-z0-9_\.]+)\s+(.*)$", raw)
        if m and (m.group(1).startswith("android.permission.") or ".permission." in m.group(1)):
            return {"raw_line": raw, "permission_name": m.group(1).strip(), "permission_desc_en": m.group(2).strip()}
        return {"raw_line": raw, "permission_name": raw, "permission_desc_en": ""}

    def load_permission_records(self, app_name: str) -> List[Dict[str, str]]:
        fp = self.find_permission_file(app_name)
        if fp is None:
            return []
        seen = set()
        records: List[Dict[str, str]] = []
        for line in safe_read_text(fp).splitlines():
            item = self.parse_permission_line(line)
            if not item:
                continue
            name = item["permission_name"]
            if name in seen:
                continue
            seen.add(name)
            records.append(item)
        return records

    def load_semantic_cache(self) -> Dict[str, Dict[str, Any]]:
        cache = try_json_load(self.semantic_cache_file, {})
        return cache if isinstance(cache, dict) else {}

    def save_semantic_cache(self, cache: Dict[str, Dict[str, Any]]) -> None:
        dump_json(self.semantic_cache_file, cache)

    def normalize_permission_semantic(self, permission_name: str, permission_desc_en: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not self.exp.use_permission_semantic_enrichment:
            return {
                "permission_name": permission_name,
                "permission_cn": permission_name,
                "data_type": "不明确",
                "capability": "不明确",
                "usage_hint": "不明确",
                "uncertainty": "高（消融实验关闭语义增强）",
            }
        key = f"{permission_name}|||{permission_desc_en}"
        if key in cache:
            return cache[key]
        if self.exp.use_prompt_constraint:
            prompt = (
                "你是一名 Android 权限语义解析助手。请根据 APK 权限记录输出 JSON，字段："
                "permission_name、permission_cn、data_type、capability、usage_hint、uncertainty。"
                f" 输入：权限名={permission_name}；英文解释={permission_desc_en or '（无）'}"
            )
        else:
            prompt = (
                "请解释这个 Android 权限，尽量返回 JSON，字段包含 permission_cn、data_type、capability、usage_hint、uncertainty。"
                f" 权限名={permission_name}；英文解释={permission_desc_en or '（无）'}"
            )
        obj = self.safe_json_object(self.llm.chat(prompt, temperature=0.05))
        result = {
            "permission_name": permission_name,
            "permission_cn": obj.get("permission_cn", permission_name),
            "data_type": obj.get("data_type", "不明确"),
            "capability": obj.get("capability", "不明确"),
            "usage_hint": obj.get("usage_hint", "不明确"),
            "uncertainty": obj.get("uncertainty", "不明确"),
        }
        cache[key] = result
        self.save_semantic_cache(cache)
        return result

    def enrich_permissions_with_semantics(self, permission_records: List[Dict[str, str]], cache: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched = []
        for item in permission_records:
            semantic = self.normalize_permission_semantic(item["permission_name"], item.get("permission_desc_en", ""), cache)
            enriched.append({
                "raw_line": item["raw_line"],
                "permission_name": item["permission_name"],
                "permission_desc_en": item.get("permission_desc_en", ""),
                "permission_cn": semantic.get("permission_cn", item["permission_name"]),
                "data_type": semantic.get("data_type", "不明确"),
                "capability": semantic.get("capability", "不明确"),
                "usage_hint": semantic.get("usage_hint", "不明确"),
                "uncertainty": semantic.get("uncertainty", "不明确"),
            })
        return enriched

    def retrieve_texts(self, queries: List[str], retrieve_n: int = 50, topk: int = 8, intent: str = "general") -> List[str]:
        if not self.retrieve or not self.exp.use_rag_retrieval:
            return []
        merged: List[str] = []
        seen = set()
        actual_intent = intent if self.exp.use_intent_retrieval else "general"
        for q in queries:
            if not q.strip():
                continue
            try:
                results = self.retrieve(query=q, intent=actual_intent, retrieve_n=retrieve_n, topk=topk)
            except TypeError:
                results = self.retrieve(q)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] RAG 检索失败: {q} -> {exc}")
                continue
            for r in results or []:
                text = (r.get("text", "") or "").strip()
                src = (r.get("meta", {}) or {}).get("section_title", "")
                combined = f"{text}\n> 来源：{src}".strip()
                norm = re.sub(r"\s+", " ", combined)
                if text and norm not in seen:
                    seen.add(norm)
                    merged.append(combined)
        return merged[:12]

    def retrieve_rules_for_app(self, category: str, is_in_39: bool) -> Dict[str, List[str]]:
        general_rules = self.retrieve_texts([
            "移动应用 权限申请 最小必要 原则 合理相关",
            "移动互联网应用程序 个人信息 最小必要 权限 申请",
            "GB/T 41391 权限申请 最小必要 合理相关",
        ], intent="general")
        category_rules: List[str] = []
        typical_features: List[str] = []
        if is_in_39 and category and category != "未分类" and self.exp.use_category_rules:
            category_rules = self.retrieve_texts([
                f"{category} GB/T 41391 基本业务功能 必要个人信息 表",
                f"{category} 权限申请 最小必要 合理相关",
            ], intent="profile")
        if is_in_39 and category and category != "未分类" and self.exp.use_typical_feature_hints:
            typical_features = self.retrieve_texts([
                f"{category} 应用 常见功能 场景 图片 视频 录音 定位 上传 下载",
                f"{category} app 常见业务功能 功能场景",
            ], intent="general")
        return {"general_rules": general_rules, "category_rules": category_rules, "typical_features": typical_features}

    def infer_keywords_for_permission(self, p: Dict[str, Any]) -> List[str]:
        raw = " ".join([
            str(p.get("permission_name") or ""),
            str(p.get("permission_cn") or ""),
            str(p.get("data_type") or ""),
            str(p.get("capability") or ""),
            str(p.get("usage_hint") or ""),
        ]).lower()
        keywords = []
        mappings = {
            "相机": ["相机", "摄像头", "拍照", "扫码"],
            "麦克风": ["录音", "音频", "语音", "麦克风"],
            "位置": ["定位", "位置", "gps", "附近"],
            "存储": ["相册", "存储", "文件", "图片", "视频"],
            "网络": ["网络", "联网", "wifi", "ip地址"],
            "通知": ["通知", "消息提醒", "角标"],
            "设备": ["imei", "imsi", "oaid", "android id", "设备标识"],
        }
        for _, kws in mappings.items():
            if any(kw.lower() in raw for kw in kws):
                keywords.extend(kws)
        return dedup_keep_order(keywords)[:10]

    def collect_policy_evidence(self, permission_item: Dict[str, Any], policy_text: str, max_items: int = 8) -> List[str]:
        if not self.exp.use_policy_evidence_in_gb_perm or not policy_text.strip():
            return []
        keywords = self.infer_keywords_for_permission(permission_item)
        scored = []
        for s in split_sentences(policy_text):
            sl = s.lower()
            score = sum(1 for kw in keywords if kw.lower() in sl)
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: (-x[0], len(x[1])))
        result, seen = [], set()
        for _, s in scored:
            norm = re.sub(r"\s+", " ", s)
            if norm in seen:
                continue
            seen.add(norm)
            result.append(shorten(s, 220))
            if len(result) >= max_items:
                break
        return result

    def build_prompt(self, app_name: str, category: str, is_in_39: bool, permission_item: Dict[str, Any], rules_bundle: Dict[str, List[str]], policy_evidence: List[str]) -> str:
        rule_text = "\n".join(f"- {x}" for x in (rules_bundle.get("general_rules", []) + rules_bundle.get("category_rules", []))) or "- 未检索到相关国标条款"
        feature_text = "\n".join(f"- {x}" for x in rules_bundle.get("typical_features", [])) or "- 未检索到该类别典型功能描述"
        evidence_text = "\n".join(f"- {x}" for x in policy_evidence) if policy_evidence else "- 未从隐私政策中召回到直接证据"
        category_line = category if is_in_39 else "非39类/未分类"
        if self.exp.use_prompt_constraint:
            return f"""
你是一名移动应用“国家标准 + 业务功能 + 权限必要性”联合分析助手。
请对单个权限进行审慎分析，输出 JSON：
{{
  "status": "业务核心必要/业务辅助合理/国标未直接枚举但可解释/业务弱相关需审查/高风险异常权限",
  "reason": "...",
  "rule_basis": "...",
  "confidence": "高/中/低",
  "risk_note": "...",
  "evidence": ["..."]
}}

应用名称：{app_name}
是否39类：{is_in_39}
类别：{category_line}
权限名称：{permission_item['permission_name']}
中文解释：{permission_item['permission_cn']}
数据类型：{permission_item['data_type']}
核心能力：{permission_item['capability']}
常见场景：{permission_item['usage_hint']}
语义不确定性：{permission_item['uncertainty']}

国家标准/原则证据：
{rule_text}

类别典型功能提示：
{feature_text}

隐私政策/业务证据：
{evidence_text}
""".strip()
        return f"""
请结合应用类别、权限信息、国标证据和业务证据，分析这个权限是否合理，并说明原因。
尽量返回 JSON，字段包含 status、reason、rule_basis、confidence、risk_note、evidence。

应用名称：{app_name}
是否39类：{is_in_39}
类别：{category_line}
权限信息：{json.dumps(permission_item, ensure_ascii=False)}
国家标准/原则证据：
{rule_text}
类别典型功能提示：
{feature_text}
隐私政策/业务证据：
{evidence_text}
""".strip()

    def conservative_adjust(self, status: str, permission_name: str, evidence: List[str]) -> str:
        if not self.exp.use_risk_calibration:
            return status
        perm_upper = permission_name.upper()
        risky = any(x in perm_upper for x in ["READ_SMS", "RECORD_AUDIO", "QUERY_ALL_PACKAGES", "ACCESS_FINE_LOCATION", "READ_CONTACTS", "SYSTEM_ALERT_WINDOW"])
        if risky and not evidence and status in {"业务核心必要", "业务辅助合理"}:
            return "业务弱相关需审查"
        return status

    def analyze_permission(self, app_name: str, category: str, is_in_39: bool, permission_item: Dict[str, Any], rules_bundle: Dict[str, List[str]], policy_text: str) -> Dict[str, Any]:
        policy_evidence = self.collect_policy_evidence(permission_item, policy_text)
        obj = self.safe_json_object(self.llm.chat(self.build_prompt(app_name, category, is_in_39, permission_item, rules_bundle, policy_evidence), temperature=0.1))
        status = obj.get("status", "业务弱相关需审查")
        if status not in ALLOWED_STATUSES:
            status = "业务弱相关需审查"
        status = self.conservative_adjust(status, permission_item["permission_name"], policy_evidence)
        return {
            "permission_name": permission_item["permission_name"],
            "permission_cn": permission_item.get("permission_cn", permission_item["permission_name"]),
            "status": status,
            "reason": normalize_space(str(obj.get("reason", "模型未返回充分说明"))),
            "rule_basis": normalize_space(str(obj.get("rule_basis", "通用原则/检索证据"))),
            "confidence": obj.get("confidence", "中"),
            "risk_note": normalize_space(str(obj.get("risk_note", "建议结合实际功能进一步核查"))),
            "evidence": obj.get("evidence", policy_evidence[:3]) if isinstance(obj.get("evidence"), list) else policy_evidence[:3],
        }

    def pick_key_risks(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        order = {"高风险异常权限": 0, "业务弱相关需审查": 1, "国标未直接枚举但可解释": 2, "业务辅助合理": 3, "业务核心必要": 4}
        return sorted(results, key=lambda x: (order.get(x["status"], 9), x["permission_name"]))[:8]

    def render_report(self, app_record: Dict[str, Any], rules_bundle: Dict[str, List[str]], results: List[Dict[str, Any]]) -> str:
        app_name = app_record["app_name"]
        category = app_record["gbt_39_category"]
        is_in_39 = app_record["is_in_39"]
        counts = Counter(r["status"] for r in results)
        lines = [
            "# 国标和权限合规分析报告（消融实验版）",
            f"- 应用名称：{app_name}",
            f"- 实验编号：{self.exp.name}",
            f"- 实验说明：{self.exp.description}",
            f"- 是否39类：{is_in_39}",
            f"- 类别：{category}",
            "",
            "## 一、分析方法说明",
            "本版本通过统一实验开关控制 RAG、权限语义增强、类别规则、业务证据、风险校准等模块。",
            "",
            "## 二、国标检索摘要",
        ]
        if rules_bundle.get("general_rules"):
            lines.append("### 2.1 通用原则")
            for x in rules_bundle["general_rules"][:4]:
                lines.append(f"- {shorten(x, 220)}")
        if rules_bundle.get("category_rules"):
            lines.append("\n### 2.2 类别相关条款")
            for x in rules_bundle["category_rules"][:5]:
                lines.append(f"- {shorten(x, 220)}")
        if rules_bundle.get("typical_features"):
            lines.append("\n### 2.3 类别典型功能提示")
            for x in rules_bundle["typical_features"][:4]:
                lines.append(f"- {shorten(x, 220)}")
        lines.extend([
            "",
            "## 三、分析结果",
            f"- 业务核心必要：{counts.get('业务核心必要', 0)} 个",
            f"- 业务辅助合理：{counts.get('业务辅助合理', 0)} 个",
            f"- 国标未直接枚举但可解释：{counts.get('国标未直接枚举但可解释', 0)} 个",
            f"- 业务弱相关需审查：{counts.get('业务弱相关需审查', 0)} 个",
            f"- 高风险异常权限：{counts.get('高风险异常权限', 0)} 个",
            "",
            "| 权限名称 | 中文解释 | 国标条款或依据 | 判断状态 | 说明 | 置信度 |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for r in results:
            lines.append(f"| {r['permission_name']} | {r['permission_cn']} | {shorten(r['rule_basis'], 60)} | {r['status']} | {shorten(r['reason'], 80)} | {r['confidence']} |")
        lines.append("\n## 四、重点权限审查")
        for item in self.pick_key_risks(results):
            lines.append(f"- **{item['permission_name']}（{item['status']}）**：{item['reason']} 风险提示：{item['risk_note']}。")
            for ev in item.get("evidence", [])[:2]:
                lines.append(f"  - 证据：{shorten(str(ev), 180)}")
        return "\n".join(lines).strip() + "\n"

    def analyze_app(self, app_record: Dict[str, Any]) -> Optional[Path]:
        app_name = app_record["app_name"]
        permission_records = self.load_permission_records(app_name)
        if not permission_records:
            print(f"[WARN] 未读取到 APK 映射权限：{app_name}")
            return None
        self.ctx.save_manifest(app_name)
        semantic_permissions = self.enrich_permissions_with_semantics(permission_records, self.load_semantic_cache())
        policy_text = safe_read_text(self.policy_dir / f"{app_name}.txt")
        rules_bundle = self.retrieve_rules_for_app(app_record["gbt_39_category"], app_record["is_in_39"])
        results = [self.analyze_permission(app_name, app_record["gbt_39_category"], app_record["is_in_39"], p, rules_bundle, policy_text) for p in semantic_permissions]
        report = self.render_report(app_record, rules_bundle, results)
        out = self.ctx.app_output_dir(app_name) / "国标和权限分析结果.md"
        safe_write_text(out, report)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=".")
    ap.add_argument("--experiment", default="baseline")
    ap.add_argument("--app", default="")
    args = ap.parse_args()
    analyzer = Analyzer(Path(args.base_dir), get_experiment(args.experiment))
    if args.app.strip():
        records = [
            {"app_name": args.app.strip(), "is_in_39": False, "gbt_39_category": "未分类"}
        ]
        for rec in analyzer.load_all_category_records():
            if rec["app_name"] == args.app.strip():
                records = [rec]
                break
    else:
        records = analyzer.load_all_category_records()
    for rec in records:
        print(f"[RUN] 国标和权限分析: {rec['app_name']}")
        analyzer.analyze_app(rec)


if __name__ == "__main__":
    main()
