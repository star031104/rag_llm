# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from common_ablation import ExperimentContext, LLMClient, safe_read_text, safe_write_text, shorten, try_import_retrieve
from exp_config import ExperimentConfig, get_experiment


COMPLIANCE_CONTROLS = [
    {
        "id": "A1",
        "dimension": "基本合规原则",
        "control": "隐私政策适用范围",
        "standard": "GB/T 35273",
        "sub_controls": [
            {"name": "是否明确个人信息处理主体", "risk": "高"},
            {"name": "是否覆盖核心功能与服务", "risk": "高"},
            {"name": "是否覆盖边缘功能与特定终端形态", "risk": "中"},
        ],
        "query": "隐私政策适用范围处理主体服务范围",
        "intent": "definition",
    },
    {
        "id": "B1",
        "dimension": "个人信息收集",
        "control": "个人信息类型与必要性",
        "standard": "GB/T 35273 / GB/T 41391",
        "sub_controls": [
            {"name": "是否逐项列明收集的个人信息类型", "risk": "高"},
            {"name": "是否说明每类信息的对应处理目的", "risk": "高"},
            {"name": "是否说明非必要信息可拒绝", "risk": "中"},
        ],
        "query": "个人信息类型收集必要性",
        "intent": "profile",
    },
    {
        "id": "C2",
        "dimension": "敏感个人信息",
        "control": "敏感个人信息单独同意",
        "standard": "GB/T 45574",
        "sub_controls": [
            {"name": "是否识别敏感个人信息", "risk": "高"},
            {"name": "是否取得单独同意", "risk": "高"},
            {"name": "是否说明对个人权益的影响", "risk": "中"},
        ],
        "query": "敏感个人信息单独同意",
        "intent": "general",
    },
    {
        "id": "D2",
        "dimension": "第三方与SDK",
        "control": "第三方SDK合规披露",
        "standard": "GB/T 43435",
        "sub_controls": [
            {"name": "是否披露SDK名称与功能", "risk": "高"},
            {"name": "是否说明SDK收集信息类型", "risk": "高"},
            {"name": "是否限制为最少必要SDK", "risk": "中"},
            {"name": "是否取得用户同意", "risk": "高"},
        ],
        "query": "第三方SDK披露个人信息",
        "intent": "general",
    },
    {
        "id": "F1",
        "dimension": "安全保障",
        "control": "个人信息安全保护措施",
        "standard": "GB/T 35273",
        "sub_controls": [
            {"name": "是否说明技术安全措施", "risk": "高"},
            {"name": "是否说明管理制度与权限控制", "risk": "中"},
            {"name": "是否说明安全事件响应机制", "risk": "中"},
        ],
        "query": "安全保护措施数据安全",
        "intent": "general",
    },
]


class Analyzer:
    def __init__(self, base_dir: Path, experiment: ExperimentConfig):
        self.base_dir = Path(base_dir).resolve()
        self.exp = experiment
        self.ctx = ExperimentContext(experiment, self.base_dir)
        self.policy_dir = self.base_dir / "dataset" / "隐私政策"
        self.retrieve = try_import_retrieve(experiment)
        self.llm = LLMClient(system_prompt="你是一名国家标准驱动的隐私合规审计专家。")

    def load_policy_text(self, path: Path) -> str:
        return safe_read_text(path)

    def retrieve_chunks(self, query: str, intent: str) -> List[Dict]:
        if not self.retrieve or not self.exp.use_control_point_rag:
            return []
        actual_intent = intent if self.exp.use_intent_retrieval else "general"
        try:
            return self.retrieve(query, intent=actual_intent)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 控制点检索失败: {query} -> {exc}")
            return []

    def build_prompt(self, control: Dict, standard_chunks: List[Dict], policy_text: str) -> str:
        std_text = "\n".join(f"- {c.get('text', '').strip()}" for c in standard_chunks if c.get("text", "").strip())
        if not std_text:
            std_text = "- 未检索到明确国家标准片段，请结合控制点要求进行审慎分析，并明确说明证据不足。"
        sub_ctrls = "\n".join(f"{i + 1}. {s['name']}（风险等级：{s['risk']}）" for i, s in enumerate(control["sub_controls"]))
        if self.exp.use_prompt_constraint:
            return f"""
你是一名隐私与数据保护合规审计专家。

【合规控制点】
{control['control']}（{control['standard']}）

【子义务拆解】
{sub_ctrls}

【国家标准依据】
{std_text}

【隐私政策原文】
{policy_text[:6000]}

请逐项判断每个【子义务】是否满足，并输出：
1. 子义务判断结果（满足 / 不满足）
2. 总体合规结论（满足 / 部分满足 / 不满足）
3. 高风险问题总结
4. 整改建议（分别从 文案层 / 产品层 / 技术层 给出）

请使用专业、审计级中文表述。
""".strip()
        return f"""
请根据给定的合规控制点、国家标准依据和隐私政策原文，分析该控制点的合规情况，并给出风险与整改建议。
控制点：{control['control']}（{control['standard']}）
子义务：
{sub_ctrls}
国家标准依据：
{std_text}
隐私政策原文：
{policy_text[:6000]}
""".strip()

    def analyze_policy(self, policy_path: Path) -> Optional[Path]:
        app_name = policy_path.stem
        policy_text = self.load_policy_text(policy_path)
        if not policy_text.strip():
            print(f"[WARN] 隐私政策为空: {policy_path.name}")
            return None
        self.ctx.save_manifest(app_name)
        high_risk_controls = []
        report = [
            "# 隐私政策合规审计报告（消融实验版）",
            f"- 应用名称：{app_name}",
            f"- 文件名称：{policy_path.name}",
            f"- 分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 实验编号：{self.exp.name}",
            f"- 实验说明：{self.exp.description}",
            "- 依据标准：GB/T 35273、GB/T 41391、GB/T 43435、GB/T 45574",
            "",
        ]
        for ctrl in COMPLIANCE_CONTROLS:
            print(f"   · 控制点分析：{ctrl['id']} {ctrl['control']}")
            std_chunks = self.retrieve_chunks(ctrl["query"], ctrl["intent"])
            analysis = self.llm.chat(self.build_prompt(ctrl, std_chunks, policy_text), temperature=0.2)
            report.append(f"## {ctrl['id']} {ctrl['control']}")
            report.append(analysis if analysis else "模型未返回有效分析结果。")
            report.append("")
            if analysis and ("高风险" in analysis or "不满足" in analysis):
                high_risk_controls.append(ctrl)
        report.append("## 总体结论")
        if high_risk_controls:
            report.append(f"本次共识别出 {len(high_risk_controls)} 个存在较高风险的控制点，建议优先进行人工复核与整改。")
            for item in high_risk_controls:
                report.append(f"- {item['id']} {item['control']}")
        else:
            report.append("当前抽取到的控制点分析中未出现显著高风险结论，但仍建议结合真实产品路径与 SDK 清单做交叉核验。")
        if self.exp.use_control_point_rag:
            report.append("\n## 检索摘要")
            for ctrl in COMPLIANCE_CONTROLS:
                chunks = self.retrieve_chunks(ctrl["query"], ctrl["intent"])
                if chunks:
                    report.append(f"### {ctrl['id']} {ctrl['control']}")
                    for c in chunks[:2]:
                        report.append(f"- {shorten(c.get('text', ''), 180)}")
        out = self.ctx.app_output_dir(app_name) / "国标和隐私政策分析结果.md"
        safe_write_text(out, "\n".join(report).strip() + "\n")
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=".")
    ap.add_argument("--experiment", default="baseline")
    ap.add_argument("--app", default="")
    args = ap.parse_args()
    analyzer = Analyzer(Path(args.base_dir), get_experiment(args.experiment))
    if args.app.strip():
        files = [Path(args.base_dir) / "dataset" / "隐私政策" / f"{args.app}.txt"]
    else:
        files = sorted((Path(args.base_dir) / "dataset" / "隐私政策").glob("*.txt"))
    for fp in files:
        print(f"[RUN] 国标和隐私政策分析: {fp.stem}")
        analyzer.analyze_policy(fp)


if __name__ == "__main__":
    main()
