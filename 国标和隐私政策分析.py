# -*- coding: utf-8 -*-
"""
国标和隐私政策分析.py

功能：
1. 读取代码同级目录下 dataset/隐私政策 中的所有 .txt 文件
2. 对每个隐私政策文件分别进行一次国标驱动的合规分析
3. 分析结果输出到代码同级目录下 output/应用名/国标和隐私政策分析结果.md

例如：
输入：dataset/隐私政策/高德地图.txt
输出：output/高德地图/国标和隐私政策分析结果.md
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict
import os
import time
import requests

# ===================== 路径配置 =====================
BASE_DIR = Path(__file__).resolve().parent

# 输入目录：代码同级目录/dataset/隐私政策
POLICY_DIR = BASE_DIR / "dataset" / "隐私政策"

# 输出根目录：代码同级目录/output
OUTPUT_ROOT_DIR = BASE_DIR / "output"
OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== LLM 配置 =====================
API_KEY = "sk-owpvsumadfmgklabdqoyyesnagsjzofcbrhfgowzoxuuifth"
CHAT_URL = os.getenv("CHAT_URL", "https://api.siliconflow.cn/v1/chat/completions")
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")

TIMEOUT = (10, 120)
MAX_RETRY = 6
SLEEP = 1.5
APP_SLEEP = 3.0

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ===================== 合规控制点 =====================
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

# ===================== RAG 检索 =====================
from demo import retrieve

# ===================== 工具函数 =====================
def load_policy_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def call_llm(prompt: str) -> str:
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "你是一名国家标准驱动的隐私合规审计专家。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    retry = 0
    while retry < MAX_RETRY:
        try:
            r = requests.post(
                CHAT_URL,
                headers=HEADERS,
                json=payload,
                timeout=TIMEOUT
            )

            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                if content and content.strip():
                    time.sleep(SLEEP)
                    return content.strip()

            if r.status_code == 429:
                wait_time = 10 * (retry + 1)
                print(f"⚠️ API 限流 (429)，等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
                retry += 1
                continue

            print(f"⚠️ API 错误: {r.status_code}")
            time.sleep(5)
            retry += 1

        except requests.exceptions.ReadTimeout:
            print("⚠️ 请求超时，重试中...")
            time.sleep(5)
            retry += 1

        except Exception as e:
            print(f"⚠️ 调用异常: {e}")
            time.sleep(5)
            retry += 1

    print("❌ LLM 请求失败，已达到最大重试次数")
    return ""


def build_prompt(control: Dict, standard_chunks: List[Dict], policy_text: str) -> str:
    std_text = "\n".join(
        [f"- {c.get('text', '').strip()}" for c in standard_chunks if c.get("text", "").strip()]
    )

    if not std_text:
        std_text = "- 未检索到明确国家标准片段，请结合控制点要求进行审慎分析，并明确说明证据不足。"

    sub_ctrls = "\n".join(
        [f"{i + 1}. {s['name']}（风险等级：{s['risk']}）" for i, s in enumerate(control["sub_controls"])]
    )

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


def analyze_policy(policy_path: Path):
    app_name = policy_path.stem
    policy_text = load_policy_text(policy_path)

    high_risk_items = []

    report = []
    report.append("# 隐私政策合规审计报告（最终版）\n")
    report.append(f"- 应用名称：{app_name}")
    report.append(f"- 文件名称：{policy_path.name}")
    report.append(f"- 分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("- 依据标准：GB/T 35273、GB/T 41391、GB/T 43435、GB/T 45574\n")

    for ctrl in COMPLIANCE_CONTROLS:
        print(f"   · 控制点分析：{ctrl['id']} {ctrl['control']}")
        std_chunks = retrieve(ctrl["query"], intent=ctrl["intent"])
        prompt = build_prompt(ctrl, std_chunks, policy_text)
        analysis = call_llm(prompt)

        report.append(f"## {ctrl['id']} {ctrl['control']}")
        report.append(analysis if analysis else "模型未返回有效分析结果。")
        report.append("")

        if analysis and ("高风险" in analysis or "不满足" in analysis):
            high_risk_items.append(ctrl["control"])

        time.sleep(SLEEP)

    report.append("## 总体合规性总结")
    report.append(f"- 覆盖合规控制点数量：{len(COMPLIANCE_CONTROLS)}")
    report.append(f"- 高风险控制点数量：{len(high_risk_items)}")
    report.append("")

    report.append("## 高风险问题清单")
    if high_risk_items:
        for item in high_risk_items:
            report.append(f"- {item}")
    else:
        report.append("- 本次分析中未识别出明确高风险控制点，但仍建议结合业务实际开展人工复核。")
    report.append("")

    report.append("## 整改优先级建议（Roadmap）")
    report.append("- **P0（立即整改）**：敏感个人信息单独同意、第三方SDK合规披露")
    report.append("- **P1（短期整改）**：个人信息收集必要性说明、适用范围边界明确")
    report.append("- **P2（持续优化）**：安全措施透明化、合规证明披露")

    app_output_dir = OUTPUT_ROOT_DIR / app_name
    app_output_dir.mkdir(parents=True, exist_ok=True)

    out_path = app_output_dir / "国标和隐私政策分析结果.md"
    out_path.write_text("\n".join(report), encoding="utf-8")

    print(f"✅ 已生成报告：{out_path}")


if __name__ == "__main__":
    if not POLICY_DIR.exists():
        raise RuntimeError(f"未找到隐私政策目录：{POLICY_DIR}")

    files = sorted(POLICY_DIR.glob("*.txt"))
    if not files:
        raise RuntimeError(f"目录下未找到 txt 文件：{POLICY_DIR}")

    print(f"📂 隐私政策目录：{POLICY_DIR}")
    print(f"📄 共检测到 {len(files)} 个隐私政策文件，开始逐个分析...\n")

    for f in files:
        print(f"🔍 正在分析：{f.name}")
        analyze_policy(f)
        time.sleep(APP_SLEEP)

    print("\n🎯 所有隐私政策国标合规审计完成")