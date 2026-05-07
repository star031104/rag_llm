# -*- coding: utf-8 -*-
"""
国标和权限分析_改进版.py
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from demo import retrieve


BASE_DIR = Path(__file__).resolve().parent
CATEGORY_DIR = BASE_DIR / "dataset" / "分类"
POLICY_DIR = BASE_DIR / "dataset" / "隐私政策"
APK_MAPPING_DIR = BASE_DIR / "dataset" / "apk映射"
SEMANTIC_CACHE_FILE = BASE_DIR / "dataset" / "权限语义缓存" / "semantic_cache.json"
OUTPUT_ROOT_DIR = BASE_DIR / "output"
OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

# 建议优先使用环境变量
SILICONFLOW_API_KEY = "sk-qjwifjustybrcyhbxvkembfapyjbtzsdlnwssphjtsxyetcp"
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
CHAT_URL = os.getenv("CHAT_URL", "https://api.siliconflow.cn/v1/chat/completions")

TIMEOUT = (10, 180)
MAX_RETRY = 6
REQUEST_INTERVAL = 1.2
APP_INTERVAL = 2.0

ALLOWED_STATUSES = {
    "业务核心必要",
    "业务辅助合理",
    "国标未直接枚举但可解释",
    "业务弱相关需审查",
    "高风险异常权限",
}

# 高风险默认审查权限：没有强证据时，不允许轻易落到“业务辅助合理”
RISKY_PERMISSIONS = {
}

# 强先验：类别白名单（可以按你后续实验继续补）
CATEGORY_CORE_RULES = {
}


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def try_json_load(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        return json.loads(safe_read_text(path))
    except Exception:
        return default


def extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {
            "true", "1", "yes", "y", "是", "属于", "39类", "in_39"
        }
    return False


def shorten(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


def extract_app_records_from_json_data(data: Any) -> List[Dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ["results", "apps", "data", "items", "app_list"]:
            if key in data and isinstance(data[key], list):
                return [x for x in data[key] if isinstance(x, dict)]
        return [data]
    return []


def normalize_app_record(record: Dict, fallback_name: str = "") -> Dict[str, Any]:
    app_name = (
        record.get("app_name")
        or record.get("app")
        or record.get("name")
        or record.get("appName")
        or record.get("应用名称")
        or fallback_name
    )
    is_in_39 = record.get(
        "is_in_39",
        record.get("in_39", record.get("是否属于39类", record.get("属于39类", False)))
    )
    category = (
        record.get("gbt_39_category")
        or record.get("category")
        or record.get("app_category")
        or record.get("39_category")
        or record.get("类别")
        or record.get("39类类别")
        or "未分类"
    )
    return {
        "app_name": str(app_name).strip(),
        "is_in_39": normalize_bool(is_in_39),
        "gbt_39_category": str(category).strip(),
    }


def load_all_category_records() -> List[Dict[str, Any]]:
    files = sorted(CATEGORY_DIR.glob("*.json"))
    if not files:
        raise RuntimeError(f"未找到分类文件：{CATEGORY_DIR}")

    merged: Dict[str, Dict[str, Any]] = {}
    for jf in files:
        try:
            data = json.loads(safe_read_text(jf))
        except Exception as e:
            print(f"[WARN] 跳过无效分类文件 {jf.name}: {e}")
            continue

        for raw in extract_app_records_from_json_data(data):
            rec = normalize_app_record(raw, fallback_name=jf.stem)
            if rec["app_name"]:
                merged[rec["app_name"]] = rec

    return sorted(merged.values(), key=lambda x: x["app_name"])


def find_permission_file(app_name: str) -> Optional[Path]:
    p1 = APK_MAPPING_DIR / f"{app_name}.txt"
    if p1.exists():
        return p1
    p2 = APK_MAPPING_DIR / f"{app_name}.apk.txt"
    if p2.exists():
        return p2
    return None


def parse_permission_line(line: str) -> Dict[str, str]:
    raw = line.strip()
    if not raw:
        return {}

    if "\t" in raw:
        a, b = raw.split("\t", 1)
        return {"raw_line": raw, "permission_name": a.strip(), "permission_desc_en": b.strip()}

    m = re.match(r"^([A-Za-z0-9_\.]+)\s+(.*)$", raw)
    if m and (m.group(1).startswith("android.permission.") or ".permission." in m.group(1)):
        return {
            "raw_line": raw,
            "permission_name": m.group(1).strip(),
            "permission_desc_en": m.group(2).strip(),
        }

    return {"raw_line": raw, "permission_name": raw, "permission_desc_en": ""}


def load_permission_records(app_name: str) -> List[Dict[str, str]]:
    fp = find_permission_file(app_name)
    if fp is None:
        return []

    records = []
    seen = set()
    for line in safe_read_text(fp).splitlines():
        line = line.strip()
        if not line:
            continue
        item = parse_permission_line(line)
        if not item:
            continue
        name = item["permission_name"]
        if name in seen:
            continue
        seen.add(name)
        records.append(item)

    return records


def load_semantic_cache() -> Dict[str, Dict[str, Any]]:
    cache = try_json_load(SEMANTIC_CACHE_FILE, default={})
    return cache if isinstance(cache, dict) else {}


def save_semantic_cache(cache: Dict[str, Dict[str, Any]]):
    SEMANTIC_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEMANTIC_CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def build_permission_semantic_prompt(permission_name: str, permission_desc_en: str) -> str:
    return (
        "你是一名 Android 权限语义解析助手。请根据以下 APK 权限记录，输出结构化中文理解。"
        "你只做语义解释，不做合规判断。输出 JSON，字段：permission_name、permission_cn、"
        "data_type、capability、usage_hint、uncertainty。"
        f"输入：权限名：{permission_name}；英文解释：{permission_desc_en if permission_desc_en else '（无）'}"
    )


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    retry = 0
    while retry < MAX_RETRY:
        try:
            r = requests.post(CHAT_URL, headers=headers, json=payload, timeout=TIMEOUT)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                time.sleep(REQUEST_INTERVAL)
                return (content or "").strip()

            if r.status_code == 429:
                wait = 6 * (retry + 1)
                print(f"[WARN] 429 限流，等待 {wait}s")
                time.sleep(wait)
                retry += 1
                continue

            print(f"[WARN] API错误 {r.status_code}: {r.text[:300]}")
            retry += 1
            time.sleep(4)

        except Exception as e:
            retry += 1
            print(f"[WARN] LLM调用异常，第{retry}/{MAX_RETRY}次：{e}")
            time.sleep(4 * retry)

    print("[WARN] LLM 调用失败，返回空字符串")
    return ""


def normalize_permission_semantic(
    permission_name: str,
    permission_desc_en: str,
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    key = f"{permission_name}|||{permission_desc_en}"
    if key in cache:
        return cache[key]

    raw = call_llm(
        "你是一名严谨的 Android 权限语义解析助手，只负责语义解释。",
        build_permission_semantic_prompt(permission_name, permission_desc_en),
        temperature=0.05,
    )
    obj = extract_json_object(raw)

    result = {
        "permission_name": permission_name,
        "permission_cn": obj.get("permission_cn", permission_name),
        "data_type": obj.get("data_type", "不明确"),
        "capability": obj.get("capability", "不明确"),
        "usage_hint": obj.get("usage_hint", "不明确"),
        "uncertainty": obj.get("uncertainty", "不明确"),
    }

    cache[key] = result
    save_semantic_cache(cache)
    return result


def enrich_permissions_with_semantics(
    permission_records: List[Dict[str, str]],
    cache: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    enriched = []
    for item in permission_records:
        semantic = normalize_permission_semantic(
            item["permission_name"],
            item.get("permission_desc_en", ""),
            cache,
        )
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


def retrieve_texts(queries: List[str], retrieve_n: int = 50, topk: int = 8) -> List[str]:
    merged, seen = [], set()
    for q in queries:
        if not q.strip():
            continue
        try:
            results = retrieve(query=q, intent="general", retrieve_n=retrieve_n, topk=topk)
        except TypeError:
            results = retrieve(q)
        except Exception as e:
            print(f"[WARN] RAG检索失败: {q} -> {e}")
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


def retrieve_rules_for_app(category: str, is_in_39: bool) -> Dict[str, List[str]]:
    general_rules = retrieve_texts(
        [
            "移动应用 权限申请 最小必要 原则 合理相关",
            "移动互联网应用程序 个人信息 最小必要 权限 申请",
            "GB/T 41391 权限申请 最小必要 合理相关",
        ],
        retrieve_n=40,
        topk=8,
    )

    category_rules, typical_features = [], []
    if is_in_39 and category and category != "未分类":
        category_rules = retrieve_texts(
            [
                f"{category} GB/T 41391 基本业务功能 必要个人信息 表",
                f"{category} 权限申请 最小必要 合理相关",
                f"{category} 典型业务功能 图片 视频 相机 存储 网络",
            ],
            retrieve_n=60,
            topk=10,
        )
        typical_features = retrieve_texts(
            [
                f"{category} 应用 常见功能 场景 图片 视频 录音 定位 上传 下载",
                f"{category} app 常见业务功能 功能场景",
            ],
            retrieve_n=40,
            topk=6,
        )

    return {
        "general_rules": general_rules,
        "category_rules": category_rules,
        "typical_features": typical_features,
    }


COMMON_KEYWORDS = {
    "CAMERA": ["相机", "摄像头", "拍照", "拍摄", "扫码", "扫描", "图片拍摄", "录像"],
    "RECORD_AUDIO": ["麦克风", "录音", "语音", "音频", "说话"],
    "LOCATION": ["定位", "位置", "地理位置", "经纬度", "gps", "附近"],
    "STORAGE": ["存储", "文件", "相册", "图片", "视频", "照片", "上传", "下载", "缓存", "本地"],
    "NETWORK": ["网络", "联网", "Wi-Fi", "wifi", "移动网络", "IP地址", "连接状态"],
    "PHONE_STATE": ["手机状态", "设备标识", "IMEI", "IMSI", "OAID", "Android ID", "电话状态"],
    "APP_LIST": ["应用列表", "已安装应用", "应用信息", "包信息", "默认浏览器", "launcher", "角标"],
    "NOTIFICATION": ["通知", "推送", "角标", "消息提醒", "红点"],
    "OVERLAY": ["悬浮窗", "悬浮球", "浮窗", "overlay"],
    "INSTALL": ["安装", "更新安装包", "下载后安装", "请求安装"],
}


def infer_keywords_for_permission(p: Dict[str, Any]) -> List[str]:
    text = " ".join([
        p.get("permission_name", ""),
        p.get("permission_cn", ""),
        p.get("data_type", ""),
        p.get("capability", ""),
        p.get("usage_hint", ""),
    ]).lower()

    kws: List[str] = []

    def add(group: str):
        for x in COMMON_KEYWORDS[group]:
            if x not in kws:
                kws.append(x)

    if any(x in text for x in ["camera", "相机", "拍照", "图像"]):
        add("CAMERA")
    if any(x in text for x in ["audio", "record", "录音", "音频", "麦克风"]):
        add("RECORD_AUDIO")
    if any(x in text for x in ["location", "位置", "定位", "gps"]):
        add("LOCATION")
    if any(x in text for x in ["storage", "media", "存储", "文件", "图片", "视频", "相册"]):
        add("STORAGE")
    if any(x in text for x in ["network", "internet", "wifi", "网络", "联网"]):
        add("NETWORK")
    if any(x in text for x in ["phone", "imei", "imsi", "oaid", "设备状态", "电话状态", "android id"]):
        add("PHONE_STATE")
    if any(x in text for x in ["package", "tasks", "app list", "应用列表", "已安装应用", "包"]):
        add("APP_LIST")
    if any(x in text for x in ["notification", "badge", "通知", "角标", "推送"]):
        add("NOTIFICATION")
    if any(x in text for x in ["alert_window", "overlay", "悬浮窗", "悬浮球"]):
        add("OVERLAY")
    if any(x in text for x in ["install", "安装"]):
        add("INSTALL")

    perm = p.get("permission_name", "").upper()
    if "CAMERA" in perm:
        add("CAMERA")
    if "AUDIO" in perm or "RECORD" in perm or "MIC" in perm:
        add("RECORD_AUDIO")
    if "LOCATION" in perm or "GPS" in perm:
        add("LOCATION")
    if any(x in perm for x in ["STORAGE", "MEDIA", "FILESYSTEM", "EXTERNAL"]):
        add("STORAGE")
    if any(x in perm for x in ["NETWORK", "INTERNET", "WIFI"]):
        add("NETWORK")
    if any(x in perm for x in ["PHONE", "IMEI", "IMSI"]):
        add("PHONE_STATE")
    if any(x in perm for x in ["PACKAGE", "TASK", "LAUNCHER", "BADGE"]):
        add("APP_LIST")
    if any(x in perm for x in ["NOTIFICATION", "BADGE"]):
        add("NOTIFICATION")
    if any(x in perm for x in ["ALERT_WINDOW", "OVERLAY"]):
        add("OVERLAY")
    if "INSTALL" in perm:
        add("INSTALL")

    if not kws:
        raw_terms = re.split(
            r"[^A-Za-z0-9\u4e00-\u9fff]+",
            p.get("permission_cn", "") + " " + p.get("capability", "")
        )
        kws = [x for x in raw_terms if len(x) >= 2][:6]

    return kws[:10]


def split_policy_sentences(policy_text: str) -> List[str]:
    text = re.sub(r"[ \t]+", " ", policy_text)
    pieces = re.split(r"(?<=[。！？；\n])", text)
    return [p.strip() for p in pieces if p.strip()]


def collect_policy_evidence(permission_item: Dict[str, Any], policy_text: str, max_items: int = 8) -> List[str]:
    if not policy_text.strip():
        return []

    keywords = infer_keywords_for_permission(permission_item)
    scored: List[Tuple[int, str]] = []

    for s in split_policy_sentences(policy_text):
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


def build_permission_analysis_prompt(
    app_name: str,
    category: str,
    is_in_39: bool,
    permission_item: Dict[str, Any],
    rules_bundle: Dict[str, List[str]],
    policy_evidence: List[str],
) -> str:
    rule_text = "\n".join(
        f"- {x}" for x in (rules_bundle.get("general_rules", []) + rules_bundle.get("category_rules", []))
    ) or "- 未检索到相关国标条款"

    feature_text = "\n".join(
        f"- {x}" for x in rules_bundle.get("typical_features", [])
    ) or "- 未检索到该类别典型功能描述"

    evidence_text = (
        "\n".join(f"- {x}" for x in policy_evidence)
        if policy_evidence else "- 未从隐私政策中召回到直接证据"
    )

    category_line = category if is_in_39 else "非39类/未分类"

    return f"""你是一名移动应用“国家标准 + 业务功能 + 权限必要性”联合分析助手。
请对单个权限进行审慎分析，目标不是尽量为权限寻找合理化解释，而是严格区分：
业务核心必要、业务辅助合理、国标未直接枚举但可解释、业务弱相关需审查、高风险异常权限。

【应用信息】
应用名称：{app_name}
是否39类：{is_in_39}
类别：{category_line}

【当前权限】
权限名称：{permission_item['permission_name']}
中文解释：{permission_item['permission_cn']}
数据类型：{permission_item['data_type']}
核心能力：{permission_item['capability']}
常见场景：{permission_item['usage_hint']}
语义不确定性：{permission_item['uncertainty']}

【国家标准/原则证据】
{rule_text}

【该类别典型功能提示】
{feature_text}

【隐私政策/业务证据】
{evidence_text}

判断要求：
1. 优先依据国家标准通用原则和类别基本业务功能，不得仅凭“可能存在的业务场景”就放宽结论。
2. “业务核心必要”必须满足：权限直接支撑该类 App 的基本业务功能，且缺少该权限会明显影响核心功能实现。
3. “业务辅助合理”必须满足：权限对明确的扩展功能/增强功能/安全保障功能有直接支撑，且当前业务证据较明确。
4. “国标未直接枚举但可解释”仅用于：国标未直接点名，但类别典型功能与权限能力高度一致，且不存在明显越权风险。
5. 若国标未直接支持，且仅能从一般体验优化、个性化、性能增强、模糊场景解释，则优先考虑“业务弱相关需审查”，而不是“业务辅助合理”。
6. 对安装管理、系统设置、系统悬浮窗、应用列表、后台控制、跨应用、系统级敏感权限，应提高审查强度；若缺少明确依据，应优先判为“业务弱相关需审查”或“高风险异常权限”。
7. “高风险异常权限”用于：权限与核心业务明显不匹配，且越权风险较高，或当前证据下无法建立合理业务链条。
8. 不能使用“违法/违规/不合规”等绝对化措辞。
9. 输出必须是 JSON，不要输出额外文字。

输出字段：
status、rule_basis、reason、confidence、evidence、risk_note。
"""


def calibrate_status(
    category: str,
    permission_name: str,
    raw_status: str,
    rule_basis: str,
    reason: str,
    policy_evidence: List[str],
) -> str:
    """
    基于规则对 LLM 输出做保守校准
    """
    perm = permission_name.strip()
    reason_text = f"{reason or ''} {rule_basis or ''}"
    has_policy = bool(policy_evidence)

    strong_rule_markers = ["A.", "附录", "基本业务功能", "必要个人信息", "服务类型", "地图导航类", "浏览器类"]
    has_direct_rule = any(x in reason_text for x in strong_rule_markers)

    cat_rules = CATEGORY_CORE_RULES.get(category, {})

    # 1) 风险权限默认偏严
    if perm in RISKY_PERMISSIONS:
        if has_direct_rule and has_policy:
            if raw_status == "业务核心必要":
                return "国标未直接枚举但可解释"
            if raw_status == "业务辅助合理":
                return "国标未直接枚举但可解释"
            return raw_status
        else:
            if raw_status in {"业务核心必要", "业务辅助合理", "国标未直接枚举但可解释"}:
                return "业务弱相关需审查"
            return raw_status

    # 2) 核心白名单提升
    if perm in cat_rules.get("core", set()):
        if raw_status in {"业务辅助合理", "国标未直接枚举但可解释", "业务弱相关需审查"}:
            return "业务核心必要"

    # 3) 辅助白名单限制，不允许过度拔高到核心必要
    if perm in cat_rules.get("aux", set()):
        if raw_status == "业务核心必要":
            return "业务辅助合理"

    # 4) 完全没有政策证据，也没有明确国标支撑，不允许给太高
    if not has_policy and not has_direct_rule:
        if raw_status == "业务核心必要":
            return "业务弱相关需审查"
        if raw_status == "业务辅助合理":
            return "国标未直接枚举但可解释"

    return raw_status


def analyze_single_permission(
    app_name: str,
    category: str,
    is_in_39: bool,
    permission_item: Dict[str, Any],
    rules_bundle: Dict[str, List[str]],
    policy_text: str,
) -> Dict[str, Any]:
    policy_evidence = collect_policy_evidence(permission_item, policy_text)

    raw = call_llm(
        "你是一名严谨的移动应用国家标准与权限必要性分析助手，只输出合法 JSON。",
        build_permission_analysis_prompt(
            app_name, category, is_in_39, permission_item, rules_bundle, policy_evidence
        ),
        temperature=0.05,
    )

    obj = extract_json_object(raw)
    status = str(obj.get("status", "")).strip()

    # 更保守的兜底逻辑：默认往严，不往松
    if status not in ALLOWED_STATUSES:
        status = "业务弱相关需审查"

    confidence = str(obj.get("confidence", "中")).strip()
    if confidence not in {"高", "中", "低"}:
        confidence = "中"

    evidence = obj.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    evidence = [shorten(x, 220) for x in evidence if str(x).strip()][:3]

    if not evidence and policy_evidence:
        evidence = policy_evidence[:3]

    rule_basis = shorten(obj.get("rule_basis", "未检索到足够直接的国标条款依据"), 80)
    reason = shorten(obj.get("reason", "未提供有效说明。"), 180)
    risk_note = shorten(obj.get("risk_note", "无"), 120)

    status = calibrate_status(
        category=category,
        permission_name=permission_item["permission_name"],
        raw_status=status,
        rule_basis=rule_basis,
        reason=reason,
        policy_evidence=policy_evidence,
    )

    return {
        "permission_name": permission_item["permission_name"],
        "permission_cn": permission_item.get("permission_cn", permission_item["permission_name"]),
        "rule_basis": rule_basis,
        "status": status,
        "reason": reason,
        "confidence": confidence,
        "evidence": evidence,
        "risk_note": risk_note,
        "policy_evidence": policy_evidence,
    }


def summarize_status_counts(results: List[Dict[str, Any]]) -> Dict[str, int]:
    return {s: sum(1 for x in results if x["status"] == s) for s in ALLOWED_STATUSES}


def pick_key_risks(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    risky_keywords = [
        "RESTART_PACKAGES",
        "REQUEST_INSTALL_PACKAGES",
        "WRITE_SETTINGS",
        "SYSTEM_ALERT_WINDOW",
        "QUERY_ALL_PACKAGES",
        "KILL_BACKGROUND_PROCESSES",
        "DISABLE_KEYGUARD",
        "SET_WALLPAPER",
    ]

    def risk_boost(item: Dict[str, Any]) -> int:
        perm = item["permission_name"].upper()
        return 0 if any(k in perm for k in risky_keywords) else 1

    priority = {
        "高风险异常权限": 0,
        "业务弱相关需审查": 1,
        "国标未直接枚举但可解释": 2,
        "业务辅助合理": 3,
        "业务核心必要": 4,
    }
    conf = {"高": 0, "中": 1, "低": 2}

    return sorted(
        results,
        key=lambda x: (
            risk_boost(x),
            priority.get(x["status"], 9),
            conf.get(x["confidence"], 9),
        )
    )[:8]


def generate_md(
    app_record: Dict[str, Any],
    semantic_permissions: List[Dict[str, Any]],
    rules_bundle: Dict[str, List[str]],
    results: List[Dict[str, Any]],
) -> str:
    app_name = app_record["app_name"]
    is_in_39 = app_record["is_in_39"]
    category = app_record["gbt_39_category"]
    counts = summarize_status_counts(results)

    lines = []
    lines.append(f"# {app_name} 国标和权限分析结果（改进版）\n")
    lines.append("## 一、基本信息\n")
    lines.append(f"- 应用名称：{app_name}")
    lines.append(f"- 是否属于39类：{is_in_39}")
    lines.append(f"- 39类类别：{category}")
    lines.append(f"- 分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 权限数量：{len(semantic_permissions)}")
    lines.append("- 分析模式：国标原则 + 类别功能 + 隐私政策证据 + 权限语义联合分析（改进版）\n")

    lines.append("## 二、分析方法说明\n")
    lines.append("本版本在原始分析基础上加入了更严格的风险权限校准逻辑，不再仅依据“可解释性”放宽权限标签。")
    lines.append("分析时综合四类证据：1. 国家标准通用原则；2. 39类类别条款与典型业务功能（若属于39类）；3. 隐私政策中的直接业务证据；4. APK 映射得到的权限语义说明。")
    lines.append("输出状态含义：业务核心必要 / 业务辅助合理 / 国标未直接枚举但可解释 / 业务弱相关需审查 / 高风险异常权限。\n")

    lines.append("## 三、国标检索摘要\n")
    if rules_bundle.get("general_rules"):
        lines.append("### 3.1 通用原则")
        for x in rules_bundle["general_rules"][:4]:
            lines.append(f"- {shorten(x, 220)}")
        lines.append("")
    if is_in_39 and rules_bundle.get("category_rules"):
        lines.append("### 3.2 类别相关条款")
        for x in rules_bundle["category_rules"][:5]:
            lines.append(f"- {shorten(x, 220)}")
        lines.append("")
    if is_in_39 and rules_bundle.get("typical_features"):
        lines.append("### 3.3 类别典型功能提示")
        for x in rules_bundle["typical_features"][:4]:
            lines.append(f"- {shorten(x, 220)}")
        lines.append("")

    lines.append("## 四、分析结果\n")
    lines.append("### 4.1 总体判断")
    if is_in_39:
        lines.append(f"{app_name} 被归入 **{category}**。本次分析不再仅依据类别标签进行机械白名单判断，而是联合业务功能证据并引入风险权限校准机制评估权限必要性。")
    else:
        lines.append(f"{app_name} 未归入 39 类，因此本次分析以通用国家标准原则和隐私政策证据为主。")

    lines.append(
        f"本次共分析 {len(results)} 个权限，其中："
        f"业务核心必要 {counts['业务核心必要']} 个，"
        f"业务辅助合理 {counts['业务辅助合理']} 个，"
        f"国标未直接枚举但可解释 {counts['国标未直接枚举但可解释']} 个，"
        f"业务弱相关需审查 {counts['业务弱相关需审查']} 个，"
        f"高风险异常权限 {counts['高风险异常权限']} 个。\n"
    )

    lines.append("### 4.2 权限匹配对比分析")
    lines.append("| 权限名称 | 中文解释 | 国标条款或依据 | 判断状态 | 说明 | 置信度 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in results:
        lines.append(
            f"| {r['permission_name']} | {r['permission_cn']} | {r['rule_basis']} | {r['status']} | {r['reason']} | {r['confidence']} |"
        )
    lines.append("")

    lines.append("### 4.3 重点权限审查")
    for item in pick_key_risks(results):
        lines.append(f"- **{item['permission_name']}（{item['status']}）**：{item['reason']} 风险提示：{item['risk_note']}。")
        if item["evidence"]:
            for ev in item["evidence"][:2]:
                lines.append(f"  - 证据：{ev}")
    lines.append("")

    lines.append("### 4.4 总结性结论")
    if counts["高风险异常权限"] > 0:
        lines.append("报告显示存在高风险异常权限，建议优先开展人工复核，重点核对真实业务功能、界面路径、SDK 来源及实际调用链。")
    elif counts["业务弱相关需审查"] > 0:
        lines.append("报告显示部分权限与核心业务的关联度偏弱，建议结合产品功能说明、隐私政策和真实代码调用情况做二次核查。")
    else:
        lines.append("报告显示多数权限可通过国家标准原则、类别功能或隐私政策证据获得解释，但仍需关注‘可解释性’与‘最小必要’之间的边界。")
    lines.append("需要注意的是，“国标未直接枚举但可解释”并不等于绝对安全，只表示其并非标准逐项点名的权限，但在当前业务证据下仍具可解释性。")

    return "\n".join(lines).strip() + "\n"


def analyze_app(app_record: Dict[str, Any], cache: Dict[str, Dict[str, Any]]):
    app_name = app_record["app_name"]

    print("=" * 80)
    print(f"开始分析：{app_name}")
    print("=" * 80)

    permission_records = load_permission_records(app_name)
    if not permission_records:
        print(f"[WARN] 未读取到 APK 映射权限：{app_name}")
        return

    print(f"[INFO] 共读取权限 {len(permission_records)} 个")

    semantic_permissions = enrich_permissions_with_semantics(permission_records, cache)

    policy_text = safe_read_text(POLICY_DIR / f"{app_name}.txt")
    if policy_text:
        print(f"[INFO] 已读取隐私政策，长度 {len(policy_text)} 字符")
    else:
        print("[WARN] 未读取到隐私政策，将更多依赖国标和功能提示")

    rules_bundle = retrieve_rules_for_app(
        app_record["gbt_39_category"],
        app_record["is_in_39"]
    )
    print(
        f"[INFO] 通用原则 {len(rules_bundle['general_rules'])} 条，"
        f"类别条款 {len(rules_bundle['category_rules'])} 条，"
        f"典型功能提示 {len(rules_bundle['typical_features'])} 条"
    )

    results = []
    for idx, p in enumerate(semantic_permissions, start=1):
        res = analyze_single_permission(
            app_name,
            app_record["gbt_39_category"],
            app_record["is_in_39"],
            p,
            rules_bundle,
            policy_text,
        )
        results.append(res)
        print(f"[INFO] {idx}/{len(semantic_permissions)} {res['permission_name']} -> {res['status']} ({res['confidence']})")

    md = generate_md(app_record, semantic_permissions, rules_bundle, results)
    out_path = OUTPUT_ROOT_DIR / app_name / "国标和权限分析结果.md"
    safe_write_text(out_path, md)

    print(f"[DONE] 报告已输出：{out_path}")
    time.sleep(APP_INTERVAL)


def main():
    app_records = load_all_category_records()
    cache = load_semantic_cache()

    print(f"[INFO] 共读取应用 {len(app_records)} 个")

    for rec in app_records:
        try:
            analyze_app(rec, cache)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[ERROR] 分析 {rec.get('app_name', '未知应用')} 失败：{e}")
            continue


if __name__ == "__main__":
    main()