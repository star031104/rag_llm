# -*- coding: utf-8 -*-
"""
隐私政策和 APK 权限一致性分析（最终完整代码）

设计原则：
1. 只输出 Markdown 报告，不输出 JSON 文件。
2. 使用 APK 原始权限文件 + apk 映射文件。
3. apk映射中存在解释时直接使用；
   apk映射中不存在、但属于 android.permission.* 的系统权限时，基于权限名做系统语义推断；
   apk映射中不存在、且不是标准系统权限的第三方/厂商权限时，基于权限名做弱语义推断，并降低置信度。
4. 采用“规则召回 + LLM 抽取 + 原文直接匹配 + LLM 判定”的混合流程。
5. 能力块抽取兼容 dict / list / fenced json，多种结构都不再炸。
6. 即使能力块抽取失败，也会直接从隐私政策原文召回证据，避免全量塌成“未告知”。
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


# =========================
# 基础配置
# =========================

SILICONFLOW_API_KEY = "sk-owpvsumadfmgklabdqoyyesnagsjzofcbrhfgowzoxuuifth"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CHAT_URL = "https://api.siliconflow.cn/v1/chat/completions"

BASE_DIR = Path(".")
POLICY_DIR = BASE_DIR / "dataset" / "隐私政策"
RAW_PERMISSION_DIR = BASE_DIR / "dataset" / "apk权限"
MAPPED_PERMISSION_DIR = BASE_DIR / "dataset" / "apk映射"
OUTPUT_DIR = BASE_DIR / "output"

WINDOW_SIZE = 1600
WINDOW_STEP = 1000
REQUEST_TIMEOUT = 180
MAX_RETRIES = 5
RETRY_SLEEP = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 能力类别与关键词
# =========================

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "相机": ["相机", "摄像头", "拍照", "拍摄", "扫码", "扫描", "证件照", "人脸", "拍照搜索"],
    "麦克风": ["麦克风", "录音", "语音", "音频", "语音搜索", "语音输入"],
    "位置": ["位置", "定位", "地理位置", "经纬度", "gps", "附近", "ssid", "wifi mac", "cell id"],
    "存储/相册": ["存储", "本地文件", "文件", "相册", "照片", "图片", "视频", "媒体", "下载", "导出", "上传", "截图图片"],
    "联系人": ["联系人", "通讯录", "address book"],
    "日历": ["日历", "calendar", "提醒事项"],
    "应用列表/应用信息": ["应用列表", "已安装应用", "应用信息", "包信息", "launcher", "默认浏览器", "桌面图标"],
    "通知/角标": ["通知", "消息提醒", "推送", "角标", "badge", "红点"],
    "悬浮窗": ["悬浮窗", "悬浮球", "浮窗", "overlay"],
    "截屏": ["截屏", "截图", "录屏", "screen capture", "media projection"],
    "网络状态": ["网络", "wifi", "联网", "热点", "连接状态", "本地网络", "ip地址", "移动网络"],
    "后台运行/自启动": ["后台", "自启动", "保活", "开机", "启动", "前台服务", "系统广播", "小组件"],
    "设备信息/标识": ["设备信息", "设备标识", "imei", "imsi", "oaid", "android id", "设备型号", "系统版本", "设备厂商"],
    "传感器": ["传感器", "陀螺仪", "加速度", "重力", "方向传感器", "光线传感器", "设备运动"],
    "剪贴板": ["剪贴板", "复制", "粘贴", "剪切板"],
    "电话/账号": ["手机号", "电话", "通话", "账号", "登录", "认证", "实名", "身份"],
    "短信": ["短信", "彩信", "message"],
    "蓝牙/NFC/附近设备": ["蓝牙", "bluetooth", "nfc", "附近设备", "nearby"],
    "安装/卸载应用": ["安装", "卸载", "安装包", "更新应用", "request install packages"],
    "其他": [],
}

SENSITIVE_CATEGORIES = {
    "相机", "麦克风", "位置", "存储/相册", "联系人", "日历",
    "应用列表/应用信息", "通知/角标", "悬浮窗", "截屏",
}

GENERAL_SYSTEM_CATEGORIES = {
    "网络状态", "后台运行/自启动", "设备信息/标识", "传感器",
    "剪贴板", "电话/账号", "短信", "蓝牙/NFC/附近设备", "安装/卸载应用",
}

ALLOWED_CAPABILITY_TYPES = set(CATEGORY_KEYWORDS.keys())


# =========================
# 工具函数
# =========================


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())



def safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return ""



def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)



def dedup_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result



def split_sentences(text: str) -> List[str]:
    text = (text or "").replace("\r", "\n")
    parts = re.split(r"(?<=[。！？；\n])|(?<=[.!?;])\s+", text)
    return [normalize_space(p) for p in parts if normalize_space(p)]



def sliding_windows(text: str, window_size: int = WINDOW_SIZE, step: int = WINDOW_STEP) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= window_size:
        return [text]
    windows: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + window_size, n)
        chunk = text[start:end].strip()
        if chunk:
            windows.append(chunk)
        if end >= n:
            break
        start += step
    return windows



def short_text(text: str, max_len: int = 180) -> str:
    text = normalize_space(text)
    return text if len(text) <= max_len else text[:max_len] + "..."



def ensure_list_of_dicts(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []



def extract_json_payload(raw: str) -> Any:
    """尽量从 LLM 输出中解析 JSON，返回 dict/list/None。"""
    if not raw:
        return None
    text = raw.strip()

    candidates = [text]

    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    candidates.extend(fenced)

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        first = text.find(open_ch)
        last = text.rfind(close_ch)
        if first != -1 and last != -1 and last > first:
            candidates.append(text[first:last + 1])

    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None



def normalize_evidence_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return dedup_keep_order(normalize_space(str(x)) for x in value if normalize_space(str(x)))
    if isinstance(value, str):
        s = normalize_space(value)
        return [s] if s else []
    return []


# =========================
# LLM 调用
# =========================


def call_llm(prompt: str, temperature: float = 0.1) -> str:
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一名严谨的移动应用隐私合规分析专家。必须按要求输出 JSON，不要输出 JSON 之外的说明。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": temperature,
        "stream": False,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                print(f"[WARN] 触发限流，第 {attempt}/{MAX_RETRIES} 次重试...")
                time.sleep(RETRY_SLEEP * attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.ReadTimeout:
            print(f"[WARN] 读取超时，第 {attempt}/{MAX_RETRIES} 次重试...")
        except requests.exceptions.ConnectTimeout:
            print(f"[WARN] 连接超时，第 {attempt}/{MAX_RETRIES} 次重试...")
        except requests.exceptions.RequestException as e:
            print(f"[WARN] 请求异常: {e}，第 {attempt}/{MAX_RETRIES} 次重试...")
        except Exception as e:
            print(f"[WARN] 未知异常: {e}，第 {attempt}/{MAX_RETRIES} 次重试...")
        time.sleep(RETRY_SLEEP * attempt)

    raise RuntimeError("LLM 调用失败，已超过最大重试次数。")


# =========================
# 权限加载与语义推断
# =========================


def find_permission_text_candidates(app_name: str, base_dir: Path) -> List[Path]:
    return [base_dir / f"{app_name}.txt", base_dir / f"{app_name}.apk.txt"]



def parse_mapped_permission_file(mapped_path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not mapped_path.exists():
        return result
    text = safe_read_text(mapped_path)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            perm, desc = line.split("\t", 1)
            result[perm.strip()] = normalize_space(desc)
        else:
            result[line] = ""
    return result



def load_mapped_permissions(app_name: str) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for candidate in find_permission_text_candidates(app_name, MAPPED_PERMISSION_DIR):
        if candidate.exists():
            current = parse_mapped_permission_file(candidate)
            merged.update(current)
            print(f"[INFO] 已读取 apk映射：{candidate}，命中 {len(current)} 条")
    return merged



def load_raw_permissions(app_name: str) -> List[str]:
    for candidate in [RAW_PERMISSION_DIR / f"{app_name}.apk.txt", RAW_PERMISSION_DIR / f"{app_name}.txt"]:
        if candidate.exists():
            raw_text = safe_read_text(candidate)
            perms = [normalize_space(line) for line in raw_text.splitlines() if normalize_space(line)]
            return dedup_keep_order(perms)
    return []



def weak_vendor_semantics(permission_name: str) -> Tuple[str, List[str], str]:
    p = permission_name.lower()
    if any(x in p for x in ["badge", "launcher", "shortcut"]):
        return "通知/角标", ["通知/角标", "应用列表/应用信息"], "根据权限名推断为桌面角标/启动器相关兼容权限"
    if any(x in p for x in ["push", "mipush", "c2d", "notification"]):
        return "通知/角标", ["通知/角标", "后台运行/自启动"], "根据权限名推断为推送/通知相关兼容权限"
    if any(x in p for x in ["broadcast", "receiver"]):
        return "后台运行/自启动", ["后台运行/自启动"], "根据权限名推断为广播/接收器相关权限"
    if any(x in p for x in ["oaid", "msa", "did", "identifier"]):
        return "设备信息/标识", ["设备信息/标识"], "根据权限名推断为设备标识相关权限"
    if any(x in p for x in ["install", "package"]):
        return "安装/卸载应用", ["安装/卸载应用", "应用列表/应用信息"], "根据权限名推断为安装包/应用管理相关权限"
    if any(x in p for x in ["gyro", "sensor"]):
        return "传感器", ["传感器"], "根据权限名推断为传感器相关权限"
    return "其他", [], "权限名无法稳定推断，仅作弱语义处理"



def infer_permission_category(permission_name: str, description: str) -> Tuple[str, List[str], str, str]:
    p = permission_name.lower()
    d = (description or "").lower()
    text = f"{p} {d}"

    def hit(*words: str) -> bool:
        return any(w in text for w in words)

    if hit("camera", "flashlight"):
        return "相机", ["相机"], "从权限名/说明识别到 camera 语义", "高"
    if hit("record_audio", "microphone", "audio", "modify_audio"):
        return "麦克风", ["麦克风"], "从权限名/说明识别到音频/麦克风语义", "高"
    if hit("fine_location", "coarse_location", "location", "gps"):
        return "位置", ["位置"], "从权限名/说明识别到定位语义", "高"
    if hit("read_external_storage", "write_external_storage", "manage_external_storage", "read_media", "media_images", "media_video", "media_visual"):
        return "存储/相册", ["存储/相册"], "从权限名/说明识别到存储/相册语义", "高"
    if hit("calendar"):
        return "日历", ["日历"], "从权限名/说明识别到日历语义", "高"
    if hit("query_all_packages", "package_usage", "get_tasks"):
        return "应用列表/应用信息", ["应用列表/应用信息"], "从权限名/说明识别到应用列表/包信息语义", "高"
    if hit("alert_window", "overlay_window"):
        return "悬浮窗", ["悬浮窗"], "从权限名/说明识别到悬浮窗语义", "高"
    if hit("notification", "badge", "push"):
        return "通知/角标", ["通知/角标"], "从权限名/说明识别到通知/角标语义", "中"
    if hit("boot_completed", "foreground_service", "wake_lock", "background", "restart_packages"):
        return "后台运行/自启动", ["后台运行/自启动"], "从权限名/说明识别到后台运行/自启动语义", "中"
    if hit("wifi", "network", "internet", "local_network"):
        return "网络状态", ["网络状态"], "从权限名/说明识别到网络语义", "高"
    if hit("vibrate", "sensor", "accelerometer", "gyroscope"):
        return "传感器", ["传感器"], "从权限名/说明识别到传感器语义", "中"
    if hit("clipboard"):
        return "剪贴板", ["剪贴板"], "从权限名/说明识别到剪贴板语义", "中"
    if hit("contact"):
        return "联系人", ["联系人"], "从权限名/说明识别到联系人语义", "高"
    if hit("call_phone", "phone", "account", "login", "credential", "authenticate"):
        return "电话/账号", ["电话/账号"], "从权限名/说明识别到电话/账号语义", "中"
    if hit("sms", "mms", "respond_via_message"):
        return "短信", ["短信"], "从权限名/说明识别到短信语义", "中"
    if hit("bluetooth", "nfc", "nearby"):
        return "蓝牙/NFC/附近设备", ["蓝牙/NFC/附近设备"], "从权限名/说明识别到蓝牙/NFC/附近设备语义", "中"
    if hit("install_packages", "request_install_packages", "delete_packages"):
        return "安装/卸载应用", ["安装/卸载应用"], "从权限名/说明识别到安装/卸载应用语义", "中"

    return weak_vendor_semantics(permission_name) + ("低",)



def infer_permission_group(permission_name: str, category: str, is_system: bool) -> str:
    p = permission_name.lower()
    if not is_system and any(x in p for x in ["launcher", "badge", "push", "mipush", "c2d", "broadcast", "shortcut", "receiver"]):
        return "厂商兼容权限"
    if category in SENSITIVE_CATEGORIES:
        return "敏感系统权限" if is_system else "厂商兼容敏感权限"
    if category in GENERAL_SYSTEM_CATEGORIES or is_system:
        return "一般系统权限" if is_system else "厂商兼容权限"
    return "厂商兼容权限" if not is_system else "一般系统权限"



def build_permission_item(permission_name: str, mapped_dict: Dict[str, str]) -> Dict[str, Any]:
    description = normalize_space(mapped_dict.get(permission_name, ""))
    is_system = permission_name.startswith("android.permission.")
    category, expected_caps, semantic_reason, initial_conf = infer_permission_category(permission_name, description)
    permission_group = infer_permission_group(permission_name, category, is_system)

    if description:
        explanation_source = "apk映射"
        if initial_conf == "低":
            initial_conf = "中"
    elif is_system:
        explanation_source = "系统权限名推断"
    else:
        explanation_source = "第三方/厂商权限名弱推断"

    return {
        "permission_name": permission_name,
        "description": description,
        "is_system": is_system,
        "category": category,
        "expected_capabilities": expected_caps,
        "permission_group": permission_group,
        "semantic_reason": semantic_reason,
        "initial_confidence": initial_conf,
        "explanation_source": explanation_source,
    }



def load_permissions_for_policy_analysis(app_name: str) -> List[Dict[str, Any]]:
    raw_permissions = load_raw_permissions(app_name)
    mapped_dict = load_mapped_permissions(app_name)
    return [build_permission_item(perm, mapped_dict) for perm in raw_permissions]


# =========================
# 阶段一：能力块抽取
# =========================


def deduplicate_capabilities(capability_blocks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    results = []
    for item in capability_blocks:
        ctype = normalize_space(item.get("type", ""))
        evidence = normalize_space(item.get("evidence", ""))
        source = normalize_space(item.get("source", ""))
        key = (ctype, evidence)
        if ctype and evidence and key not in seen:
            seen.add(key)
            results.append({"type": ctype, "evidence": evidence, "source": source or "unknown"})
    return results



def rule_extract_capabilities(policy_text: str) -> List[Dict[str, str]]:
    sentences = split_sentences(policy_text)
    caps: List[Dict[str, str]] = []
    for sent in sentences:
        lower = sent.lower()
        for cap_type, keywords in CATEGORY_KEYWORDS.items():
            if cap_type == "其他":
                continue
            if any(k.lower() in lower for k in keywords):
                caps.append({"type": cap_type, "evidence": sent, "source": "rule"})
    return deduplicate_capabilities(caps)



def parse_capability_output(raw: str) -> List[Dict[str, str]]:
    data = extract_json_payload(raw)
    items: List[Dict[str, Any]] = []

    if isinstance(data, list):
        items = ensure_list_of_dicts(data)
    elif isinstance(data, dict):
        for key in ("capabilities", "blocks", "items", "data"):
            if isinstance(data.get(key), list):
                items = ensure_list_of_dicts(data.get(key))
                break
        if not items and data.get("type") and data.get("evidence"):
            items = [data]

    results: List[Dict[str, str]] = []
    for item in items:
        ctype = normalize_space(str(item.get("type") or item.get("category") or item.get("capability") or ""))
        evidence = normalize_space(str(item.get("evidence") or item.get("text") or item.get("sentence") or ""))
        if ctype in ALLOWED_CAPABILITY_TYPES and evidence:
            results.append({"type": ctype, "evidence": evidence, "source": "llm"})
    return deduplicate_capabilities(results)



def llm_extract_capabilities(window_text: str) -> List[Dict[str, str]]:
    prompt = f"""
请从下面这段移动应用隐私政策文本中，抽取与权限、设备能力、数据访问行为相关的能力块。

能力类型只允许从下面列表中选择，不要新增类别：
{json.dumps(sorted(list(ALLOWED_CAPABILITY_TYPES - {'其他'})), ensure_ascii=False)}

要求：
1. 只提取文本中有直接证据支撑的内容。
2. evidence 必须尽量保留原文关键句，不要编造。
3. 如果没有相关内容，可以返回空数组。
4. 只输出 JSON。你可以输出以下任一种结构：
   - {{"capabilities": [{{"type": "相机", "evidence": "..."}}]}}
   - [{{"type": "相机", "evidence": "..."}}]

隐私政策文本：
{window_text}
""".strip()

    raw = call_llm(prompt, temperature=0.1)
    return parse_capability_output(raw)



def extract_capabilities(policy_text: str) -> List[Dict[str, str]]:
    windows = sliding_windows(policy_text)
    all_caps = rule_extract_capabilities(policy_text)
    print(f"[INFO] 规则层预抽取能力块：{len(all_caps)} 个")
    print(f"[INFO] 隐私政策切分为 {len(windows)} 个窗口，开始抽取能力块...")

    for i, window_text in enumerate(windows, 1):
        try:
            caps = llm_extract_capabilities(window_text)
            all_caps.extend(caps)
            print(f"[INFO] 窗口 {i}/{len(windows)}：提取到 {len(caps)} 个能力块")
        except Exception as e:
            print(f"[WARN] 窗口 {i}/{len(windows)} 能力块抽取失败：{e}")

    return deduplicate_capabilities(all_caps)


# =========================
# 阶段二：证据召回与匹配
# =========================


def retrieve_relevant_evidence(policy_text: str, capability_blocks: List[Dict[str, str]], permission_item: Dict[str, Any], top_k: int = 8) -> List[str]:
    category = permission_item.get("category", "")
    expected_caps = permission_item.get("expected_capabilities", []) or []
    permission_name = permission_item["permission_name"].lower()
    description = (permission_item.get("description", "") or "").lower()

    candidate_sentences = split_sentences(policy_text)
    for cap in capability_blocks:
        ev = normalize_space(cap.get("evidence", ""))
        if ev:
            candidate_sentences.append(ev)
    candidate_sentences = dedup_keep_order(candidate_sentences)

    scored: List[Tuple[int, str]] = []
    for sent in candidate_sentences:
        lower = sent.lower()
        score = 0

        if category and category in sent:
            score += 6

        for cap in [category] + expected_caps:
            for kw in CATEGORY_KEYWORDS.get(cap, []):
                if kw.lower() in lower:
                    score += 3

        for token in re.split(r"[_\.]+", permission_name):
            token = token.strip()
            if len(token) >= 4 and token in lower:
                score += 1

        for token in re.split(r"[_\s\.-]+", description):
            token = token.strip()
            if len(token) >= 4 and token in lower:
                score += 1

        # 一些高价值原文特征加分，避免 SDK 列表类政策被漏掉
        if category == "设备信息/标识" and any(x in lower for x in ["imei", "imsi", "oaid", "android id", "设备标识", "设备信息"]):
            score += 5
        if category == "应用列表/应用信息" and any(x in lower for x in ["已安装应用", "应用列表", "包信息"]):
            score += 5
        if category == "存储/相册" and any(x in lower for x in ["存储的个人文件", "本地文件", "图片", "照片", "视频", "相册"]):
            score += 5
        if category == "网络状态" and any(x in lower for x in ["网络信息", "ip地址", "wifi", "移动网络", "联网"]):
            score += 4
        if category == "电话/账号" and any(x in lower for x in ["手机状态", "读取手机状态和身份", "手机号", "账号", "认证"]):
            score += 4
        if category == "通知/角标" and any(x in lower for x in ["消息提醒", "推送", "角标", "通知"]):
            score += 4
        if category == "安装/卸载应用" and any(x in lower for x in ["安装", "更新", "安装包"]):
            score += 4

        if score > 0:
            scored.append((score, sent))

    scored.sort(key=lambda x: (-x[0], len(x[1])))
    return [s for _, s in scored[:top_k]]



def quick_rule_match(permission_item: Dict[str, Any], evidence_list: List[str]) -> Optional[Dict[str, Any]]:
    category = permission_item.get("category", "")
    if not category or category == "其他" or not evidence_list:
        return None

    hit_count = 0
    for sent in evidence_list:
        lower = sent.lower()
        if any(k.lower() in lower for k in CATEGORY_KEYWORDS.get(category, [])):
            hit_count += 1

    if category in SENSITIVE_CATEGORIES:
        if hit_count >= 2:
            return {
                "status": "已明确告知",
                "reason": f"隐私政策中存在与 {category} 直接对应的功能或数据处理场景，证据较直接。",
                "evidence": evidence_list[:3],
                "final_confidence": "高",
            }
        return None

    if hit_count >= 2:
        return {
            "status": "已明确告知",
            "reason": f"隐私政策中存在与 {category} 直接对应的功能或数据处理场景，证据较为直接。",
            "evidence": evidence_list[:3],
            "final_confidence": "高",
        }
    if hit_count >= 1:
        return {
            "status": "间接告知",
            "reason": f"隐私政策中存在与 {category} 相关的描述，但未逐字对应到具体权限，适合判为间接告知。",
            "evidence": evidence_list[:3],
            "final_confidence": "中",
        }
    return None



def adjust_status_for_low_confidence(status: str, permission_item: Dict[str, Any]) -> str:
    if permission_item.get("initial_confidence") != "低":
        return status
    # 对第三方/厂商弱推断权限收紧一档，避免乱判“已明确告知”
    if status == "已明确告知":
        return "间接告知"
    return status



def build_permission_prompt(permission_item: Dict[str, Any], evidence_list: List[str], capability_blocks: List[Dict[str, str]]) -> str:
    permission_context = {
        "permission_name": permission_item["permission_name"],
        "is_system": permission_item.get("is_system", False),
        "category": permission_item.get("category", ""),
        "expected_capabilities": permission_item.get("expected_capabilities", []),
        "permission_group": permission_item.get("permission_group", ""),
        "semantic_reason": permission_item.get("semantic_reason", ""),
        "initial_confidence": permission_item.get("initial_confidence", ""),
        "explanation_source": permission_item.get("explanation_source", ""),
        "description": permission_item.get("description", "") or "无映射解释，已基于权限名进行语义推断",
    }

    focused_caps = []
    expected_set = set(permission_item.get("expected_capabilities", []))
    for cap in capability_blocks:
        if cap.get("type") in expected_set or cap.get("type") == permission_item.get("category"):
            focused_caps.append(cap)
    focused_caps = focused_caps[:10]

    prompt = f"""
你是一名移动应用隐私合规分析专家。请根据给定权限信息、原文证据句和能力块，判断该权限在隐私政策中是否得到说明。

判断标准：
1. 已明确告知：隐私政策中有与该权限直接对应的能力、数据类型或使用场景，证据直接清晰。
2. 间接告知：隐私政策中有相关场景，但没有直接点明该权限，需要适度推断。
3. 未告知：没有足够相关证据，或者证据过弱。

重要约束：
- 必须基于给定证据判断，不得凭空臆测。
- 若权限是第三方/厂商权限且 initial_confidence=低，请更保守，除非证据很强，否则不要判“已明确告知”。
- 对 launcher/badge/push/broadcast 等兼容权限，若隐私政策已描述角标、桌面图标、消息提醒、系统广播等，可考虑“间接告知”。
- 只输出 JSON。

输出格式：
{{
  "status": "已明确告知" 或 "间接告知" 或 "未告知",
  "reason": "简要专业说明",
  "evidence": ["证据1", "证据2"],
  "final_confidence": "高" 或 "中" 或 "低"
}}

权限信息：
{json.dumps(permission_context, ensure_ascii=False, indent=2)}

证据句：
{json.dumps(evidence_list, ensure_ascii=False, indent=2)}

相关能力块：
{json.dumps(focused_caps, ensure_ascii=False, indent=2)}
""".strip()
    return prompt



def parse_match_output(raw: str, fallback_evidence: List[str]) -> Dict[str, Any]:
    data = extract_json_payload(raw)
    if not isinstance(data, dict):
        return {
            "status": "未告知",
            "reason": "模型返回结果无法解析，默认按未告知处理。",
            "evidence": fallback_evidence[:3],
            "final_confidence": "低",
        }

    status = normalize_space(str(data.get("status", "")))
    if status not in {"已明确告知", "间接告知", "未告知"}:
        status = "未告知"
    reason = normalize_space(str(data.get("reason", ""))) or "未提供有效原因说明。"
    evidence = normalize_evidence_list(data.get("evidence")) or fallback_evidence[:3]
    final_conf = normalize_space(str(data.get("final_confidence", "")))
    if final_conf not in {"高", "中", "低"}:
        final_conf = "中" if status != "未告知" else "低"
    return {"status": status, "reason": reason, "evidence": evidence, "final_confidence": final_conf}



def match_permission(permission_item: Dict[str, Any], policy_text: str, capability_blocks: List[Dict[str, str]]) -> Dict[str, Any]:
    evidence_list = retrieve_relevant_evidence(policy_text, capability_blocks, permission_item, top_k=8)

    # 先尝试规则直判，高频稳定场景直接落地
    rule_result = quick_rule_match(permission_item, evidence_list)
    if rule_result is not None:
        rule_result["status"] = adjust_status_for_low_confidence(rule_result["status"], permission_item)
        if permission_item.get("initial_confidence") == "低" and rule_result["final_confidence"] == "高":
            rule_result["final_confidence"] = "中"
        return rule_result

    # 即使能力块为空，只要原文召回到了证据，也继续交给 LLM 判定
    prompt = build_permission_prompt(permission_item, evidence_list, capability_blocks)
    raw = call_llm(prompt, temperature=0.1)
    result = parse_match_output(raw, evidence_list)
    result["status"] = adjust_status_for_low_confidence(result["status"], permission_item)
    if permission_item.get("initial_confidence") == "低" and result["final_confidence"] == "高":
        result["final_confidence"] = "中"
    return result


# =========================
# Markdown 输出
# =========================


def summarize_by_group(analysis_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: {"已明确告知": 0, "间接告知": 0, "未告知": 0, "总数": 0})
    for item in analysis_results:
        group = item["permission"].get("permission_group", "未分组")
        status = item["result"].get("status", "未告知")
        summary[group]["总数"] += 1
        summary[group][status] += 1
    return dict(summary)



def generate_md(app_name: str, capability_blocks: List[Dict[str, str]], analysis_results: List[Dict[str, Any]]) -> str:
    total = len(analysis_results)
    explicit_count = sum(1 for x in analysis_results if x["result"]["status"] == "已明确告知")
    indirect_count = sum(1 for x in analysis_results if x["result"]["status"] == "间接告知")
    missing_count = sum(1 for x in analysis_results if x["result"]["status"] == "未告知")
    group_summary = summarize_by_group(analysis_results)

    lines: List[str] = []
    lines.append(f"# {app_name} 隐私政策和权限分析结果\n")
    lines.append("## 一、分析说明\n")
    lines.append("本报告用于评估 APK 权限与隐私政策描述的一致性。")
    lines.append("本版分析逻辑：")
    lines.append("- 使用原始 APK 权限文件与 apk 映射文件；")
    lines.append("- 标准系统权限不再依赖映射是否命中；")
    lines.append("- 能力块抽取同时使用规则层和 LLM；")
    lines.append("- 能力块为空时，仍直接基于隐私政策原文召回证据；")
    lines.append("- 第三方/厂商权限若无映射解释，则基于权限名做弱语义判断，并降低置信度；")
    lines.append("- 只输出 Markdown 报告。\n")

    lines.append("## 二、总体统计\n")
    lines.append(f"- 权限总数：{total}")
    lines.append(f"- 已明确告知：{explicit_count}")
    lines.append(f"- 间接告知：{indirect_count}")
    lines.append(f"- 未告知：{missing_count}\n")

    lines.append("## 三、分层统计\n")
    for group, stat in group_summary.items():
        lines.append(f"### {group}")
        lines.append(f"- 总数：{stat['总数']}")
        lines.append(f"- 已明确告知：{stat['已明确告知']}")
        lines.append(f"- 间接告知：{stat['间接告知']}")
        lines.append(f"- 未告知：{stat['未告知']}\n")

    lines.append("## 四、隐私政策能力块提取结果\n")
    if capability_blocks:
        for i, cap in enumerate(capability_blocks, 1):
            lines.append(f"### {i}. {cap['type']}")
            lines.append(f"- 证据：{cap['evidence']}")
            if cap.get("source"):
                lines.append(f"- 来源：{cap['source']}")
            lines.append("")
    else:
        lines.append("未从隐私政策中抽取到明显的权限相关能力块，但后续分析仍会直接回看原文证据。\n")

    lines.append("## 五、逐权限分析结果\n")
    for idx, item in enumerate(analysis_results, 1):
        perm = item["permission"]
        result = item["result"]
        lines.append(f"### {idx}. {perm['permission_name']}")
        lines.append(f"- 权限类型：{'系统权限' if perm.get('is_system') else '非系统权限'}")
        lines.append(f"- 权限分层：{perm.get('permission_group', '未分层')}")
        lines.append(f"- 归一化类别：{perm.get('category', '其他')}")
        lines.append(f"- 解释来源：{perm.get('explanation_source', '未知')}")
        lines.append(f"- 初始解释置信度：{perm.get('initial_confidence', '低')}")
        if perm.get("description"):
            lines.append(f"- 权限解释：{perm['description']}")
        else:
            lines.append("- 权限解释：无映射解释，已基于权限名做语义推断")
        lines.append(f"- 语义提示：{perm.get('semantic_reason', '')}")
        lines.append(f"- 合规状态：**{result['status']}**")
        lines.append(f"- 最终置信度：{result.get('final_confidence', '低')}")
        lines.append(f"- 原因说明：{result['reason']}")
        if result.get("evidence"):
            lines.append("- 证据：")
            for ev in result["evidence"]:
                lines.append(f"  - {ev}")
        else:
            lines.append("- 证据：无明确证据")
        lines.append("")

    return "\n".join(lines)


# =========================
# 单应用分析
# =========================


def analyze_one_app(policy_path: Path):
    app_name = policy_path.stem
    print(f"\n{'=' * 80}")
    print(f"开始分析：{app_name}")
    print(f"{'=' * 80}")

    policy_text = safe_read_text(policy_path).strip()
    if not policy_text:
        print(f"[WARN] 隐私政策为空，跳过：{policy_path}")
        return

    permissions = load_permissions_for_policy_analysis(app_name)
    if not permissions:
        print(f"[WARN] 未找到 APK 权限，跳过：{app_name}")
        return

    print(f"[INFO] 共读取权限 {len(permissions)} 个")
    system_count = sum(1 for x in permissions if x["is_system"])
    print(f"[INFO] 系统权限 {system_count} 个，非系统权限 {len(permissions) - system_count} 个")

    capability_blocks = extract_capabilities(policy_text)
    print(f"[INFO] 去重后能力块数量：{len(capability_blocks)}")

    analysis_results: List[Dict[str, Any]] = []
    print("[INFO] 开始逐权限匹配分析...")
    for idx, perm_item in enumerate(permissions, 1):
        perm_name = perm_item["permission_name"]
        try:
            result = match_permission(perm_item, policy_text, capability_blocks)
            analysis_results.append({"permission": perm_item, "result": result})
            print(f"[INFO] {idx}/{len(permissions)} {perm_name} -> {result['status']} ({result.get('final_confidence', '低')})")
        except Exception as e:
            print(f"[WARN] 分析权限失败：{perm_name}，错误：{e}")
            analysis_results.append({
                "permission": perm_item,
                "result": {
                    "status": "未告知",
                    "reason": f"分析过程异常：{e}",
                    "evidence": [],
                    "final_confidence": "低",
                },
            })

    app_output_dir = OUTPUT_DIR / app_name
    ensure_dir(app_output_dir)
    output_md = app_output_dir / "隐私政策和权限分析结果.md"
    output_md.write_text(generate_md(app_name, capability_blocks, analysis_results), encoding="utf-8")
    print(f"[DONE] 报告已输出：{output_md}")


# =========================
# 主程序
# =========================


def main():
    if not POLICY_DIR.exists():
        print(f"[ERROR] 隐私政策目录不存在：{POLICY_DIR}")
        return

    policy_files = sorted(POLICY_DIR.glob("*.txt"))
    if not policy_files:
        print(f"[ERROR] 未找到隐私政策文件：{POLICY_DIR}")
        return

    print(f"[INFO] 共发现 {len(policy_files)} 个隐私政策文件")
    for policy_path in policy_files:
        try:
            analyze_one_app(policy_path)
        except Exception as e:
            print(f"[ERROR] 分析应用失败：{policy_path.stem}，错误：{e}")

    print("\n全部分析完成。")


if __name__ == "__main__":
    main()
