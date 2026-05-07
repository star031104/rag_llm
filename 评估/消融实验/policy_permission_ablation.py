# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common_ablation import (
    ExperimentContext,
    LLMClient,
    dedup_keep_order,
    extract_json_payload,
    normalize_evidence_list,
    normalize_space,
    safe_read_text,
    safe_write_text,
    shorten,
    sliding_windows,
    split_sentences,
)
from exp_config import ExperimentConfig, get_experiment


CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "相机": ["相机", "摄像头", "拍照", "拍摄", "扫码", "扫描", "证件照", "人脸"],
    "麦克风": ["麦克风", "录音", "语音", "音频", "语音输入"],
    "位置": ["位置", "定位", "地理位置", "经纬度", "gps", "附近"],
    "存储/相册": ["存储", "本地文件", "文件", "相册", "照片", "图片", "视频", "媒体", "下载", "上传"],
    "联系人": ["联系人", "通讯录"],
    "日历": ["日历", "calendar", "提醒事项"],
    "应用列表/应用信息": ["应用列表", "已安装应用", "应用信息", "包信息", "launcher", "默认浏览器", "桌面图标"],
    "通知/角标": ["通知", "消息提醒", "推送", "角标", "badge", "红点"],
    "悬浮窗": ["悬浮窗", "悬浮球", "浮窗", "overlay"],
    "截屏": ["截屏", "截图", "录屏", "screen capture", "media projection"],
    "网络状态": ["网络", "wifi", "联网", "热点", "连接状态", "本地网络", "ip地址"],
    "后台运行/自启动": ["后台", "自启动", "保活", "开机", "启动", "前台服务", "系统广播"],
    "设备信息/标识": ["设备信息", "设备标识", "imei", "imsi", "oaid", "android id", "设备型号"],
    "传感器": ["传感器", "陀螺仪", "加速度", "光线传感器"],
    "剪贴板": ["剪贴板", "复制", "粘贴"],
    "电话/账号": ["手机号", "电话", "通话", "账号", "登录", "认证", "实名"],
    "短信": ["短信", "彩信", "message"],
    "蓝牙/NFC/附近设备": ["蓝牙", "bluetooth", "nfc", "附近设备", "nearby"],
    "安装/卸载应用": ["安装", "卸载", "安装包", "更新应用", "request install packages"],
    "其他": [],
}
SENSITIVE_CATEGORIES = {"相机", "麦克风", "位置", "存储/相册", "联系人", "日历", "应用列表/应用信息", "通知/角标", "悬浮窗", "截屏"}
ALLOWED_CAPABILITY_TYPES = set(CATEGORY_KEYWORDS.keys())


class Analyzer:
    def __init__(self, base_dir: Path, experiment: ExperimentConfig):
        self.base_dir = Path(base_dir).resolve()
        self.exp = experiment
        self.ctx = ExperimentContext(experiment, self.base_dir)
        self.policy_dir = self.base_dir / "dataset" / "隐私政策"
        self.raw_permission_dir = self.base_dir / "dataset" / "apk权限"
        self.mapped_permission_dir = self.base_dir / "dataset" / "apk映射"
        self.llm = LLMClient(system_prompt="你是一名严谨的移动应用隐私合规分析专家。必须按要求输出 JSON。")

    def find_permission_text_candidates(self, app_name: str, base_dir: Path) -> List[Path]:
        return [base_dir / f"{app_name}.txt", base_dir / f"{app_name}.apk.txt"]

    def parse_mapped_permission_file(self, mapped_path: Path) -> Dict[str, str]:
        result: Dict[str, str] = {}
        if not mapped_path.exists():
            return result
        for line in safe_read_text(mapped_path).splitlines():
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                perm, desc = line.split("\t", 1)
                result[perm.strip()] = normalize_space(desc)
            else:
                result[line] = ""
        return result

    def load_mapped_permissions(self, app_name: str) -> Dict[str, str]:
        merged: Dict[str, str] = {}
        for candidate in self.find_permission_text_candidates(app_name, self.mapped_permission_dir):
            if candidate.exists():
                merged.update(self.parse_mapped_permission_file(candidate))
        return merged

    def load_raw_permissions(self, app_name: str) -> List[str]:
        for candidate in [self.raw_permission_dir / f"{app_name}.apk.txt", self.raw_permission_dir / f"{app_name}.txt"]:
            if candidate.exists():
                perms = [normalize_space(line) for line in safe_read_text(candidate).splitlines() if normalize_space(line)]
                return dedup_keep_order(perms)
        return []

    def weak_vendor_semantics(self, permission_name: str):
        """
        E2: w/o Semantic Normalization 时使用的弱语义推断。
        必须返回 4 个值，与 infer_permission_category 的返回格式保持一致：
        (category, expected_caps, semantic_reason, initial_conf)
        """
        p = (permission_name or "").upper()

        mapping = {
            "CAMERA": (
                "相机",
                ["拍照", "上传图片", "扫码", "头像设置", "图像采集"],
                "基于权限名粗粒度判断，该权限通常对应相机/拍摄相关能力。",
                0.72,
            ),
            "RECORD_AUDIO": (
                "麦克风",
                ["录音", "语音输入", "语音通话", "语音识别"],
                "基于权限名粗粒度判断，该权限通常对应麦克风/录音相关能力。",
                0.72,
            ),
            "ACCESS_FINE_LOCATION": (
                "位置",
                ["定位", "导航", "附近推荐", "位置服务", "打卡"],
                "基于权限名粗粒度判断，该权限通常对应精确位置能力。",
                0.74,
            ),
            "ACCESS_COARSE_LOCATION": (
                "位置",
                ["定位", "附近推荐", "位置服务"],
                "基于权限名粗粒度判断，该权限通常对应粗略位置能力。",
                0.70,
            ),
            "READ_EXTERNAL_STORAGE": (
                "存储/相册",
                ["读取图片", "选择文件", "上传附件", "读取相册"],
                "基于权限名粗粒度判断，该权限通常对应外部存储读取能力。",
                0.70,
            ),
            "WRITE_EXTERNAL_STORAGE": (
                "存储/相册",
                ["保存图片", "导出文件", "缓存文件", "写入相册"],
                "基于权限名粗粒度判断，该权限通常对应外部存储写入能力。",
                0.70,
            ),
            "READ_MEDIA_IMAGES": (
                "存储/相册",
                ["读取图片", "选择图片", "上传图片", "访问相册"],
                "基于权限名粗粒度判断，该权限通常对应图片媒体读取能力。",
                0.72,
            ),
            "READ_MEDIA_VIDEO": (
                "存储/相册",
                ["读取视频", "选择视频", "上传视频", "访问相册"],
                "基于权限名粗粒度判断，该权限通常对应视频媒体读取能力。",
                0.72,
            ),
            "READ_MEDIA_AUDIO": (
                "存储/相册",
                ["读取音频", "导入音频", "访问音频文件"],
                "基于权限名粗粒度判断，该权限通常对应音频媒体读取能力。",
                0.70,
            ),
            "READ_CONTACTS": (
                "联系人",
                ["读取通讯录", "邀请好友", "联系人匹配"],
                "基于权限名粗粒度判断，该权限通常对应联系人读取能力。",
                0.74,
            ),
            "WRITE_CONTACTS": (
                "联系人",
                ["写入联系人", "保存联系人"],
                "基于权限名粗粒度判断，该权限通常对应联系人写入能力。",
                0.70,
            ),
            "READ_CALENDAR": (
                "日历",
                ["读取日程", "同步日历", "提醒事项"],
                "基于权限名粗粒度判断，该权限通常对应日历读取能力。",
                0.70,
            ),
            "WRITE_CALENDAR": (
                "日历",
                ["写入日程", "创建提醒", "同步事件"],
                "基于权限名粗粒度判断，该权限通常对应日历写入能力。",
                0.70,
            ),
            "POST_NOTIFICATIONS": (
                "通知/角标",
                ["消息提醒", "通知推送", "服务通知"],
                "基于权限名粗粒度判断，该权限通常对应通知发送能力。",
                0.72,
            ),
            "SYSTEM_ALERT_WINDOW": (
                "悬浮窗",
                ["悬浮球", "悬浮窗展示", "桌面覆盖层"],
                "基于权限名粗粒度判断，该权限通常对应悬浮窗能力。",
                0.76,
            ),
            "READ_PHONE_STATE": (
                "电话/账号",
                ["设备识别", "通话状态识别", "账号关联"],
                "基于权限名粗粒度判断，该权限通常对应电话状态或设备识别能力。",
                0.68,
            ),
            "READ_SMS": (
                "短信",
                ["读取验证码", "短信验证", "短信内容读取"],
                "基于权限名粗粒度判断，该权限通常对应短信读取能力。",
                0.78,
            ),
            "BLUETOOTH_CONNECT": (
                "蓝牙/NFC/附近设备",
                ["连接蓝牙设备", "附近设备交互"],
                "基于权限名粗粒度判断，该权限通常对应蓝牙连接能力。",
                0.70,
            ),
            "NFC": (
                "蓝牙/NFC/附近设备",
                ["NFC识别", "近场通信"],
                "基于权限名粗粒度判断，该权限通常对应NFC能力。",
                0.70,
            ),
            "QUERY_ALL_PACKAGES": (
                "应用列表/应用信息",
                ["读取应用列表", "应用检测", "应用兼容性判断"],
                "基于权限名粗粒度判断，该权限通常对应应用列表读取能力。",
                0.80,
            ),
        }

        for key, value in mapping.items():
            if key in p:
                return value

        # 默认兜底：仍然返回 4 个值
        return (
            "其他",
            [],
            "未进行语义规范化，仅根据原始权限名进行弱语义兜底判断。",
            0.50,
        )

    def infer_permission_category(self, permission_name: str, description: str) -> Tuple[str, List[str], str, str]:
        text = f"{permission_name.lower()} {(description or '').lower()}"
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
        if hit("boot_completed", "foreground_service", "wake_lock", "background"):
            return "后台运行/自启动", ["后台运行/自启动"], "从权限名/说明识别到后台运行/自启动语义", "中"
        if hit("wifi", "network", "internet", "local_network"):
            return "网络状态", ["网络状态"], "从权限名/说明识别到网络语义", "高"
        if hit("vibrate", "sensor", "accelerometer", "gyroscope"):
            return "传感器", ["传感器"], "从权限名/说明识别到传感器语义", "中"
        if hit("clipboard"):
            return "剪贴板", ["剪贴板"], "从权限名/说明识别到剪贴板语义", "中"
        if hit("contact"):
            return "联系人", ["联系人"], "从权限名/说明识别到联系人语义", "高"
        if hit("call_phone", "phone", "account", "login", "authenticate"):
            return "电话/账号", ["电话/账号"], "从权限名/说明识别到电话/账号语义", "中"
        if hit("sms", "mms"):
            return "短信", ["短信"], "从权限名/说明识别到短信语义", "中"
        if hit("bluetooth", "nfc", "nearby"):
            return "蓝牙/NFC/附近设备", ["蓝牙/NFC/附近设备"], "从权限名/说明识别到蓝牙/NFC/附近设备语义", "中"
        if hit("install_packages", "request_install_packages", "delete_packages"):
            return "安装/卸载应用", ["安装/卸载应用"], "从权限名/说明识别到安装/卸载应用语义", "中"
        return self.weak_vendor_semantics(permission_name) + ("低",)

    def build_permission_item(self, permission_name: str, mapped_dict: Dict[str, str]) -> Dict[str, Any]:
        description = normalize_space(mapped_dict.get(permission_name, ""))
        is_system = permission_name.startswith("android.permission.")
        if self.exp.use_permission_semantic_enrichment:
            result = self.infer_permission_category(permission_name, description)

            if isinstance(result, (list, tuple)):
                if len(result) >= 4:
                    category = result[0]
                    expected_caps = result[1]
                    semantic_reason = result[2]
                    initial_conf = result[3]
                elif len(result) == 3:
                    category, expected_caps, semantic_reason = result
                    initial_conf = 0.70
                else:
                    category, expected_caps, semantic_reason, initial_conf = "其他", [], "权限语义推断返回值异常，使用默认值。", 0.50
            else:
                category, expected_caps, semantic_reason, initial_conf = "其他", [], "权限语义推断返回值异常，使用默认值。", 0.50
        else:
            # E4：去掉语义增强后的“粗粒度版”
            category, expected_caps, semantic_reason, initial_conf = self.weak_vendor_semantics(permission_name)
            if is_system and category == "其他":
                category, expected_caps, semantic_reason, initial_conf = "其他", [], "消融实验关闭语义增强，仅保留原始权限名", "低"
        if not description:
            description = "无映射解释"
        return {
            "permission_name": permission_name,
            "description": description,
            "is_system": is_system,
            "category": category,
            "expected_capabilities": expected_caps,
            "semantic_reason": semantic_reason,
            "initial_confidence": initial_conf,
        }

    def load_permissions(self, app_name: str) -> List[Dict[str, Any]]:
        mapped = self.load_mapped_permissions(app_name)
        return [self.build_permission_item(perm, mapped) for perm in self.load_raw_permissions(app_name)]

    def deduplicate_capabilities(self, blocks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        out = []
        for item in blocks:
            ctype = normalize_space(item.get("type", ""))
            evidence = normalize_space(item.get("evidence", ""))
            key = (ctype, evidence)
            if ctype and evidence and key not in seen:
                seen.add(key)
                out.append({"type": ctype, "evidence": evidence, "source": item.get("source", "unknown")})
        return out

    def rule_extract_capabilities(self, policy_text: str) -> List[Dict[str, str]]:
        caps: List[Dict[str, str]] = []
        for sent in split_sentences(policy_text):
            lower = sent.lower()
            for cap_type, keywords in CATEGORY_KEYWORDS.items():
                if cap_type == "其他":
                    continue
                if any(k.lower() in lower for k in keywords):
                    caps.append({"type": cap_type, "evidence": sent, "source": "rule"})
        return self.deduplicate_capabilities(caps)

    def parse_capability_output(self, raw: str) -> List[Dict[str, str]]:
        data = extract_json_payload(raw)
        items: List[Dict[str, Any]] = []
        if isinstance(data, list):
            items = [x for x in data if isinstance(x, dict)]
        elif isinstance(data, dict):
            for key in ("capabilities", "blocks", "items", "data"):
                if isinstance(data.get(key), list):
                    items = [x for x in data[key] if isinstance(x, dict)]
                    break
            if not items and data.get("type") and data.get("evidence"):
                items = [data]
        results = []
        for item in items:
            ctype = normalize_space(str(item.get("type") or item.get("category") or item.get("capability") or ""))
            evidence = normalize_space(str(item.get("evidence") or item.get("text") or item.get("sentence") or ""))
            if ctype in ALLOWED_CAPABILITY_TYPES and evidence:
                results.append({"type": ctype, "evidence": evidence, "source": "llm"})
        return self.deduplicate_capabilities(results)

    def llm_extract_capabilities(self, window_text: str) -> List[Dict[str, str]]:
        if self.exp.use_prompt_constraint:
            prompt = f"""
请从下面这段移动应用隐私政策文本中，抽取与权限、设备能力、数据访问行为相关的能力块。
能力类型只允许从下面列表中选择：{json.dumps(sorted(list(ALLOWED_CAPABILITY_TYPES - {'其他'})), ensure_ascii=False)}
只输出 JSON。
隐私政策文本：
{window_text}
""".strip()
        else:
            prompt = f"""
请阅读下面这段移动应用隐私政策文本，抽取其中提到的设备能力、数据访问行为或权限相关功能。
尽量返回 JSON 数组，每项包含 type 和 evidence 两个字段。
隐私政策文本：
{window_text}
""".strip()
        return self.parse_capability_output(self.llm.chat(prompt, temperature=0.1))

    def extract_capabilities(self, policy_text: str) -> List[Dict[str, str]]:
        blocks: List[Dict[str, str]] = []
        if self.exp.use_rule_capability_extraction:
            blocks.extend(self.rule_extract_capabilities(policy_text))
        if self.exp.use_llm_capability_extraction:
            for idx, window in enumerate(sliding_windows(policy_text), 1):
                try:
                    caps = self.llm_extract_capabilities(window)
                    blocks.extend(caps)
                    print(f"[INFO] 能力块窗口 {idx}: {len(caps)} 个")
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] 能力块抽取失败: {exc}")
        return self.deduplicate_capabilities(blocks)

    def retrieve_relevant_evidence(self, policy_text: str, permission_item: Dict[str, Any], capability_blocks: List[Dict[str, str]], max_items: int = 8) -> List[str]:
        evidence: List[str] = []
        if self.exp.use_direct_policy_evidence:
            keywords = CATEGORY_KEYWORDS.get(permission_item.get("category", ""), []) + permission_item.get("expected_capabilities", []) + [permission_item["permission_name"], permission_item.get("description", "")]
            scored = []
            for sent in split_sentences(policy_text):
                lower = sent.lower()
                score = sum(1 for kw in keywords if kw and kw.lower() in lower)
                if score > 0:
                    scored.append((score, sent))
            scored.sort(key=lambda x: (-x[0], len(x[1])))
            evidence.extend([s for _, s in scored[:max_items]])
        focus_types = set(permission_item.get("expected_capabilities", [])) | {permission_item.get("category", "")}
        for cap in capability_blocks:
            if cap.get("type") in focus_types:
                evidence.append(cap.get("evidence", ""))
        return dedup_keep_order(normalize_space(x) for x in evidence if normalize_space(x))[:max_items]

    def rule_prejudge(self, permission_item: Dict[str, Any], evidence_list: List[str]) -> Optional[Dict[str, Any]]:
        if not self.exp.use_rule_prejudge:
            return None
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
                return {"status": "已明确告知", "reason": f"存在与 {category} 直接对应的原文证据。", "evidence": evidence_list[:3], "final_confidence": "高"}
            return None
        if hit_count >= 2:
            return {"status": "已明确告知", "reason": f"存在与 {category} 直接对应的功能或数据处理场景。", "evidence": evidence_list[:3], "final_confidence": "高"}
        if hit_count >= 1:
            return {"status": "间接告知", "reason": f"存在与 {category} 相关描述，但未逐字对应到具体权限。", "evidence": evidence_list[:3], "final_confidence": "中"}
        return None

    def adjust_status(self, status: str, permission_item: Dict[str, Any]) -> str:
        if not self.exp.use_conservative_post_adjust:
            return status
        if permission_item.get("initial_confidence") != "低":
            return status
        return "间接告知" if status == "已明确告知" else status

    def llm_judge(self, permission_item: Dict[str, Any], evidence_list: List[str], capability_blocks: List[Dict[str, str]]) -> Dict[str, Any]:
        focused_caps = [cap for cap in capability_blocks if cap.get("type") in set(permission_item.get("expected_capabilities", [])) | {permission_item.get("category", "")}] [:10]
        if self.exp.use_prompt_constraint:
            prompt = f"""
你是一名移动应用隐私合规分析专家。请根据给定权限信息、原文证据句和能力块，判断该权限在隐私政策中是否得到说明。
输出 JSON：{{"status": "已明确告知/间接告知/未告知", "reason": "...", "evidence": ["..."], "final_confidence": "高/中/低"}}
权限信息：
{json.dumps(permission_item, ensure_ascii=False, indent=2)}
证据句：
{json.dumps(evidence_list, ensure_ascii=False, indent=2)}
相关能力块：
{json.dumps(focused_caps, ensure_ascii=False, indent=2)}
""".strip()
        else:
            prompt = f"""
请结合下面的权限信息、证据句和能力块，判断该权限在隐私政策中的披露情况，并给出理由。
尽量返回 JSON，字段包含 status、reason、evidence、final_confidence。
权限信息：
{json.dumps(permission_item, ensure_ascii=False, indent=2)}
证据句：
{json.dumps(evidence_list, ensure_ascii=False, indent=2)}
相关能力块：
{json.dumps(focused_caps, ensure_ascii=False, indent=2)}
""".strip()
        obj = extract_json_payload(self.llm.chat(prompt, temperature=0.1)) or {}
        status = obj.get("status", "未告知")
        status = self.adjust_status(status, permission_item)
        return {
            "status": status,
            "reason": normalize_space(str(obj.get("reason", "模型未给出充分说明"))),
            "evidence": normalize_evidence_list(obj.get("evidence"))[:3] or evidence_list[:3],
            "final_confidence": obj.get("final_confidence", "中"),
        }

    def analyze_one_app(self, app_name: str) -> Optional[Path]:
        policy_text = safe_read_text(self.policy_dir / f"{app_name}.txt")
        if not policy_text:
            print(f"[WARN] 未找到隐私政策: {app_name}")
            return None
        permissions = self.load_permissions(app_name)
        if not permissions:
            print(f"[WARN] 未找到权限文件: {app_name}")
            return None
        self.ctx.save_manifest(app_name)
        capability_blocks = self.extract_capabilities(policy_text)
        results: List[Dict[str, Any]] = []
        for item in permissions:
            evidence = self.retrieve_relevant_evidence(policy_text, item, capability_blocks)
            pre = self.rule_prejudge(item, evidence)
            if pre is not None:
                judge = pre
            else:
                judge = self.llm_judge(item, evidence, capability_blocks)
            results.append({
                "permission_name": item["permission_name"],
                "category": item.get("category", "其他"),
                "status": judge["status"],
                "reason": judge["reason"],
                "evidence": judge["evidence"],
                "final_confidence": judge["final_confidence"],
            })
        report = self.render_report(app_name, capability_blocks, results)
        out = self.ctx.app_output_dir(app_name) / "隐私政策和权限分析结果.md"
        safe_write_text(out, report)
        return out

    def render_report(self, app_name: str, capability_blocks: List[Dict[str, str]], results: List[Dict[str, Any]]) -> str:
        cnt = Counter(x["status"] for x in results)
        lines = [
            "# 隐私政策和 APK 权限一致性分析报告（消融实验版）",
            f"- 应用名称：{app_name}",
            f"- 实验编号：{self.exp.name}",
            f"- 实验说明：{self.exp.description}",
            "",
            "## 一、能力块抽取摘要",
            f"- 共抽取能力块 {len(capability_blocks)} 个。",
        ]
        for cap in capability_blocks[:12]:
            lines.append(f"- [{cap['type']}] {shorten(cap['evidence'], 120)}")
        lines.append("")
        lines.extend([
            "## 二、权限判定统计",
            f"- 已明确告知：{cnt.get('已明确告知', 0)} 个",
            f"- 间接告知：{cnt.get('间接告知', 0)} 个",
            f"- 未告知：{cnt.get('未告知', 0)} 个",
            "",
            "## 三、权限明细",
            "| 权限名称 | 类别 | 判定 | 原因 | 置信度 |",
            "| --- | --- | --- | --- | --- |",
        ])
        for r in results:
            lines.append(f"| {r['permission_name']} | {r['category']} | {r['status']} | {shorten(r['reason'], 80)} | {r['final_confidence']} |")
        lines.append("")
        lines.append("## 四、重点证据")
        for r in results:
            if r["status"] != "未告知":
                lines.append(f"### {r['permission_name']} - {r['status']}")
                lines.append(f"- 说明：{r['reason']}")
                for ev in r["evidence"][:3]:
                    lines.append(f"- 证据：{shorten(ev, 180)}")
                lines.append("")
        return "\n".join(lines).strip() + "\n"


def discover_apps(base_dir: Path) -> List[str]:
    policy_dir = Path(base_dir) / "dataset" / "隐私政策"
    return sorted(p.stem for p in policy_dir.glob("*.txt"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=".")
    ap.add_argument("--experiment", default="baseline")
    ap.add_argument("--app", default="")
    args = ap.parse_args()
    analyzer = Analyzer(Path(args.base_dir), get_experiment(args.experiment))
    apps = [args.app] if args.app.strip() else discover_apps(args.base_dir)
    for app in apps:
        print(f"[RUN] 隐私政策和权限分析: {app}")
        analyzer.analyze_one_app(app)


if __name__ == "__main__":
    main()
