# -*- coding: utf-8 -*-
import json
import time
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import faiss
import requests
from openai import OpenAI

# =========================
# 基础配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)  # 保持相对路径稳定，避免 faiss 在 Windows 中文绝对路径下抽风

SILICONFLOW_API_KEY = "sk-owpvsumadfmgklabdqoyyesnagsjzofcbrhfgowzoxuuifth"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = "BAAI/bge-m3"

STORE_DIR = "rag_store"   # 故意保留相对路径
INTRO_DIR = Path("dataset") / "应用简介"
POLICY_DIR = Path("dataset") / "隐私政策"
OUTPUT_DIR = Path("分类")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_TOPK = 5
SEPARATION_THRESHOLD = 0.18
ALWAYS_ALLOW_OUTSIDE = True
RETRIEVE_N = 140
TOPK_PROFILE_RAW = 40
TOPK_PROFILE_FINAL = 14
TOPK_METHOD_RAW = 18
TOPK_METHOD_FINAL = 10
EVIDENCE_MAX_CHARS_A = 5200
EVIDENCE_MAX_CHARS_B = 2600
EVIDENCE_PER_HIT_CHARS = 420
EMBED_CACHE_DIR = Path(".embed_cache")

MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 1.5
REQUEST_INTERVAL = 1.5
APP_INTERVAL = 3.0
INCLUDE_RAG_DEBUG = True

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


# =========================
# 通用重试包装
# =========================
def with_retries(fn, *args, **kwargs):
    last_err = None
    for i in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            wait_time = BASE_BACKOFF_SECONDS * (2 ** i)
            print(f"⚠️ 调用失败，第 {i + 1}/{MAX_RETRIES} 次重试，等待 {wait_time:.1f}s... 原因：{e}")
            time.sleep(wait_time)
    raise last_err


# =========================
# RAG 存储
# =========================
class RagStore:
    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.faiss_index_path = self.store_dir / "faiss.index"
        self.idmap_path = self.store_dir / "idmap.json"
        self.docs_path = self.store_dir / "docs.jsonl"

        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"Missing: {self.faiss_index_path}")
        if not self.idmap_path.exists():
            raise FileNotFoundError(f"Missing: {self.idmap_path}")
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Missing: {self.docs_path}")

        self.index = faiss.read_index(str(self.faiss_index_path))
        self.idmap = self._load_idmap(self.idmap_path)
        self.docs = self._load_docs(self.docs_path)

    @staticmethod
    def _load_idmap(path: Path) -> Dict[int, str]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}

    @staticmethod
    def _load_docs(path: Path) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = obj.get("chunk_id")
                if cid:
                    out[cid] = obj
        return out


# =========================
# Embedding
# =========================
class Embedder:
    def __init__(self, api_key: str, model: str, cache_dir: Path):
        self.api_key = api_key
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.url = "https://api.siliconflow.cn/v1/embeddings"

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256((self.model + "\n" + text).encode("utf-8")).hexdigest()

    def embed(self, text: str) -> np.ndarray:
        key = self._cache_key(text)
        fp = self.cache_dir / f"{key}.npy"
        if fp.exists():
            vec = np.load(fp).astype("float32")[None, :]
            faiss.normalize_L2(vec)
            return vec

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": [text],
            "encoding_format": "float"
        }

        retry = 0
        while retry < MAX_RETRIES:
            try:
                r = requests.post(self.url, json=payload, headers=headers, timeout=60)

                if r.status_code == 200:
                    data = r.json()
                    vec = np.array(data["data"][0]["embedding"], dtype="float32")
                    np.save(fp, vec)
                    vec = vec[None, :]
                    faiss.normalize_L2(vec)
                    time.sleep(REQUEST_INTERVAL)
                    return vec

                if r.status_code == 429:
                    wait_time = 10 * (retry + 1)
                    print(f"⚠️ Embedding API 限流 (429)，等待 {wait_time}s 后重试...")
                    time.sleep(wait_time)
                    retry += 1
                    continue

                raise RuntimeError(f"Embedding HTTP {r.status_code}: {r.text}")

            except Exception as e:
                retry += 1
                wait_time = 5 * retry
                print(f"⚠️ Embedding 调用异常，第 {retry}/{MAX_RETRIES} 次重试，等待 {wait_time}s... 原因：{e}")
                time.sleep(wait_time)

        raise RuntimeError("Embedding 请求失败，已达到最大重试次数")


# =========================
# 检索
# =========================
def preferred_types(intent: str) -> Optional[List[str]]:
    intent = (intent or "").strip().lower()
    if intent in ("profile", "appendixa", "category"):
        return ["appendixA_profile", "appendixA_table_item", "text"]
    if intent in ("definition", "clause", "method"):
        return ["text", "table_note", "table_footnote"]
    return None


def retrieve(store: RagStore,
             embedder: Embedder,
             query: str,
             intent: str,
             retrieve_n: int,
             topk: int) -> List[Dict[str, Any]]:
    vq = embedder.embed(query)
    scores, ids = store.index.search(vq, retrieve_n)

    pref = preferred_types(intent)
    results: List[Dict[str, Any]] = []
    seen: set = set()

    def push(internal_id: int, score: float) -> None:
        cid = store.idmap.get(internal_id)
        if not cid or cid in seen:
            return
        doc = store.docs.get(cid)
        if not doc:
            return
        meta = doc.get("meta", {}) or {}
        t = meta.get("type", "unknown")
        results.append({
            "chunk_id": cid,
            "score": float(score),
            "type": t,
            "meta": meta,
            "text": doc.get("text", "")
        })
        seen.add(cid)

    for internal_id, score in zip(ids[0], scores[0]):
        if internal_id == -1:
            continue
        internal_id = int(internal_id)
        cid = store.idmap.get(internal_id)
        if not cid:
            continue
        doc = store.docs.get(cid)
        if not doc:
            continue
        meta = doc.get("meta", {}) or {}
        t = meta.get("type", "unknown")
        if pref is not None and t not in pref:
            continue
        push(internal_id, score)
        if len(results) >= topk:
            return results

    for internal_id, score in zip(ids[0], scores[0]):
        if internal_id == -1:
            continue
        push(int(internal_id), float(score))
        if len(results) >= topk:
            break

    return results


# =========================
# 检索结果处理
# =========================
def _norm_key(text: str) -> str:
    t = "".join(text.split())
    if len(t) > 240:
        t = t[:240]
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def diversify_hits(hits: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if not hits:
        return []
    hits_sorted = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)

    chosen: List[Dict[str, Any]] = []
    seen_text = set()
    seen_src = set()

    for h in hits_sorted:
        meta = h.get("meta", {}) or {}
        src = meta.get("section_path") or meta.get("section_title") or ""
        appendix = meta.get("appendix_letter") or ""
        table_id = meta.get("table_id") or ""
        src_key = f"{src}|{appendix}|{table_id}"

        text = (h.get("text") or "").strip()
        tk = _norm_key(text)

        if tk in seen_text:
            continue
        if src_key in seen_src and len(chosen) < max(3, k // 2):
            continue

        chosen.append(h)
        seen_text.add(tk)
        seen_src.add(src_key)
        if len(chosen) >= k:
            return chosen

    if len(chosen) < k:
        for h in hits_sorted:
            if h in chosen:
                continue
            text = (h.get("text") or "").strip()
            tk = _norm_key(text)
            if tk in seen_text:
                continue
            chosen.append(h)
            seen_text.add(tk)
            if len(chosen) >= k:
                break

    return chosen


def format_hits(hits: List[Dict[str, Any]], max_total_chars: int, per_hit_chars: int) -> str:
    blocks: List[str] = []
    used = 0
    for h in hits:
        meta = h.get("meta", {}) or {}
        sec = meta.get("section_path") or meta.get("section_title") or ""
        appendix = meta.get("appendix_letter") or ""
        table_id = meta.get("table_id") or ""
        src = " | ".join([
            x for x in [
                sec,
                f"appendix={appendix}" if appendix else "",
                f"table={table_id}" if table_id else ""
            ] if x
        ])

        text = (h.get("text") or "").strip().replace("\n", " ")
        if len(text) > per_hit_chars:
            text = text[:per_hit_chars] + "…"

        line = f"[{h.get('type','?')} score={h.get('score',0):.4f} chunk_id={h.get('chunk_id','')}] {src}\n{text}"
        if used + len(line) > max_total_chars:
            break
        blocks.append(line)
        used += len(line)
    return "\n\n".join(blocks).strip()


def filter_hits_by_standard_prefix(hits: List[Dict[str, Any]], prefix: str = "GB/T 41391-2022::") -> List[Dict[str, Any]]:
    out = []
    for h in hits:
        cid = h.get("chunk_id", "")
        if isinstance(cid, str) and cid.startswith(prefix):
            out.append(h)
    return out


# =========================
# 文件读取
# =========================
def load_app_descriptions(intro_dir: Path) -> Dict[str, str]:
    folder = Path(intro_dir)
    if not folder.exists():
        raise FileNotFoundError(f"找不到目录：{folder}")
    txts = sorted(folder.glob("*.txt"))
    if not txts:
        raise FileNotFoundError(f"目录下没有 .txt：{folder}")

    out: Dict[str, str] = {}
    for fp in txts:
        content = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if content:
            out[fp.stem] = content

    if not out:
        raise ValueError(f"所有 txt 都是空的：{folder}")
    return out


def load_policy_text(app_name: str) -> str:
    fp = POLICY_DIR / f"{app_name}.txt"
    if not fp.exists():
        return ""
    return fp.read_text(encoding="utf-8", errors="ignore").strip()


# =========================
# Chat / JSON
# =========================
def safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e + 1])
        raise


def call_chat(api_key: str, model: str, messages: List[Dict[str, str]],
              temperature: float = 0.0, max_tokens: int = 900) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

    retry = 0
    while retry < MAX_RETRIES:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                time.sleep(REQUEST_INTERVAL)
                return safe_parse_json(content)

            print("⚠️ 模型返回为空，准备重试...")
            retry += 1
            time.sleep(REQUEST_INTERVAL)

        except Exception as e:
            msg = str(e)
            retry += 1
            if "429" in msg or "Too Many Requests" in msg:
                wait_time = 10 * retry
                print(f"⚠️ Chat API 限流 (429)，等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
            else:
                wait_time = 5 * retry
                print(f"⚠️ Chat 调用异常，第 {retry}/{MAX_RETRIES} 次重试，等待 {wait_time}s... 原因：{e}")
                time.sleep(wait_time)

    raise RuntimeError("Chat 请求失败，已达到最大重试次数")


# =========================
# Prompt：阶段一，先判断是否属于39类
# =========================
def build_stage1_messages(app_name: str, intro: str) -> List[Dict[str, str]]:
    cats = "、".join(GBT_39_CATEGORIES)

    system = (
        "你是应用分类初筛助手。你的任务是：先根据应用简介判断该应用是否属于GB/T 41391-2022中的39类应用。"
        "如果可能属于，再给出最可能的Top-5候选类别。"
        "输出只能是JSON，且只输出JSON。"
    )

    user = f"""
【应用名】
{app_name}

【39类候选集合】
{cats}

【应用简介】
{intro[:4200] + ("…" if len(intro) > 4200 else "")}

【输出JSON（只输出JSON）】
{{
  "app_name": "{app_name}",
  "likely_in_39": true/false,
  "judgement_reason": "一句到两句判断理由",
  "candidate_scores": [
    {{"category": "某类", "score": 0.0-1.0, "why": "一句理由"}},
    ...
  ]
}}

【硬约束】
- 必须先判断 likely_in_39
- 如果 likely_in_39=true，candidate_scores 至少给出Top-5
- 如果 likely_in_39=false，candidate_scores 仍然给出Top-5，但可以明确说明只是相近候选而非最终归类
- category 必须严格匹配候选集合中的某一项
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_stage1(obj: Dict[str, Any]) -> Optional[str]:
    if not isinstance(obj, dict):
        return "not_a_dict"

    if "likely_in_39" not in obj:
        return "missing_likely_in_39"

    cs = obj.get("candidate_scores")
    if not isinstance(cs, list) or len(cs) < STAGE1_TOPK:
        return "missing_candidate_scores"

    for it in cs[:STAGE1_TOPK]:
        if not isinstance(it, dict):
            return "bad_candidate_item"
        if "category" not in it or "score" not in it:
            return "candidate_fields_missing"
        cat = it["category"]
        if cat not in GBT_39_CATEGORIES:
            return "invalid_candidate_category"
        try:
            float(it["score"])
        except Exception:
            return "bad_score"

    return None


def get_top_candidates_and_separation(candidate_scores: List[Dict[str, Any]]) -> Tuple[List[str], float]:
    cs = candidate_scores[:]

    def score_of(x):
        try:
            return float(x.get("score", 0.0))
        except Exception:
            return 0.0

    cs = sorted(cs, key=score_of, reverse=True)

    top = []
    for it in cs:
        cat = it.get("category")
        if cat and cat not in top:
            top.append(cat)
        if len(top) >= STAGE1_TOPK:
            break

    s1 = score_of(cs[0]) if len(cs) >= 1 else 0.0
    s2 = score_of(cs[1]) if len(cs) >= 2 else 0.0
    return top, (s1 - s2)


# =========================
# Prompt：属于39类时，判断具体类别
# =========================
def build_category_driven_queries(app_intro: str, top_candidates: List[str]) -> List[str]:
    queries = []
    for cat in top_candidates[:3]:
        queries.append(f"{cat} 附录A 画像 定义 基本业务功能 适用场景\n应用简介：{app_intro}")
    return queries


def build_stage2_in39_messages(
    app_name: str,
    intro: str,
    rag_a: str,
    rag_b: str,
    allowed_categories: List[str],
    separation: float
) -> List[Dict[str, str]]:
    cats = "、".join(allowed_categories)

    system = (
        "你是国标合规分类助手。这个应用已经在前一步被判定为可能属于39类。"
        "你必须结合RAG证据，在候选集合中确定其最合理的39类类别。"
        "输出只能是JSON，且只输出JSON。"
    )

    user = f"""
【应用名】
{app_name}

【候选集合（你只能从这里选最终类别）】
{cats}

【候选分离度提示】
Top1-Top2 = {separation:.3f}

【应用简介】
{intro[:3500] + ("…" if len(intro) > 3500 else "")}

【RAG证据A：与该简介/候选类别对齐的国标片段】
{rag_a}

【RAG证据B：国标中的判定方法/功能划分条款】
{rag_b}

【输出JSON（只输出JSON）】
{{
  "app_name": "{app_name}",
  "is_in_39": true,
  "gbt_39_category": "必须从候选集合中选一个39类",
  "alt_category": "",
  "confidence": 0.0-1.0,
  "reasons": ["3-6条短句"],
  "evidence": [{{"quote": "从证据A/B复制短句(<=40字)", "chunk_id": "对应chunk_id"}}]
}}

【硬约束】
- is_in_39 必须为 true
- gbt_39_category 必须从候选集合中选一个39类
- alt_category 必须为空字符串
- evidence 至少2条，且 chunk_id 必须来自证据A/B中出现过的 chunk_id
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# =========================
# Prompt：不属于39类时，用隐私政策判断非39类类别
# =========================
def build_stage2_non39_messages(app_name: str, intro: str, policy_text: str) -> List[Dict[str, str]]:
    system = (
        "你是应用分类助手。该应用已经被判定为不属于GB/T 41391-2022中的39类应用。"
        "现在请结合应用简介与隐私政策，判断这个应用更像什么类别。"
        "输出只能是JSON，且只输出JSON。"
    )

    policy_excerpt = policy_text[:6000] + ("…" if len(policy_text) > 6000 else "")

    user = f"""
【应用名】
{app_name}

【应用简介】
{intro[:3500] + ("…" if len(intro) > 3500 else "")}

【隐私政策】
{policy_excerpt if policy_excerpt else "（未提供有效隐私政策文本）"}

【任务】
请判断该应用不属于39类后，更接近什么“非39类应用类别”。

【输出JSON（只输出JSON）】
{{
  "app_name": "{app_name}",
  "is_in_39": false,
  "gbt_39_category": "不属于39类",
  "alt_category": "给出一个尽量准确的非39类类别，如AI创作、AI问答、效率办公、系统工具、开发工具、内容生成等",
  "confidence": 0.0-1.0,
  "reasons": ["3-6条短句，说明为什么不属于39类，以及为什么更像该类别"],
  "evidence": [
    {{"source": "intro/policy", "quote": "引用应用简介或隐私政策中的短句(<=50字)"}}
  ]
}}

【硬约束】
- is_in_39 必须为 false
- gbt_39_category 必须为 “不属于39类”
- alt_category 必须非空
- evidence 至少2条
- 判断必须优先依据隐私政策中体现的业务功能、信息处理对象和服务场景
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# =========================
# 校验
# =========================
def extract_chunk_ids_from_context(ctx: str) -> set:
    out = set()
    marker = "chunk_id="
    i = 0
    while True:
        j = ctx.find(marker, i)
        if j == -1:
            break
        k = j + len(marker)
        end = k
        while end < len(ctx) and ctx[end] not in ["]", " ", "\n", "\r", "\t"]:
            end += 1
        out.add(ctx[k:end])
        i = end
    return out


def safe_json_fix_call(api_key: str, messages: List[Dict[str, str]], reason: str, max_tokens: int) -> Dict[str, Any]:
    fix = {
        "role": "user",
        "content": f"你刚才输出不符合硬约束，错误原因={reason}。请严格重写，只输出JSON。"
    }
    return with_retries(call_chat, api_key, CHAT_MODEL, messages + [fix], 0.0, max_tokens)


def validate_stage2_in39(obj: Dict[str, Any], allowed_categories: List[str]) -> Optional[str]:
    if not isinstance(obj, dict):
        return "not_a_dict"

    cat = obj.get("gbt_39_category")
    is_in_39 = obj.get("is_in_39")
    alt = obj.get("alt_category")

    if cat is None or is_in_39 is None or alt is None:
        return "missing_fields"

    if cat not in allowed_categories:
        return "category_not_in_allowed_set"

    if is_in_39 is not True:
        return "is_in_39_should_be_true"

    if isinstance(alt, str) and alt.strip():
        return "alt_category_should_be_empty_when_in_39"

    ev = obj.get("evidence", [])
    if not isinstance(ev, list) or len(ev) < 2:
        return "insufficient_evidence"

    for it in ev:
        if not isinstance(it, dict) or "chunk_id" not in it or "quote" not in it:
            return "bad_evidence_item"

    return None


def validate_stage2_non39(obj: Dict[str, Any]) -> Optional[str]:
    if not isinstance(obj, dict):
        return "not_a_dict"

    if obj.get("is_in_39") is not False:
        return "is_in_39_should_be_false"

    if obj.get("gbt_39_category") != "不属于39类":
        return "gbt_39_category_should_be_outside"

    alt = obj.get("alt_category", "")
    if not isinstance(alt, str) or not alt.strip():
        return "alt_category_required"

    ev = obj.get("evidence", [])
    if not isinstance(ev, list) or len(ev) < 2:
        return "insufficient_evidence"

    return None


def enforce_evidence_chunk_ids(out: Dict[str, Any], allowed: set) -> Optional[str]:
    ev = out.get("evidence", [])
    if not isinstance(ev, list):
        return "evidence_not_list"
    for it in ev:
        cid = it.get("chunk_id")
        if cid not in allowed:
            return "evidence_chunk_id_not_in_context"
    return None


# =========================
# 单应用处理
# =========================
def process_one_app(
    app_name: str,
    intro: str,
    store: RagStore,
    embedder: Embedder,
    rag_b: str,
    hits_method: List[Dict[str, Any]]
) -> Dict[str, Any]:
    # ---------- 阶段一：先判断是否属于39类 ----------
    msgs1 = build_stage1_messages(app_name, intro)
    out1 = with_retries(call_chat, SILICONFLOW_API_KEY, CHAT_MODEL, msgs1, 0.0, 700)

    err1 = validate_stage1(out1)
    if err1:
        out1 = safe_json_fix_call(SILICONFLOW_API_KEY, msgs1, err1, 700)

    candidate_scores = out1.get("candidate_scores", [])
    likely_in_39 = bool(out1.get("likely_in_39", False))
    judgement_reason = out1.get("judgement_reason", "")

    top_candidates, separation = get_top_candidates_and_separation(candidate_scores)
    boundary_is_ambiguous = (separation < SEPARATION_THRESHOLD)

    # ---------- 如果属于39类：判断具体哪一类 ----------
    if likely_in_39:
        allowed = top_candidates[:]
        if ALWAYS_ALLOW_OUTSIDE and "不属于39类" not in allowed:
            allowed.append("不属于39类")

        q_base = f"根据应用简介判断所属服务类型（39类），返回最相关的附录A条目/画像/定义：\n{intro}"
        hits_base_raw = retrieve(
            store, embedder, q_base, intent="profile",
            retrieve_n=RETRIEVE_N, topk=TOPK_PROFILE_RAW
        )

        hits_cat_all: List[Dict[str, Any]] = []
        cat_queries = build_category_driven_queries(intro, top_candidates)
        for q in cat_queries:
            hs = retrieve(
                store, embedder, q, intent="profile",
                retrieve_n=RETRIEVE_N, topk=12
            )
            hits_cat_all.extend(hs)

        hits_all = hits_base_raw + hits_cat_all
        hits_all = filter_hits_by_standard_prefix(hits_all, "GB/T 41391-2022::")
        hits_app = diversify_hits(hits_all, k=min(TOPK_PROFILE_FINAL, len(hits_all)))

        rag_a = format_hits(
            hits_app,
            max_total_chars=EVIDENCE_MAX_CHARS_A,
            per_hit_chars=EVIDENCE_PER_HIT_CHARS
        )
        if not rag_a:
            rag_a = "(未检索到与该简介相关的国标片段)"

        # 这里强制走39类细分类
        allowed_in39_only = [x for x in allowed if x != "不属于39类"]
        msgs2 = build_stage2_in39_messages(app_name, intro, rag_a, rag_b, allowed_in39_only, separation)
        out2 = with_retries(call_chat, SILICONFLOW_API_KEY, CHAT_MODEL, msgs2, 0.0, 980)

        err2 = validate_stage2_in39(out2, allowed_in39_only)
        if err2:
            out2 = safe_json_fix_call(SILICONFLOW_API_KEY, msgs2, err2, 980)

        allowed_chunk_ids = extract_chunk_ids_from_context(rag_a) | extract_chunk_ids_from_context(rag_b)
        err3 = enforce_evidence_chunk_ids(out2, allowed_chunk_ids)
        if err3:
            out2 = safe_json_fix_call(SILICONFLOW_API_KEY, msgs2, err3, 980)

        final_obj = {
            **out2,
            "stage1_likely_in_39": likely_in_39,
            "stage1_judgement_reason": judgement_reason,
            "candidate_scores": candidate_scores[:STAGE1_TOPK],
            "separation": float(separation),
            "boundary_is_ambiguous": bool(boundary_is_ambiguous),
        }

        if INCLUDE_RAG_DEBUG:
            final_obj["rag_debug"] = {
                "top_candidates_used": allowed_in39_only,
                "rag_a_hits": [
                    {"chunk_id": h["chunk_id"], "score": h["score"], "type": h["type"]}
                    for h in hits_app
                ],
                "rag_b_hits": [
                    {"chunk_id": h["chunk_id"], "score": h["score"], "type": h["type"]}
                    for h in hits_method
                ],
                "category_driven_queries": cat_queries,
            }

        return final_obj

    # ---------- 如果不属于39类：根据隐私政策判断非39类类别 ----------
    policy_text = load_policy_text(app_name)
    msgs_non39 = build_stage2_non39_messages(app_name, intro, policy_text)
    out_non39 = with_retries(call_chat, SILICONFLOW_API_KEY, CHAT_MODEL, msgs_non39, 0.0, 900)

    err_non39 = validate_stage2_non39(out_non39)
    if err_non39:
        out_non39 = safe_json_fix_call(SILICONFLOW_API_KEY, msgs_non39, err_non39, 900)

    final_obj = {
        **out_non39,
        "stage1_likely_in_39": likely_in_39,
        "stage1_judgement_reason": judgement_reason,
        "candidate_scores": candidate_scores[:STAGE1_TOPK],
        "separation": float(separation),
        "boundary_is_ambiguous": bool(boundary_is_ambiguous),
    }

    return final_obj


# =========================
# 主流程
# =========================
def main():
    if not SILICONFLOW_API_KEY or "把你的key" in SILICONFLOW_API_KEY:
        raise ValueError("请在代码开头把 SILICONFLOW_API_KEY 填成你的真实 key")

    if not INTRO_DIR.exists():
        raise FileNotFoundError(f"未找到应用简介目录：{INTRO_DIR}")

    store = RagStore(STORE_DIR)
    embedder = Embedder(
        api_key=SILICONFLOW_API_KEY,
        model=EMBED_MODEL,
        cache_dir=EMBED_CACHE_DIR
    )

    method_query = "应用功能划分 基本业务功能 扩展业务功能 其他服务类型 判定 方法 条款"
    hits_method_raw = retrieve(
        store, embedder, method_query, intent="definition",
        retrieve_n=RETRIEVE_N, topk=TOPK_METHOD_RAW
    )
    hits_method_raw = filter_hits_by_standard_prefix(hits_method_raw, "GB/T 41391-2022::")
    hits_method = diversify_hits(hits_method_raw, k=min(TOPK_METHOD_FINAL, len(hits_method_raw)))
    rag_b = format_hits(
        hits_method,
        max_total_chars=EVIDENCE_MAX_CHARS_B,
        per_hit_chars=EVIDENCE_PER_HIT_CHARS
    )
    if not rag_b:
        rag_b = "(未检索到判定方法条款)"

    app_descs = load_app_descriptions(INTRO_DIR)

    print(f"📂 应用简介目录：{INTRO_DIR}")
    print(f"📂 隐私政策目录：{POLICY_DIR}")
    print(f"📂 输出目录：{OUTPUT_DIR}")
    print(f"📄 共检测到应用简介：{len(app_descs)} 个")
    print("\n🚀 开始分类...\n")

    for app_name, intro in app_descs.items():
        print(f"🔍 正在分类：{app_name}")
        try:
            final_obj = process_one_app(
                app_name=app_name,
                intro=intro,
                store=store,
                embedder=embedder,
                rag_b=rag_b,
                hits_method=hits_method
            )

            out_path = OUTPUT_DIR / f"{app_name}.json"
            out_path.write_text(
                json.dumps(final_obj, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"✅ 已写入：{out_path}")

        except Exception as e:
            print(f"❌ 分类失败：{app_name}，原因：{e}")

        print(f"⏸️ 应用级等待 {APP_INTERVAL}s，避免触发限流...")
        time.sleep(APP_INTERVAL)

    print("\n🎯 所有应用分类完成")


if __name__ == "__main__":
    main()