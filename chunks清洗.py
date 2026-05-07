# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

IN_PATH = Path("chunks.jsonl")
OUT_PATH = Path("chunks_final.jsonl")
MAX_TEXT_BODY = 3500
MAX_PACK_BODY = 9000
MAX_STRONG_SENT = 12
MAX_TABLE_CTX = 1200
MAX_REF_TEXT = 1600
EMIT_E_CATEGORY = True
WS = re.compile(r"[ \t]+")
SENT_SPLIT = re.compile(r"(?<=[。！？；\n])")
STRONG_PAT = re.compile(
    r"(应当|不得|必须|禁止|不应|仅可|仅能|仅用于|不得以|应在|应按|应符合|应提供|应告知|应取得|应删除|应匿名化|宜|可以)"
)
PERM_CONST_PAT = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")  # READ_CALENDAR, ACCESS_FINE_LOCATION...
ANDROID_PERM_FULL = re.compile(r"\bandroid\.permission\.[A-Z0-9_]{2,}\b")

def norm(s: str) -> str:
    return WS.sub(" ", (s or "").strip())

def clamp(s: str, n: int) -> str:
    s = (s or "").strip()
    return s[:n] if len(s) > n else s

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def dump_jsonl(path: Path, records: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------- text helpers ----------------
def guess_text_tag(text: str, meta: Dict[str, Any]) -> str:
    title = (meta.get("section_title") or "") + "\n" + (meta.get("section_path") or "")
    t = (title + "\n" + (text or "")).strip()
    if "定义" in t or "术语" in t or "definition" in t.lower():
        return "定义"
    if any(k in t for k in ("范围", "适用", "必要个人信息", "个人信息范围")):
        return "范围"
    if STRONG_PAT.search(t):
        return "要求"
    return "说明"

def extract_strong_sentences(text: str, k: int = MAX_STRONG_SENT) -> List[str]:
    if not text:
        return []
    sents = [x.strip() for x in SENT_SPLIT.split(text) if x.strip()]
    strong = []
    for s in sents:
        if STRONG_PAT.search(s):
            strong.append(s)
        if len(strong) >= k:
            break
    return strong

def build_norm_text_record(ch: Dict[str, Any]) -> Dict[str, Any]:
    meta = ch.get("meta", {}) or {}
    text = ch.get("text", "") or ""
    standard_id = meta.get("standard_id", "")
    section_path = meta.get("section_path") or ""
    section_title = meta.get("section_title") or ""
    appendix = meta.get("appendix_letter")
    tag = guess_text_tag(text, meta)
    strong = extract_strong_sentences(text)
    header = [f"[标准] {standard_id}".strip()]
    if section_path:
        header.append(f"[位置] {section_path}")
    elif section_title:
        header.append(f"[位置] {section_title}")
    if appendix:
        header.append(f"[附录] {appendix}")
    header.append(f"[类型] {tag}")
    body = clamp(text, MAX_TEXT_BODY)
    strong_block = ""
    if strong:
        strong_block = "[强约束句]\n- " + "\n- ".join(strong)
    emb = "\n".join([*header, "[正文]", body, strong_block]).strip()
    return {
        "chunk_id": f"{ch.get('chunk_id')}::norm",
        "text": ch.get("text", ""),
        "meta": {**meta, "type": "norm-text", "source_chunk_id": ch.get("chunk_id")},
        "embedding_text": emb
    }

def parse_md_or_pipe_table(raw: str) -> Optional[Tuple[List[str], List[List[str]]]]:
    if not raw:
        return None
    lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
    for i in range(len(lines) - 2):
        if "|" not in lines[i] or "|" not in lines[i + 1]:
            continue
        delim = lines[i + 1].strip()
        if not (re.search(r"[-:]{3,}", delim) and "|" in delim):
            continue
        headers = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows = []
        j = i + 2
        while j < len(lines) and "|" in lines[j]:
            row = [c.strip() for c in lines[j].strip().strip("|").split("|")]
            rows.append(row)
            j += 1
        out_rows = []
        for r in rows:
            if len(r) < len(headers):
                r = r + [""] * (len(headers) - len(r))
            elif len(r) > len(headers):
                r = r[:len(headers)]
            out_rows.append(r)
        return headers, out_rows
    for i in range(min(10, len(lines))):
        if lines[i].count("|") >= 4 and any(k in lines[i] for k in ("权限名", "服务类型", "类别", "描述")):
            headers = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            rows = []
            for j in range(i + 1, len(lines)):
                if lines[j].count("|") >= 4:
                    row = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                    if len(row) < len(headers):
                        row += [""] * (len(headers) - len(row))
                    elif len(row) > len(headers):
                        row = row[:len(headers)]
                    rows.append(row)
                else:
                    if rows and j > i + 3:
                        break
            if rows:
                return headers, rows
    return None

def is_2022_appendix_a_business_table(meta: Dict[str, Any]) -> bool:
    title = meta.get("table_title") or ""
    appx = meta.get("appendix_letter")
    if appx == "A" and "必要个人信息" in title and ("使用要求" in title or "范围" in title):
        return True
    if "App 必要个人信息范围和使用要求" in title:
        return True
    return False

def is_2022_appendix_d_table(meta: Dict[str, Any]) -> bool:
    title = meta.get("table_title") or ""
    appx = meta.get("appendix_letter")
    if appx == "D" and ("权限" in title):
        return True
    if "可收集个人信息权限" in title:
        return True
    return False

def is_2022_appendix_e_table(meta: Dict[str, Any]) -> bool:
    appx = meta.get("appendix_letter")
    title = meta.get("table_title") or ""
    if appx == "E":
        return True
    if "相关程度较低" in title or "相关性" in title:
        return True
    return False

def is_2025_sensitive_table(meta: Dict[str, Any]) -> bool:
    title = meta.get("table_title") or ""
    std = meta.get("standard_id") or ""
    appx = meta.get("appendix_letter")
    if "2025" in std and appx == "A" and ("敏感个人信息" in title) and ("类别" in title):
        return True
    if appx == "A" and ("敏感个人信息" in title) and ("类别" in title) and ("GB/T" in std and "2025" in std):
        return True
    return False

def extract_category_name_from_section_title(section_title: str) -> Optional[str]:
    s = section_title or ""
    m = re.search(r"A\.\d+\s*([^\s]+类)", s)
    if m:
        return m.group(1).strip()
    return None

def extract_category_name_from_table_title(table_title: str) -> Optional[str]:
    t = table_title or ""
    m = re.search(r"表\s*A\.\d+\s*([^\s]+类)", t)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"([^\s]+类)\s*(App|应用)", t)
    if m2:
        return m2.group(1).strip()
    return None

def collect_referenced_text(meta: Dict[str, Any], chunk_by_id: Dict[str, Dict[str, Any]], keep_keywords: List[str]) -> str:
    ref_ids = meta.get("referenced_by_chunk_ids") or []
    parts = []
    for cid in ref_ids:
        ref = chunk_by_id.get(cid)
        if not ref:
            continue
        if safe_get(ref, "meta", "type") != "text":
            continue
        tx = ref.get("text") or ""
        if not tx.strip():
            continue
        ok = False
        if keep_keywords:
            if any(k in tx for k in keep_keywords):
                ok = True
        if not ok:
            if safe_get(ref, "meta", "appendix_letter") == meta.get("appendix_letter"):
                ok = True
        if ok:
            parts.append(tx.strip())
    return clamp("\n\n".join(parts), MAX_REF_TEXT)

def build_appendix_a_rows_from_table(
    table_ch: Dict[str, Any],
    chunk_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str], str]:
    meta = table_ch.get("meta", {}) or {}
    raw = table_ch.get("text", "") or ""
    standard_id = meta.get("standard_id", "")
    appendix = meta.get("appendix_letter") or "A"
    table_id = meta.get("table_id", "")
    table_title = meta.get("table_title", "")
    cat = extract_category_name_from_section_title(meta.get("section_title","")) or \
          extract_category_name_from_table_title(table_title)
    ctx = clamp(meta.get("table_context_text","") or "", MAX_TABLE_CTX)
    ref_text = collect_referenced_text(
        meta, chunk_by_id,
        keep_keywords=[cat or "", "必要个人信息", "使用要求", f"表{table_id}", "附录A"]
    )
    parsed = parse_md_or_pipe_table(raw)
    rows_triplets: List[Tuple[str, str, str]] = []
    if parsed:
        headers, rows = parsed
        def find_col(keys: List[str]) -> Optional[int]:
            for i, h in enumerate(headers):
                for k in keys:
                    if k in h:
                        return i
            return None
        c_service = find_col(["服务类型", "服务", "类型"])
        c_pi = find_col(["必要个人信息", "必要信息", "个人信息"])
        c_req = find_col(["使用要求", "要求", "用途"])
        for r in rows:
            service = r[c_service].strip() if c_service is not None else ""
            pi = r[c_pi].strip() if c_pi is not None else ""
            req = r[c_req].strip() if c_req is not None else ""
            if req == "" and pi and re.match(r"^(用于|应|不得|完成|仅用于)", pi):
                req = pi
                pi = ""
            rows_triplets.append((service, pi, req))
        last_service = ""
        fixed = []
        for service, pi, req in rows_triplets:
            if service:
                last_service = service
            else:
                service = last_service
            fixed.append((service, pi, req))
        rows_triplets = fixed
    else:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        for ln in lines[:80]:
            if "用于" in ln and any(k in ln for k in ("定位", "导航", "路线", "规划")):
                rows_triplets.append(("", "", ln))
    cat = cat or "未知类别"
    row_records: List[Dict[str, Any]] = []
    row_sents_for_pack: List[str] = []
    for idx, (service, pi, req) in enumerate(rows_triplets, start=1):
        service = norm(service) or "（未标注服务类型）"
        pi = norm(pi)
        req = norm(req)
        emb_lines = [
            f"[标准] {standard_id}",
            f"[附录] {appendix}",
            f"[类别] {cat}",
            f"[表] {table_id} {table_title}".strip(),
            f"[服务类型] {service}",
        ]
        if pi:
            emb_lines.append(f"[必要个人信息] {pi}")
        if req:
            emb_lines.append(f"[使用要求] {req}")
        if ctx:
            emb_lines.append("[表格上下文]")
            emb_lines.append(clamp(ctx, 500))
        if ref_text:
            emb_lines.append("[关联说明]")
            emb_lines.append(clamp(ref_text, 600))
        emb = "\n".join([x for x in emb_lines if x]).strip()
        rid = f"{standard_id}::{appendix}::{cat}::{table_id}::row{idx}"
        row_records.append({
            "chunk_id": rid,
            "text": "\n".join([service, pi, req]).strip(),
            "meta": {
                "type": "category-row",
                "standard_id": standard_id,
                "appendix_letter": appendix,
                "category": cat,
                "service_type": service,
                "table_id": table_id,
                "table_title": table_title,
                "source_table_chunk_id": table_ch.get("chunk_id"),
            },
            "embedding_text": emb
        })
        kv = f"服务类型={service}"
        if pi:
            kv += f"；必要个人信息={pi}"
        if req:
            kv += f"；使用要求={req}"
        row_sents_for_pack.append(kv)
    rows_text = "\n- " + "\n- ".join(row_sents_for_pack) if row_sents_for_pack else ""
    return row_records, cat, rows_text

def build_appendix_a_category_pack_from_text(
    standard_id: str,
    category: str,
    appendix: str,
    text_chunks: List[Dict[str, Any]],
    rows_text: str = "",
    table_refs: Optional[List[Tuple[str, str]]] = None
) -> Dict[str, Any]:
    bodies = []
    seen = set()
    for ch in text_chunks:
        t = (ch.get("text") or "").strip()
        if not t:
            continue
        key = t[:120]
        if key in seen:
            continue
        seen.add(key)
        bodies.append(t)
    body_text = clamp("\n\n".join(bodies), MAX_PACK_BODY)
    ref_part = ""
    if table_refs:
        ref_part = "\n".join([f"- {tid} {ttl}".strip() for tid, ttl in table_refs if tid or ttl]).strip()
        if ref_part:
            ref_part = "[相关表格]\n" + ref_part
    emb = "\n".join([
        f"[标准] {standard_id}",
        f"[附录] {appendix}",
        f"[类别证据包] {category}",
        "[用途] 汇总该类别App的必要个人信息范围与使用要求，用于隐私政策合规与最小必要评估。",
        "[类别相关文本]",
        body_text,
        ref_part,
        "[表格行级要求]" if rows_text else "",
        clamp(rows_text, MAX_PACK_BODY) if rows_text else "",
    ]).strip()
    return {
        "chunk_id": f"{standard_id}::{appendix}::{category}::pack",
        "text": emb,
        "meta": {
            "type": "category-pack",
            "standard_id": standard_id,
            "appendix_letter": appendix,
            "category": category,
            "table_refs": table_refs or [],
        },
        "embedding_text": emb
    }

def build_appendix_d_permission_rows(table_ch: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = table_ch.get("meta", {}) or {}
    raw = table_ch.get("text", "") or ""
    standard_id = meta.get("standard_id", "")
    appendix = meta.get("appendix_letter") or "D"
    table_id = meta.get("table_id", "")
    table_title = meta.get("table_title", "")
    ctx = clamp(meta.get("table_context_text", "") or "", 800)
    parsed = parse_md_or_pipe_table(raw)
    items = []
    if parsed:
        headers, rows = parsed
        def find_col(keys: List[str]) -> Optional[int]:
            for i, h in enumerate(headers):
                for k in keys:
                    if k in h:
                        return i
            return None
        c_group = find_col(["权限分组", "分组"])
        c_name = find_col(["权限名", "权限名称", "权限"])
        c_desc = find_col(["功能描述", "描述"])
        c_pi = find_col(["可访问的个人信息", "可访问个人信息", "个人信息"])
        c_ex = find_col(["业务功能示例", "示例", "业务示例"])
        for r in rows:
            row_text = " ".join([x for x in r if x])
            perm = ""
            if c_name is not None and c_name < len(r):
                perm = r[c_name].strip()
            m = PERM_CONST_PAT.search(perm) or PERM_CONST_PAT.search(row_text)
            if not m:
                continue
            perm_const = m.group(0)
            items.append({
                "permission": perm_const,
                "permission_group": r[c_group].strip() if c_group is not None and c_group < len(r) else "",
                "desc": r[c_desc].strip() if c_desc is not None and c_desc < len(r) else "",
                "pi": r[c_pi].strip() if c_pi is not None and c_pi < len(r) else "",
                "ex": r[c_ex].strip() if c_ex is not None and c_ex < len(r) else "",
            })
    else:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        current_group = ""
        for ln in lines:
            mg = re.match(r"^([A-Z_]{3,})\s+(.+)$", ln)
            if mg and ("权限" not in ln) and (len(mg.group(2)) <= 10):
                current_group = mg.group(1)
            m = PERM_CONST_PAT.search(ln)
            if m:
                perm_const = m.group(0)
                items.append({
                    "permission": perm_const,
                    "permission_group": current_group,
                    "desc": ln,
                    "pi": "",
                    "ex": "",
                })
        seen = set()
        dedup = []
        for it in items:
            if it["permission"] in seen:
                continue
            seen.add(it["permission"])
            dedup.append(it)
        items = dedup
    out = []
    for it in items:
        perm = it["permission"]
        full = f"android.permission.{perm}"
        emb = "\n".join([
            f"[标准] {standard_id}",
            f"[附录] {appendix}",
            f"[表] {table_id} {table_title}".strip(),
            "[用途] 本条说明该权限可能访问的个人信息范围与常见业务示例，用于判断权限必要性与披露合规。",
            f"[权限名] {perm}",
            f"[权限全名] {full}",
            (f"[权限分组] {norm(it['permission_group'])}" if it["permission_group"] else ""),
            (f"[功能描述/抽取行] {norm(it['desc'])}" if it["desc"] else ""),
            (f"[可访问个人信息] {norm(it['pi'])}" if it["pi"] else ""),
            (f"[业务示例] {norm(it['ex'])}" if it["ex"] else ""),
            ("[上下文]\n" + ctx if ctx else ""),
        ]).strip()
        out.append({
            "chunk_id": f"{standard_id}::{appendix}::{perm}::perm",
            "text": perm,
            "meta": {
                "type": "permission-row",
                "standard_id": standard_id,
                "appendix_letter": appendix,
                "table_id": table_id,
                "table_title": table_title,
                "permission": perm,
                "permission_full": full,
                "permission_group": it.get("permission_group", ""),
                "source_table_chunk_id": table_ch.get("chunk_id"),
            },
            "embedding_text": emb
        })
    return out

def build_e_rule_from_intro_texts(standard_id: str, intro_texts: List[str]) -> Dict[str, Any]:
    body = "\n\n".join([t.strip() for t in intro_texts if t.strip()])
    body = clamp(body, 2400)
    if not body:
        body = (
            "附录E给出与常见服务类型相关程度较低的安卓系统权限提示表。"
            "“×”表示该权限与该服务类型的主要功能相关程度较低；未标注“×”不等同于合理或必要，仅表示本附录未给出低相关提示。"
            "应结合业务功能与必要个人信息范围综合判断权限申请的必要性与合理性。"
        )
    emb = "\n".join([
        f"[标准] {standard_id}",
        "[附录E读表规则]",
        "附录E用于“相关程度较低”的提示，不是“禁止/允许”清单。",
        "“×”=相关程度较低；无“×”≠合理/必要，仅表示未提示。",
        "[原文/说明摘录]",
        body
    ]).strip()
    return {
        "chunk_id": f"{standard_id}::E::rule",
        "text": body,
        "meta": {"type": "e-rule", "standard_id": standard_id, "appendix_letter": "E"},
        "embedding_text": emb
    }

def extract_e_header_categories(headers: List[str]) -> List[str]:
    non_cat = {"序号", "权限分组", "权限名", "权限名称", "权限", "读写", "说明", "功能描述"}
    cats = []
    for h in headers:
        h2 = norm(h)
        if not h2 or h2 in non_cat:
            continue
        if len(h2) <= 12 or "类" in h2:
            cats.append(h2)
    return cats

def parse_e_matrix_from_table_text(raw: str) -> Optional[Tuple[List[str], List[Tuple[str, List[str]]]]]:
    parsed = parse_md_or_pipe_table(raw)
    if parsed:
        headers, rows = parsed
        perm_col = None
        for i, h in enumerate(headers):
            if "权限名" in h or "权限名称" in h:
                perm_col = i
                break
        if perm_col is None:
            for i in range(len(headers)):
                cnt = 0
                for r in rows[:30]:
                    if i < len(r) and PERM_CONST_PAT.fullmatch(r[i].strip() or ""):
                        cnt += 1
                if cnt >= 3:
                    perm_col = i
                    break
        if perm_col is None:
            return None
        cats = []
        cat_cols = []
        non_cat = {"序号", "权限分组", "权限名", "权限名称", "权限", "读写", "说明", "功能描述"}
        for i, h in enumerate(headers):
            if i == perm_col:
                continue
            h2 = norm(h)
            if not h2 or h2 in non_cat:
                continue
            cats.append(h2)
            cat_cols.append(i)
        if not cats:
            for i in range(len(headers)):
                if i in (0, 1, perm_col):
                    continue
                h2 = norm(headers[i])
                if h2 and h2 not in non_cat:
                    cats.append(h2)
                    cat_cols.append(i)
        if not cats:
            return None
        out_rows = []
        for r in rows:
            if perm_col >= len(r):
                continue
            perm_cell = r[perm_col].strip()
            m = PERM_CONST_PAT.search(perm_cell) or PERM_CONST_PAT.search(" ".join(r))
            if not m:
                continue
            perm = m.group(0)
            marked = []
            for col_i, catname in zip(cat_cols, cats):
                cell = r[col_i].strip() if col_i < len(r) else ""
                if cell in ("×", "x", "X", "✗", "✕"):
                    marked.append(catname)
            out_rows.append((perm, marked))
        return cats, out_rows
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        m = PERM_CONST_PAT.search(ln)
        if not m:
            continue
        perm = m.group(0)
        marked = []
        for km in re.finditer(r"([一-龥]{2,8})\s*[:：]\s*([×xX✗✕])", ln):
            marked.append(km.group(1))
        rows.append((perm, marked))
        cat_union = sorted({c for _, cats in rows for c in cats})
        return cat_union, rows
    return None

def build_e_mappings(
    standard_id: str,
    e_tables: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    perm2cats = defaultdict(set)
    cat2perms = defaultdict(set)
    for tbl in e_tables:
        raw = tbl.get("text", "") or ""
        res = parse_e_matrix_from_table_text(raw)
        if not res:
            continue
        cats, rows = res
        for perm, marked in rows:
            for cat in marked:
                perm2cats[perm].add(cat)
                cat2perms[cat].add(perm)
    perm_recs = []
    for perm in sorted(perm2cats.keys()):
        cats_list = sorted(perm2cats[perm])
        emb = "\n".join([
            f"[标准] {standard_id}",
            "[附录E低相关提示]",
            "[用途] 本条给出该权限在哪些服务类型下被提示“相关程度较低”，用于最小必要与合理性审查。",
            f"[权限名] {perm}",
            f"[低相关服务类型] {', '.join(cats_list) if cats_list else '（未解析到×标注；请检查表抽取质量）'}",
            "[注意] 未标注不等于合理或必要，应结合业务功能与必要个人信息范围综合判断。"
        ]).strip()
        perm_recs.append({
            "chunk_id": f"{standard_id}::E::perm::{perm}",
            "text": perm,
            "meta": {
                "type": "e-permission",
                "standard_id": standard_id,
                "appendix_letter": "E",
                "permission": perm,
                "low_related_categories": cats_list,
            },
            "embedding_text": emb
        })
    cat_recs = []
    if EMIT_E_CATEGORY:
        for cat in sorted(cat2perms.keys()):
            perms = sorted(cat2perms[cat])
            emb = "\n".join([
                f"[标准] {standard_id}",
                "[附录E低相关提示]",
                "[用途] 本条给出该服务类型下被提示“相关程度较低”的权限集合，用于超规风险提示与重点审查。",
                f"[服务类型] {cat}",
                f"[低相关权限] {', '.join(perms) if perms else '（无）'}",
                "[注意] 该集合是“低相关提示”，不是禁止清单。"
            ]).strip()
            cat_recs.append({
                "chunk_id": f"{standard_id}::E::cat::{cat}",
                "text": cat,
                "meta": {
                    "type": "e-category",
                    "standard_id": standard_id,
                    "appendix_letter": "E",
                    "category": cat,
                    "low_related_permissions": perms,
                },
                "embedding_text": emb
            })
    return perm_recs, cat_recs

def build_sensitive_rows_and_pack(table_ch: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = table_ch.get("meta", {}) or {}
    raw = table_ch.get("text", "") or ""
    standard_id = meta.get("standard_id", "")
    appendix = meta.get("appendix_letter") or "A"
    table_id = meta.get("table_id", "")
    table_title = meta.get("table_title", "")
    parsed = parse_md_or_pipe_table(raw)
    rows = []
    if parsed:
        headers, mdrows = parsed
        c_cat = None
        c_desc = None
        for i, h in enumerate(headers):
            if "类别" in h:
                c_cat = i
            if "描述" in h:
                c_desc = i
        if c_cat is None:
            c_cat = 0
        if c_desc is None:
            c_desc = 1 if len(headers) > 1 else 0
        for r in mdrows:
            cat = r[c_cat].strip() if c_cat < len(r) else ""
            desc = r[c_desc].strip() if c_desc < len(r) else ""
            if not cat:
                continue
            rows.append((cat, desc))
    else:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        for ln in lines:
            parts = re.split(r"\s{2,}", ln)
            if len(parts) >= 2 and len(parts[0]) <= 12:
                rows.append((parts[0].strip(), parts[1].strip()))
    row_recs = []
    row_sents = []
    for i, (cat, desc) in enumerate(rows, start=1):
        cat = norm(cat)
        desc = norm(desc)
        emb = "\n".join([
            f"[标准] {standard_id}",
            f"[附录] {appendix}",
            f"[表] {table_id} {table_title}".strip(),
            "[用途] 本条给出敏感个人信息类别及描述，用于隐私政策披露与风险提示。",
            f"[敏感个人信息类别] {cat}",
            f"[描述] {desc}"
        ]).strip()
        row_recs.append({
            "chunk_id": f"{standard_id}::{appendix}::{table_id}::sensrow{i}",
            "text": f"{cat} {desc}".strip(),
            "meta": {
                "type": "sensitive-row",
                "standard_id": standard_id,
                "appendix_letter": appendix,
                "table_id": table_id,
                "table_title": table_title,
                "sensitive_category": cat,
            },
            "embedding_text": emb
        })
        row_sents.append(f"类别={cat}；描述={desc}")
    pack_emb = "\n".join([
        f"[标准] {standard_id}",
        f"[附录] {appendix}",
        f"[敏感个人信息证据包] {table_title}".strip(),
        "[内容]",
        "\n- " + "\n- ".join(row_sents) if row_sents else "（空）"
    ]).strip()
    pack_rec = {
        "chunk_id": f"{standard_id}::{appendix}::{table_id}::sensitive-pack",
        "text": pack_emb,
        "meta": {
            "type": "sensitive-pack",
            "standard_id": standard_id,
            "appendix_letter": appendix,
            "table_id": table_id,
            "table_title": table_title,
        },
        "embedding_text": pack_emb
    }
    return row_recs, pack_rec

def main():
    assert IN_PATH.exists(), f"Input not found: {IN_PATH}"
    chunks = load_jsonl(IN_PATH)
    chunk_by_id = {c.get("chunk_id"): c for c in chunks if c.get("chunk_id")}
    out: List[Dict[str, Any]] = []
    appendix_a_text_by_cat: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    appendix_a_text_by_section: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    e_intro_texts_by_std = defaultdict(list)
    e_tables_by_std = defaultdict(list)
    for ch in chunks:
        meta = ch.get("meta", {}) or {}
        tp = meta.get("type")
        std = meta.get("standard_id", "")
        if tp == "text":
            out.append(build_norm_text_record(ch))
            if meta.get("appendix_letter") == "A" and std.endswith("2022"):
                sect_title = meta.get("section_title") or ""
                cat = extract_category_name_from_section_title(sect_title)
                if cat:
                    appendix_a_text_by_cat[(std, cat)].append(ch)
                if sect_title.startswith("A."):
                    appendix_a_text_by_section[(std, sect_title)].append(ch)
            if meta.get("appendix_letter") == "E":
                t = (ch.get("text") or "").strip()
                if t and len(t) < 2600 and ("×" in t or "相关程度较低" in t or "附录E" in t):
                    e_intro_texts_by_std[std].append(t)
        elif tp == "table":
            if is_2022_appendix_e_table(meta):
                e_tables_by_std[std].append(ch)
    a_rows_by_cat = defaultdict(list)
    a_rows_text_by_cat = defaultdict(str)
    a_table_refs_by_cat = defaultdict(list)
    for ch in chunks:
        meta = ch.get("meta", {}) or {}
        if meta.get("type") != "table":
            continue
        std = meta.get("standard_id", "")
        if is_2022_appendix_a_business_table(meta) and std.endswith("2022"):
            rows, cat, rows_text = build_appendix_a_rows_from_table(ch, chunk_by_id)
            out.extend(rows)
            if cat:
                a_rows_by_cat[(std, cat)].extend(rows)
                a_rows_text_by_cat[(std, cat)] = rows_text
                a_table_refs_by_cat[(std, cat)].append((meta.get("table_id",""), meta.get("table_title","")))
        elif is_2022_appendix_d_table(meta) and std.endswith("2022"):
            out.extend(build_appendix_d_permission_rows(ch))
        elif is_2025_sensitive_table(meta):
            srows, spack = build_sensitive_rows_and_pack(ch)
            out.extend(srows)
            out.append(spack)
    for (std, cat), text_chunks in appendix_a_text_by_cat.items():
        rows_text = a_rows_text_by_cat.get((std, cat), "")
        table_refs = a_table_refs_by_cat.get((std, cat), [])
        pack = build_appendix_a_category_pack_from_text(
            standard_id=std,
            category=cat,
            appendix="A",
            text_chunks=text_chunks,
            rows_text=rows_text,
            table_refs=table_refs
        )
        out.append(pack)
    for std, e_tables in e_tables_by_std.items():
        if not e_tables:
            continue
        rule = build_e_rule_from_intro_texts(std, e_intro_texts_by_std.get(std, []))
        out.append(rule)
        perm_recs, cat_recs = build_e_mappings(std, e_tables)
        out.extend(perm_recs)
        out.extend(cat_recs)
    dump_jsonl(OUT_PATH, out)
    type_cnt = defaultdict(int)
    for r in out:
        type_cnt[safe_get(r, "meta", "type", default="")] += 1
    print("Input chunks:", len(chunks))
    print("Output records:", len(out))
    print("Type distribution:", dict(sorted(type_cnt.items(), key=lambda x: (-x[1], x[0]))))
    print("Output:", str(OUT_PATH))

if __name__ == "__main__":
    main()
