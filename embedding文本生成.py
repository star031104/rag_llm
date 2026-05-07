import json
import re

IN_PATH = "chunks_final.jsonl"
OUT_PATH = "chunks_with_embedding_text.jsonl"

def clean(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    s = s.replace("\u0000", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def join_nonempty(parts, sep="\n"):
    parts = [clean(p) for p in parts if clean(p)]
    return sep.join(parts).strip()

def kv_lines(d: dict, keys: list[str], title: str = None):
    lines = []
    if title:
        lines.append(f"{title}：")
    for k in keys:
        v = d.get(k, "")
        v = clean(v)
        if v and v != "—":
            lines.append(f"- {k}: {v}")
    return "\n".join(lines).strip()

def build_embedding_text(obj: dict, max_chars: int = 2200) -> str:
    meta = obj.get("meta", {}) or {}
    t = meta.get("type", "unknown")
    standard_id = meta.get("standard_id", "")
    doc_name = meta.get("doc_name", "")
    section_path = meta.get("section_path", "") or meta.get("section_title", "")
    section_num = meta.get("section_num", "") or meta.get("section_no", "")

    header = join_nonempty([
        f"标准：{standard_id}" if standard_id else "",
        f"文档：{doc_name}" if doc_name else "",
        f"位置：{section_num} {section_path}".strip() if (section_num or section_path) else "",
        f"类型：{t}"
    ])

    body = ""
    raw_text = clean(obj.get("text", ""))

    if t == "text":
        term_cn = meta.get("term_cn", "")
        term_en = meta.get("term_en", "")
        text_role = meta.get("text_role", "")
        keywords = meta.get("keywords", [])
        kw = "、".join([clean(x) for x in (keywords or []) if clean(x)]) if isinstance(keywords, list) else clean(keywords)
        hints = join_nonempty([
            f"术语：{term_cn}（{term_en}）" if (term_cn or term_en) else "",
            f"角色：{text_role}" if text_role else "",
            f"关键词：{kw}" if kw else "",
        ])
        body = join_nonempty([
            hints,
            raw_text
        ])
    elif t == "table_row":
        table_id = meta.get("table_id", "")
        table_title = meta.get("table_title", "")
        table_context = meta.get("table_context_text", "")
        table_head = join_nonempty([
            f"表格：{table_id} {table_title}".strip() if (table_id or table_title) else "",
            f"表格语境：{clean(table_context)}" if table_context else "",
        ])
        normalized = kv_lines(meta, [
            "permission_group_en",
            "permission_group_cn",
            "permission_code",
            "permission_name_cn",
            "permission_cn",
            "function_desc_display",
            "pii_access_display",
            "biz_example_display",
            "row_role",
            "appendix_letter",
        ], title="结构化字段")
        col_headers = meta.get("col_headers", [])
        if isinstance(col_headers, list) and col_headers:
            cols = " | ".join([clean(x) for x in col_headers if clean(x)])
            cols_line = f"列：{cols}"
        else:
            cols_line = ""
        body = join_nonempty([
            table_head,
            cols_line,
            normalized,
            raw_text
        ])
    else:
        body = raw_text
    text = join_nonempty([header, body])
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text

def main():
    with open(IN_PATH, "r", encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            obj["embedding_text"] = build_embedding_text(obj)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
