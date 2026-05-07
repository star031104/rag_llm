import os, re, json, glob, html
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
P_END_RE = re.compile(r"</p\s*>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
MULTI_NL_RE = re.compile(r"\n{3,}")
SPACE_RE = re.compile(r"[ \t]+")
SPACE_BEFORE_NL_RE = re.compile(r"[ \t]+\n")
TOC_DOTLINE_RE = re.compile(r"…{2,}\s*(\d+|[IVXLC]+)\s*$", re.IGNORECASE)
HASH_RE = re.compile(r"^\s*#+\s*")
BULLET_RE = re.compile(r"^\s*[•·\-–—]+\s*")
NUM_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,8})\s*([^\d\s].+?)\s*$")
ONLY_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,8})\s*$")
LIST_ITEM_RE = re.compile(r"^\s*\d+\s*[)）]\s*")
CN_PAREN_ITEM_RE = re.compile(r"^\s*[（(]\d+[）)]\s*")
APPENDIX_TITLE_RE = re.compile(r"^\s*附录\s*([A-Z])\b(.*)$")
APP_SEC_WITH_TITLE_RE = re.compile(r"^\s*([A-Z])\.(\d+)\s*([^\d\s].+)?\s*$")
APP_ONLY_CODE_RE = re.compile(r"^\s*([A-Z])\.(\d+)\s*$")
TABLE_ID_RE = re.compile(r"(?:表|Table)\s*([A-Z]?\s*\.?\s*\d+(?:\.\d+)*)", flags=re.IGNORECASE)
CONT_RE = re.compile(r"(续|（续）|\(续\)|（续\d+）|\(续\d+\))")
TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", flags=re.IGNORECASE | re.DOTALL)
TD_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", flags=re.IGNORECASE | re.DOTALL)
TEXT_TABLE_REF_RE = re.compile(r"表\s*([A-Z]?\s*\.?\s*\d+(?:\.\d+)*)")

def strip_html_tags(s: str) -> str:
    if not s:
        return ""
    s = BR_RE.sub("\n", s)
    s = P_END_RE.sub("\n", s)
    s = TAG_RE.sub("", s)
    if "&" in s:
        s = html.unescape(s)
    s = MULTI_NL_RE.sub("\n\n", s)
    return s.strip()

def normalize_space_keep_lines(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = SPACE_RE.sub(" ", s)
    s = SPACE_BEFORE_NL_RE.sub("\n", s)
    s = MULTI_NL_RE.sub("\n\n", s)
    return s.strip()

def normalize_space(s: str) -> str:
    return normalize_space_keep_lines(s)

def infer_standard_id(doc_name: str) -> str:
    base = os.path.splitext(os.path.basename(doc_name or ""))[0]
    m = re.search(r"GBT\+(\d+)-(\d{4})", base, flags=re.IGNORECASE)
    if m:
        return f"GB/T {m.group(1)}-{m.group(2)}"
    return base or "UNKNOWN"

def is_noise_block(label: str, raw: str) -> bool:
    if not raw or len(raw.strip()) <= 1:
        return True
    if "Powered by TCPDF" in raw:
        return True
    if label in {"header", "footer", "header_image", "footer_image", "number", "aside_text"}:
        return True
    return False

def iter_blocks_in_order(doc: Dict[str, Any]):
    doc_name = doc.get("doc_name", "")
    for page in (doc.get("pages") or []):
        page_index = page.get("page_index")
        blocks = ((page.get("res") or {}).get("parsing_res_list") or [])
        for b in blocks:
            yield doc_name, page_index, b.get("block_label", ""), (b.get("block_content") or "")

SKIP_HEADINGS = {"目次", "前言", "引言", "参考文献"}

def is_toc_block_like(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 8:
        return False
    dotline = sum(1 for ln in lines if TOC_DOTLINE_RE.search(ln))
    return (dotline / max(1, len(lines))) >= 0.55

def sanitize_line(line: str) -> str:
    s = (line or "").strip()
    s = HASH_RE.sub("", s)
    s = BULLET_RE.sub("", s)
    return s.strip()

TEXT_MIN_CHARS = 220

def clamp_dotted_num(num: str, max_parts: int) -> Tuple[str, bool]:
    parts = num.split(".")
    if len(parts) <= max_parts:
        return num, False
    return ".".join(parts[:max_parts]), True

@dataclass
class Chunk:
    chunk_id: str
    text: str
    meta: Dict[str, Any]

def extract_table_id_and_title(near_title_text: str) -> Tuple[Optional[str], Optional[str], bool]:
    s = strip_html_tags(near_title_text)
    is_cont = bool(CONT_RE.search(s))
    m = TABLE_ID_RE.search(s)
    table_id = None
    if m:
        table_id = re.sub(r"\s+", "", m.group(1)).replace("..", ".")
    table_title = None
    if table_id:
        idx = s.lower().find("表")
        if idx >= 0:
            after = s[idx:]
            after = TABLE_ID_RE.sub("", after, count=1).strip()
            after = re.sub(r"^[\s:：\-—]+", "", after)
            after = CONT_RE.sub("", after).strip()
            table_title = after or None
    return table_id, table_title, is_cont

def html_table_to_matrix(table_html: str) -> List[List[str]]:
    trs = TR_RE.findall(table_html)
    matrix: List[List[str]] = []
    for tr in trs:
        tds = TD_RE.findall(tr)
        row = [normalize_space(strip_html_tags(td)) for td in tds]
        if row:
            matrix.append(row)
    return [r for r in matrix if any(c.strip() for c in r)]

TABLE_ROW_LIMIT_MD = 80
TABLE_COL_LIMIT_MD = 10

def table_matrix_to_text(matrix: List[List[str]], table_id: Optional[str], table_title: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    extra: Dict[str, Any] = {}
    header_line = f"表 {table_id}" if table_id else "表格"
    if table_title:
        header_line += f"：{table_title}"
    if not matrix:
        return header_line + "\n(空表/解析失败)", extra
    n_rows = len(matrix)
    n_cols = max(len(r) for r in matrix)
    extra["table_n_rows"] = n_rows
    extra["table_n_cols"] = n_cols
    extra["col_headers"] = (matrix[0] or [])[:50]
    if n_rows <= TABLE_ROW_LIMIT_MD and n_cols <= TABLE_COL_LIMIT_MD:
        norm = [r + [""] * (n_cols - len(r)) for r in matrix]
        header = norm[0]
        body = norm[1:] if n_rows > 1 else []
        md = [header_line]
        md.append("| " + " | ".join(header) + " |")
        md.append("| " + " | ".join(["---"] * n_cols) + " |")
        for r in body:
            md.append("| " + " | ".join(r) + " |")
        return "\n".join(md).strip(), extra
    out = [header_line]
    header = matrix[0]
    body = matrix[1:] if len(matrix) > 1 else []
    use_header = sum(1 for c in header if c.strip()) >= min(2, len(header))
    for i, row in enumerate(body[:400]):
        if use_header:
            pairs = []
            for k, v in zip(header, row):
                k = (k or "").strip() or "列"
                v = (v or "").strip()
                if v:
                    pairs.append(f"{k}：{v}")
            if pairs:
                out.append(f"- 第{i+1}行： " + "；".join(pairs))
        else:
            out.append(f"- 第{i+1}行： " + "；".join(c for c in row if (c or "").strip()))
    return "\n".join(out).strip(), extra

class ContextBuffer:
    def __init__(self, max_chars: int = 800):
        self.max_chars = max_chars
        self.q = deque()
        self.n = 0

    def reset(self):
        self.q.clear()
        self.n = 0

    def add(self, text: str):
        if not text:
            return
        self.q.append(text)
        self.n += len(text) + 1
        while self.n > self.max_chars and self.q:
            left = self.q.popleft()
            self.n -= len(left) + 1

    def get(self) -> str:
        return "\n".join(self.q).strip()

class SinglePassChunker:
    def __init__(self, doc: Dict[str, Any]):
        self.doc = doc
        self.doc_name = doc.get("doc_name", "")
        self.standard_id = infer_standard_id(self.doc_name)
        self.chunks: List[Chunk] = []
        self.seq_text = 0
        self.cur_key: Optional[str] = None
        self.cur_title: Optional[str] = None
        self.cur_buf: List[str] = []
        self.region: str = "front"
        self.appendix_letter: Optional[str] = None
        self.path_titles: List[str] = []
        self.seq_table = 0
        self.table_index: Dict[str, int] = {}
        self.last_table_id_seen: Optional[str] = None
        self.pending_table_id: Optional[str] = None
        self.pending_table_title: Optional[str] = None
        self.pending_is_cont: bool = False
        self.cur_appendix_letter_for_table: Optional[str] = None
        self.ctx = ContextBuffer(max_chars=800)

    def _set_section(self, key: str, title: str):
        self.cur_key = key
        self.cur_title = (title.strip() if title else None)

    def _set_path(self, level1: str, level2: Optional[str] = None):
        self.path_titles = [level1] + ([level2] if level2 else [])

    def _flush_text(self, force: bool = False):
        if not self.cur_buf:
            return
        body = normalize_space("\n".join(self.cur_buf))
        self.cur_buf = []
        if not body:
            return
        if is_toc_block_like(body):
            return
        if (not force) and len(body) < TEXT_MIN_CHARS and self.chunks:
            for j in range(len(self.chunks) - 1, -1, -1):
                if self.chunks[j].meta.get("type") == "text":
                    self.chunks[j].text = normalize_space(self.chunks[j].text + "\n\n" + body)
                    return
        head = ""
        if self.cur_key and self.cur_title:
            head = f"{self.cur_key} {self.cur_title}\n"
        elif self.cur_key:
            head = f"{self.cur_key}\n"
        cid = f"{self.standard_id}::sec{self.cur_key or 'UNNUM'}::text::{self.seq_text}"
        meta = {
            "standard_id": self.standard_id,
            "doc_name": self.doc_name,
            "type": "text",
            "section_num": self.cur_key,
            "section_title": self.cur_title,
            "region": self.region,
            "is_appendix": (self.region == "appendix"),
            "appendix_letter": self.appendix_letter if self.region == "appendix" else None,
            "section_path": " > ".join(self.path_titles) if self.path_titles else None,
        }
        self.chunks.append(Chunk(cid, normalize_space(head + body), meta))
        self.seq_text += 1

    def _ensure_region_main(self):
        if self.region != "main":
            self.region = "main"

    def _ensure_region_appendix(self, letter: str, full_title: str):
        self._flush_text(force=True)
        self.region = "appendix"
        self.appendix_letter = letter
        self.cur_appendix_letter_for_table = letter
        self.ctx.reset()
        anchor_key = f"APPENDIX_{letter}"
        self._set_section(anchor_key, full_title)
        self._set_path(f"附录{letter}", None)
        self.cur_buf.append(full_title)

    def process_non_table_block(self, raw: str):
        block_text = normalize_space_keep_lines(strip_html_tags(raw))
        if not block_text:
            return
        lines: List[str] = []
        for ln in block_text.splitlines():
            s = sanitize_line(ln)
            if s:
                lines.append(s)
        if not lines:
            return
        self.ctx.add("\n".join(lines))
        MAIN_MAX_PARTS = 3
        i = 0
        while i < len(lines):
            line = lines[i]
            if line == "参考文献":
                self._flush_text(force=True)
                self.region = "refs"
                i += 1
                continue
            if line in SKIP_HEADINGS:
                self._flush_text(force=True)
                self.region = "front"
                i += 1
                continue
            if TOC_DOTLINE_RE.search(line) and self.region == "front":
                i += 1
                continue
            mapp = APPENDIX_TITLE_RE.match(line)
            if mapp:
                letter = mapp.group(1)
                rest = (mapp.group(2) or "").strip()
                full_title = f"附录{letter}{(' ' + rest) if rest else ''}".strip()
                self._ensure_region_appendix(letter, full_title)
                i += 1
                continue
            if LIST_ITEM_RE.match(line) or CN_PAREN_ITEM_RE.match(line):
                if self.region in {"main", "appendix"}:
                    self.cur_buf.append(line)
                i += 1
                continue
            if self.region in {"front", "main"}:
                m = NUM_HEADING_RE.match(line)
                if m:
                    raw_num = m.group(1)
                    title = (m.group(2) or "").strip()
                    if title.startswith(")") or title.startswith("）"):
                        if self.region == "main":
                            self.cur_buf.append(line)
                        i += 1
                        continue
                    num, deeper = clamp_dotted_num(raw_num, MAIN_MAX_PARTS)
                    if deeper:
                        if self.region == "main":
                            self.cur_buf.append(line)
                        i += 1
                        continue
                    if self.region != "main":
                        self._flush_text(force=True)
                        self._ensure_region_main()
                    self._flush_text(force=True)
                    self._set_section(num, title or (lines[i+1] if i+1 < len(lines) else ""))
                    chap = num.split(".")[0]
                    self._set_path(chap, f"{num} {self.cur_title}".strip())
                    if not title and i + 1 < len(lines):
                        self.cur_title = lines[i+1]
                        self._set_path(chap, f"{num} {self.cur_title}".strip())
                        i += 2
                        continue
                    i += 1
                    continue
                m2 = ONLY_NUM_RE.match(line)
                if m2:
                    raw_num = m2.group(1)
                    num, deeper = clamp_dotted_num(raw_num, MAIN_MAX_PARTS)
                    if deeper:
                        if self.region == "main":
                            self.cur_buf.append(line)
                        i += 1
                        continue
                    title = lines[i+1] if i+1 < len(lines) else ""
                    if title in SKIP_HEADINGS:
                        self._flush_text(force=True)
                        self.region = "front"
                        i += 2
                        continue
                    if self.region != "main":
                        self._flush_text(force=True)
                        self._ensure_region_main()
                    self._flush_text(force=True)
                    self._set_section(num, title)
                    chap = num.split(".")[0]
                    self._set_path(chap, f"{num} {title}".strip())
                    i += 2
                    continue
                if self.region == "main":
                    self.cur_buf.append(line)
                i += 1
                continue
            if self.region == "appendix":
                ma = APP_SEC_WITH_TITLE_RE.match(line)
                if ma:
                    letter = ma.group(1)
                    n1 = ma.group(2)
                    title = (ma.group(3) or "").strip()
                    if self.appendix_letter and letter != self.appendix_letter:
                        self.cur_buf.append(line); i += 1; continue
                    key = f"{letter}.{n1}"
                    self._flush_text(force=True)
                    self._set_section(key, title or (lines[i+1] if i+1 < len(lines) else ""))
                    self._set_path(f"附录{letter}", f"{key} {self.cur_title}".strip())
                    if not title and i + 1 < len(lines):
                        self.cur_title = lines[i+1]
                        self._set_path(f"附录{letter}", f"{key} {self.cur_title}".strip())
                        i += 2; continue
                    i += 1; continue
                ma2 = APP_ONLY_CODE_RE.match(line)
                if ma2:
                    letter, n1 = ma2.group(1), ma2.group(2)
                    if self.appendix_letter and letter != self.appendix_letter:
                        self.cur_buf.append(line); i += 1; continue
                    key = f"{letter}.{n1}"
                    title2 = lines[i+1] if i+1 < len(lines) else ""
                    self._flush_text(force=True)
                    self._set_section(key, title2)
                    self._set_path(f"附录{letter}", f"{key} {title2}".strip())
                    i += 2
                    continue
                self.cur_buf.append(line)
                i += 1
                continue
            i += 1

    def process_table_title_block(self, raw: str):
        tid, ttitle, is_cont = extract_table_id_and_title(raw)
        if tid:
            self.pending_table_id = tid
            self.pending_table_title = ttitle
            self.pending_is_cont = is_cont

    def process_table_block(self, raw: str, page_index: Any):
        matrix = html_table_to_matrix(raw)
        table_id = self.pending_table_id or self.last_table_id_seen
        table_title = self.pending_table_title
        is_cont = self.pending_is_cont
        table_text, extra = table_matrix_to_text(matrix, table_id, table_title)
        context_text = self.ctx.get()
        if table_id and is_cont and table_id in self.table_index:
            idx = self.table_index[table_id]
            self.chunks[idx].text = normalize_space(self.chunks[idx].text + "\n\n---\n\n" + table_text)
            pages = self.chunks[idx].meta.setdefault("table_pages", [])
            if page_index not in pages:
                pages.append(page_index)
            self.chunks[idx].meta["is_continued"] = True
            if not self.chunks[idx].meta.get("table_context_text") and context_text:
                self.chunks[idx].meta["table_context_text"] = context_text
        else:
            cid = f"{self.standard_id}::table{table_id or 'UNKNOWN'}::table::{self.seq_table}"
            meta = {
                "standard_id": self.standard_id,
                "doc_name": self.doc_name,
                "type": "table",
                "table_id": table_id,
                "table_title": table_title,
                "is_continuation": bool(is_cont),
                "table_pages": [page_index],
                "appendix_letter": self.cur_appendix_letter_for_table,
                "table_context_text": context_text or None,
            }
            meta.update(extra)
            self.chunks.append(Chunk(cid, table_text, meta))
            if table_id:
                self.table_index[table_id] = len(self.chunks) - 1
            self.seq_table += 1
        self.pending_table_id = None
        self.pending_table_title = None
        self.pending_is_cont = False
        if table_id:
            self.last_table_id_seen = table_id

    def finalize(self):
        self._flush_text(force=True)
        filtered: List[Chunk] = []
        for c in self.chunks:
            if c.meta.get("type") == "text":
                if c.meta.get("region") in {"main", "appendix"}:
                    filtered.append(c)
            else:
                filtered.append(c)
        self.chunks = filtered
        return self.chunks

def link_text_table_refs_same_standard(all_chunks: List[Chunk]) -> None:
    table_key_to_chunk_ids: Dict[Tuple[str, str], List[str]] = {}
    for c in all_chunks:
        if c.meta.get("type") == "table":
            sid = c.meta.get("standard_id")
            tid = c.meta.get("table_id")
            if sid and tid:
                table_key_to_chunk_ids.setdefault((sid, tid), []).append(c.chunk_id)
    table_key_to_text_chunk_ids: Dict[Tuple[str, str], List[str]] = {}
    for c in all_chunks:
        if c.meta.get("type") != "text":
            continue
        sid = c.meta.get("standard_id")
        if not sid:
            continue
        found = []
        for m in TEXT_TABLE_REF_RE.finditer(c.text):
            tid = re.sub(r"\s+", "", m.group(1)).replace("..", ".")
            if tid:
                found.append(tid)
        if not found:
            continue
        uniq, seen = [], set()
        for x in found:
            if x not in seen:
                uniq.append(x); seen.add(x)
        c.meta["ref_table_ids"] = uniq
        tcids = []
        for tid in uniq:
            tcids.extend(table_key_to_chunk_ids.get((sid, tid), []))
        if tcids:
            tcids2, seen2 = [], set()
            for x in tcids:
                if x not in seen2:
                    tcids2.append(x); seen2.add(x)
            c.meta["ref_table_chunk_ids"] = tcids2
        for tid in uniq:
            table_key_to_text_chunk_ids.setdefault((sid, tid), []).append(c.chunk_id)
    for c in all_chunks:
        if c.meta.get("type") != "table":
            continue
        sid = c.meta.get("standard_id")
        tid = c.meta.get("table_id")
        refs = table_key_to_text_chunk_ids.get((sid, tid), []) if sid and tid else []
        c.meta["referenced_by_text"] = bool(refs)
        if refs:
            c.meta["referenced_by_chunk_ids"] = refs

DOT_SPACE_RE = re.compile(r"\s*\.\s*")

def _normalize_num_token(s: str) -> str:
    s = (s or "").strip()
    s = DOT_SPACE_RE.sub(".", s)         # "3 .11" -> "3.11"
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def fix_broken_section_titles_inplace(all_chunks: List[Chunk]) -> int:
    changed = 0
    for c in all_chunks:
        if c.meta.get("type") != "text":
            continue
        st = c.meta.get("section_title")
        if not isinstance(st, str):
            continue
        if not re.fullmatch(r"\.\d+(?:\.\d+)*", st.strip()):
            continue
        lines = (c.text or "").strip().splitlines()
        if not lines:
            continue
        first = lines[0].strip()
        num = _normalize_num_token(first)
        if not re.fullmatch(r"\d+(?:\.\d+)+", num):
            if len(lines) >= 2:
                num2 = _normalize_num_token(lines[0].strip() + lines[1].strip())
                if re.fullmatch(r"\d+(?:\.\d+)+", num2):
                    num = num2
                else:
                    continue
            else:
                continue
        title = ""
        for ln in lines[1:6]:
            if ln.strip() and not re.fullmatch(r"\.\d+(?:\.\d+)*", ln.strip()):
                title = ln.strip()
                break
        c.meta["section_num"] = num
        if title:
            c.meta["section_title"] = title
        sp = c.meta.get("section_path")
        if isinstance(sp, str) and sp.strip():
            parts = [p.strip() for p in sp.split(">")]
            parts[-1] = f"{num} {c.meta.get('section_title') or ''}".strip()
            c.meta["section_path"] = " > ".join(parts)
        cid = c.chunk_id or ""
        if isinstance(cid, str) and "::sec" in cid:
            c.chunk_id = re.sub(r"::sec[^:]+::", f"::sec{num}::", cid, count=1)
        changed += 1
    return changed

def _find_last_section_heading_start(buffer_text: str, section_id: str) -> Optional[int]:
    pat = rf"(^|\n)\s*{re.escape(section_id)}\s+.+"
    last = None
    for m in re.finditer(pat, buffer_text, flags=re.MULTILINE):
        last = m
    return None if last is None else last.start()

def rebuild_table_context_inplace(all_chunks: List[Chunk],
                                  buffer_keep: int = 20000,
                                  context_window: int = 3000) -> int:
    changed = 0
    buffer_text = ""
    last_table_boundary = 0
    cur_doc = None

    def _trim_buffer():
        nonlocal buffer_text, last_table_boundary
        if len(buffer_text) > buffer_keep:
            cut = len(buffer_text) - buffer_keep
            buffer_text = buffer_text[-buffer_keep:]
            last_table_boundary = max(0, last_table_boundary - cut)

    for c in all_chunks:
        doc_name = c.meta.get("doc_name")
        if doc_name != cur_doc:
            cur_doc = doc_name
            buffer_text = ""
            last_table_boundary = 0
        if c.meta.get("type") == "text":
            t = (c.text or "").strip()
            if t:
                buffer_text = (buffer_text + "\n" + t).strip()
                _trim_buffer()
            continue
        if c.meta.get("type") != "table":
            continue
        table_id = (c.meta.get("table_id") or "").strip()
        old = c.meta.get("table_context_text") or ""
        if not buffer_text:
            new = ""
        else:
            start = _find_last_section_heading_start(buffer_text, table_id) if table_id else None
            if start is None:
                start = last_table_boundary
            ctx = buffer_text[start:].strip()
            if len(ctx) > context_window:
                ctx = ctx[-context_window:].strip()
            if len(ctx) < 80:
                widen_start = max(0, start - 1200)
                ctx2 = buffer_text[widen_start:].strip()
                if len(ctx2) > context_window:
                    ctx2 = ctx2[-context_window:].strip()
                ctx = ctx2
            new = ctx
        if new and new != old:
            c.meta["table_context_text"] = new
            changed += 1
        last_table_boundary = len(buffer_text)
    return changed


def discover_json_files(base_dir: str, json_folder_name: str = "OUT_JSON") -> List[str]:
    folder = os.path.join(base_dir, json_folder_name)
    return sorted(glob.glob(os.path.join(folder, "*.json")))

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    json_paths = discover_json_files(base_dir, "OUT_JSON")
    if not json_paths:
        raise FileNotFoundError(f"No .json files found in {os.path.join(base_dir, 'OUT_JSON')}")
    all_chunks: List[Chunk] = []
    for p in json_paths:
        with open(p, "r", encoding="utf-8") as f:
            doc = json.load(f)
        chunker = SinglePassChunker(doc)
        for _doc_name, page_index, label, raw in iter_blocks_in_order(doc):
            if is_noise_block(label, raw):
                continue
            if label in {"figure_title", "table_title"}:
                chunker.process_table_title_block(raw)
                continue
            if label == "table":
                chunker.process_table_block(raw, page_index)
                continue
            chunker.process_non_table_block(raw)
        all_chunks.extend(chunker.finalize())
    fixed_sec = fix_broken_section_titles_inplace(all_chunks)
    fixed_ctx = rebuild_table_context_inplace(all_chunks, buffer_keep=20000, context_window=3000)
    link_text_table_refs_same_standard(all_chunks)
    out_path = os.path.join(base_dir, "chunks.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"input json files: {len(json_paths)} (folder: {os.path.join(base_dir, 'OUT_JSON')})")
    print(f"output chunks: {len(all_chunks)} -> {out_path}")
    print("single-pass enabled: each doc blocks traversed exactly once")

if __name__ == "__main__":
    main()
