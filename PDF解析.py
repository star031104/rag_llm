import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List
from paddleocr import PaddleOCRVL


CONFIG = {
    "pdf_root": r".\dataset\PDF",
    "out_md_dir": r".\dataset\PDF解析\OUT_MD",
    "out_json_dir": r".\dataset\PDF解析\OUT_JSON",
    "out_assets_dir": r".\dataset\PDF解析\OUT_ASSETS",
    "recursive": True,
    "device": "gpu:0",
    "enable_cpu_fallback": True,
    "use_doc_orientation_classify": True,
    "use_doc_unwarping": True,
    "use_layout_detection": True,
    "format_block_content": True,
    "use_queues_in_predict": False,
    "skip_if_exists": True,
}


os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")


def safe_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name)
    return name[:200] if len(name) > 200 else name


def is_oom_error(e: Exception) -> bool:
    msg = str(e)
    return ("ResourceExhaustedError" in msg) or ("Out of memory" in msg) or ("Cannot allocate" in msg)


def iter_pdfs(root: Path, recursive: bool):
    if recursive:
        yield from sorted([p for p in root.rglob("*.pdf") if p.is_file()])
    else:
        yield from sorted([p for p in root.glob("*.pdf") if p.is_file()])


def build_pipeline(device: str) -> PaddleOCRVL:
    return PaddleOCRVL(
        device=device,
        use_doc_orientation_classify=CONFIG["use_doc_orientation_classify"],
        use_doc_unwarping=CONFIG["use_doc_unwarping"],
        use_layout_detection=CONFIG["use_layout_detection"],
        format_block_content=CONFIG["format_block_content"],
    )


def predict_pdf(pipeline: PaddleOCRVL, pdf_path: Path):
    if CONFIG["use_queues_in_predict"]:
        try:
            return pipeline.predict(str(pdf_path), use_queues=True)
        except TypeError:
            return pipeline.predict(str(pdf_path))
    return pipeline.predict(str(pdf_path))


def merge_pdf_markdown(pipeline: PaddleOCRVL, page_markdown_list: List[Dict[str, Any]]) -> str:
    return pipeline.concatenate_markdown_pages(page_markdown_list)


def build_pdf_level_json(
    pdf_path: Path,
    page_json_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "doc_type": "pdf",
        "source_pdf": str(pdf_path),
        "doc_name": pdf_path.name,
        "pages": page_json_list,
        "num_pages": len(page_json_list),
    }


def save_markdown_assets(markdown_images: Dict[str, Any], assets_root: Path):
    if not markdown_images:
        return
    for rel_path, pil_img in markdown_images.items():
        save_path = assets_root / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pil_img.save(save_path)
        except Exception:
            pass


def process_one_pdf(
    pdf_path: Path,
    gpu_pipeline: PaddleOCRVL,
    cpu_pipeline_holder: Dict[str, Any],
    out_md_dir: Path,
    out_json_dir: Path,
    out_assets_dir: Path,
):
    pdf_stem = safe_name(pdf_path.stem)
    out_md_path = out_md_dir / f"{pdf_stem}.md"
    out_json_path = out_json_dir / f"{pdf_stem}.json"
    if CONFIG["skip_if_exists"] and out_md_path.exists() and out_json_path.exists():
        print(f"[SKIP] {pdf_path.name} (already done)")
        return
    def run_with_pipeline(pipeline: PaddleOCRVL):
        page_md_infos: List[Dict[str, Any]] = []
        page_json_infos: List[Dict[str, Any]] = []
        output = predict_pdf(pipeline, pdf_path)
        for page_idx, res in enumerate(output):
            md_info = getattr(res, "markdown", None)
            js_info = getattr(res, "json", None)
            if md_info is None or js_info is None:
                raise RuntimeError("Result object missing `markdown` or `json` attributes in this version.")
            if isinstance(js_info, dict):
                js_info = {"page_index": page_idx, **js_info}
            else:
                js_info = {"page_index": page_idx, "raw": js_info}
            page_md_infos.append(md_info)
            page_json_infos.append(js_info)
            md_images = md_info.get("markdown_images", {}) if isinstance(md_info, dict) else {}
            if md_images:
                save_markdown_assets(md_images, out_assets_dir / pdf_stem)
        merged_md = merge_pdf_markdown(pipeline, page_md_infos)
        merged_json = build_pdf_level_json(pdf_path, page_json_infos)
        return merged_md, merged_json
    try:
        merged_md, merged_json = run_with_pipeline(gpu_pipeline)
    except Exception as e:
        if CONFIG["enable_cpu_fallback"] and is_oom_error(e):
            print(f"[OOM][GPU] {pdf_path.name} -> fallback to CPU")
            if cpu_pipeline_holder.get("cpu") is None:
                print("[INFO] Initializing CPU pipeline (slow, only once)...")
                cpu_pipeline_holder["cpu"] = build_pipeline("cpu")
            merged_md, merged_json = run_with_pipeline(cpu_pipeline_holder["cpu"])
        else:
            raise
    out_md_dir.mkdir(parents=True, exist_ok=True)
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_assets_dir.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text(merged_md, encoding="utf-8")
    out_json_path.write_text(json.dumps(merged_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {pdf_path.name} -> {out_md_path.name}, {out_json_path.name}")


def main():
    pdf_root = Path(CONFIG["pdf_root"]).expanduser().resolve()
    out_md_dir = Path(CONFIG["out_md_dir"]).expanduser().resolve()
    out_json_dir = Path(CONFIG["out_json_dir"]).expanduser().resolve()
    out_assets_dir = Path(CONFIG["out_assets_dir"]).expanduser().resolve()
    if not pdf_root.exists():
        raise FileNotFoundError(f"pdf_root 不存在：{pdf_root}")
    pdfs = list(iter_pdfs(pdf_root, CONFIG["recursive"]))
    if not pdfs:
        print(f"[WARN] 未找到 PDF：{pdf_root}")
        return
    print(f"[INFO] Found {len(pdfs)} PDFs under {pdf_root}")
    print("[INFO] Initializing GPU pipeline...")
    gpu_pipeline = build_pipeline(CONFIG["device"])
    cpu_pipeline_holder = {"cpu": None}
    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] Processing: {pdf_path.relative_to(pdf_root)}")
        try:
            process_one_pdf(
                pdf_path,
                gpu_pipeline=gpu_pipeline,
                cpu_pipeline_holder=cpu_pipeline_holder,
                out_md_dir=out_md_dir,
                out_json_dir=out_json_dir,
                out_assets_dir=out_assets_dir,
            )
        except Exception as e:
            print(f"[ERROR] {pdf_path.name}: {e}")
            continue
    print("\n[DONE] All PDFs processed.")


if __name__ == "__main__":
    main()
