import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz
from PIL import Image
from tqdm import tqdm


CONFIG = {
    "input_dir": r"./dataset/PDF",
    "output_dir": r"./dataset/PNG",
    "recursive": True,
    "dpi": 600,
    "workers": max(1, (os.cpu_count() or 4) - 1),
    "transparent": False,           # 是否输出透明背景（RGBA）
    "force_overwrite": False,
    "optimize_png": True,
    "stop_on_error": False,
    "jpeg_fallback": True,
}


def safe_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name)
    return name[:200] if len(name) > 200 else name


def find_pdfs(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted([p for p in input_dir.rglob("*.pdf") if p.is_file()])
    return sorted([p for p in input_dir.glob("*.pdf") if p.is_file()])


def render_one_page(
    pdf_path: Path,
    out_root: Path,
    page_index: int,
    dpi: int,
    transparent: bool,
    force_overwrite: bool,
    optimize_png: bool,
    jpeg_fallback: bool,
):
    pdf_stem = safe_name(pdf_path.stem)
    out_dir = out_root / pdf_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page_{page_index + 1:04d}.png"
    if out_path.exists() and not force_overwrite:
        return "skipped", str(out_path)
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=bool(transparent))
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        img.save(out_path, format="PNG", optimize=bool(optimize_png))
        return "rendered", str(out_path)
    except Exception as e:
        if jpeg_fallback:
            try:
                page = doc.load_page(page_index)
                scale = dpi / 72.0
                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat, alpha=bool(transparent))
                pix.save(str(out_path))
                return "rendered_fallback", str(out_path)
            except Exception as e2:
                return "error", f"{pdf_path} page {page_index + 1}: {e} | fallback failed: {e2}"
        return "error", f"{pdf_path} page {page_index + 1}: {e}"
    finally:
        doc.close()


def main():
    input_dir = Path(CONFIG["input_dir"]).expanduser().resolve()
    output_dir = Path(CONFIG["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    pdfs = find_pdfs(input_dir, CONFIG["recursive"])
    if not pdfs:
        print(f"[WARN] 在目录中未找到 PDF: {input_dir}")
        return
    print(f"[INFO] Found {len(pdfs)} PDF(s):")
    for p in pdfs[:10]:
        print("  -", p)
    if len(pdfs) > 10:
        print(f"  ... (+{len(pdfs)-10} more)")
    tasks: list[tuple[Path, int]] = []
    print("[INFO] Scanning page counts...")
    for pdf in tqdm(pdfs, desc="Scanning PDFs"):
        try:
            doc = fitz.open(pdf)
            n = doc.page_count
            doc.close()
            for i in range(n):
                tasks.append((pdf, i))
        except Exception as e:
            msg = f"[ERROR] Failed to open {pdf}: {e}"
            print(msg)
            if CONFIG["stop_on_error"]:
                return
    if not tasks:
        print("[WARN] 没有可渲染的页面任务。")
        return
    print(f"[INFO] Total pages: {len(tasks)}")
    print(f"[INFO] Output: {output_dir}")
    print(
        f"[INFO] dpi={CONFIG['dpi']} workers={CONFIG['workers']} "
        f"recursive={CONFIG['recursive']} transparent={CONFIG['transparent']} "
        f"force_overwrite={CONFIG['force_overwrite']}"
    )
    rendered = 0
    skipped = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=CONFIG["workers"]) as ex:
        futures = [
            ex.submit(
                render_one_page,
                pdf, output_dir, page_index,
                CONFIG["dpi"],
                CONFIG["transparent"],
                CONFIG["force_overwrite"],
                CONFIG["optimize_png"],
                CONFIG["jpeg_fallback"],
            )
            for (pdf, page_index) in tasks
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Rendering pages"):
            status, info = fut.result()
            if status.startswith("rendered"):
                rendered += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print("[ERROR]", info)
                if CONFIG["stop_on_error"]:
                    break
    print("\n[DONE]")
    print(f"  rendered: {rendered}")
    print(f"  skipped : {skipped}")
    print(f"  errors  : {errors}")
    print(f"  output  : {output_dir}")


if __name__ == "__main__":
    main()
