"""
Text extractor from the SpiritRAG collection of documents.

Usage:
    pip install pymupdf
    python extract_pdfs.py data -o corpus.csv
    python extract_pdfs.py data -o corpus.csv --trim-top 0.07 --trim-bottom 0.06
"""
import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

pymupdf.TOOLS.mupdf_display_errors(False)  # keep the console quiet on slightly broken PDFs

META_FIELDS = ["symbol", "publication_date", "session_year", "title"]
CSV_FIELDS = META_FIELDS + ["content"]
LOG_FIELDS = ["symbol", "pdf_path", "status", "n_pages", "n_two_col_pages", "n_chars", "note"]

# Zone used for column detection (ignores running header / footer).
DETECT_TOP, DETECT_BOTTOM = 0.08, 0.06
CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# --------------------------------------------------------------------------- #
# Two-column detection
# --------------------------------------------------------------------------- #
def page_is_two_column(page, min_words=60, min_gutter=10.0, tol_frac=0.04,
                       side_share=0.2, y_overlap=0.6):
    """
    True if the page body has a vertical white 'gutter' near the middle of the
    text area, with a real chunk of text on both sides of it.

    * gutter  = run of >= min_gutter pt, within the central 30% of the text width,
                that (almost) no word crosses. A few crossing words are tolerated
                (tol_frac) so a full-width title above the columns doesn't hide it.
    * Justified single-column text never has such a gutter, because most lines
      run straight through the middle of the page.
    * Both sides must hold >= side_share of the words and overlap vertically,
      which rules out e.g. a lone right-aligned signature block.
    """
    h, w = page.rect.height, page.rect.width
    words = [
        wd for wd in page.get_text("words")
        if h * DETECT_TOP <= (wd[1] + wd[3]) / 2 <= h * (1 - DETECT_BOTTOM)
    ]
    n = len(words)
    if n < min_words:
        return False

    x_lo = min(wd[0] for wd in words)
    x_hi = max(wd[2] for wd in words)
    span = x_hi - x_lo
    if span < 0.4 * w:
        return False

    # how many words cover each 1-pt bin across the text width
    nb = int(span) + 3
    diff = [0] * (nb + 1)
    for wd in words:
        diff[int(wd[0] - x_lo)] += 1
        diff[int(wd[2] - x_lo) + 1] -= 1
    cover, run = [], 0
    for d in diff[:nb]:
        run += d
        cover.append(run)

    tol = tol_frac * n
    lo_bin, hi_bin = int(0.35 * span), int(0.65 * span)
    best, start = (0, 0, 0), None  # (width, start_bin, end_bin)
    for i in range(lo_bin, hi_bin + 1):
        if cover[i] <= tol:
            if start is None:
                start = i
            if i - start + 1 > best[0]:
                best = (i - start + 1, start, i)
        else:
            start = None
    if best[0] < min_gutter:
        return False

    gutter_mid = x_lo + (best[1] + best[2]) / 2
    left = [wd for wd in words if (wd[0] + wd[2]) / 2 < gutter_mid]
    right = [wd for wd in words if (wd[0] + wd[2]) / 2 >= gutter_mid]
    if min(len(left), len(right)) < side_share * n:
        return False

    ly = [(wd[1] + wd[3]) / 2 for wd in left]
    ry = [(wd[1] + wd[3]) / 2 for wd in right]
    total = max(max(ly), max(ry)) - min(min(ly), min(ry))
    overlap = min(max(ly), max(ry)) - max(min(ly), min(ry))
    return total > 0 and overlap / total >= y_overlap


def detect_two_column_pages(doc, max_pages=40):
    """Return {page_index: bool} for the (sampled) pages that carry body text."""
    n = doc.page_count
    idxs = range(n) if n <= max_pages else sorted({round(i * (n - 1) / (max_pages - 1)) for i in range(max_pages)})
    return {i: page_is_two_column(doc[i]) for i in idxs}


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def page_text(page, trim_top=0.0, trim_bottom=0.0):
    """Paragraph-level text: each PDF text block -> one line, blocks split by blank line."""
    h = page.rect.height
    paras = []
    for x0, y0, x1, y1, txt, _no, btype in page.get_text("blocks", sort=True):
        if btype != 0:  # skip images
            continue
        yc = (y0 + y1) / 2
        if yc < h * trim_top or yc > h * (1 - trim_bottom):
            continue
        txt = re.sub(r"\s*\n\s*", " ", txt)          # un-wrap hard line breaks
        txt = re.sub(r"[ \t\xa0]+", " ", txt).strip()
        if txt:
            paras.append(txt)
    return "\n\n".join(paras)


def clean(text):
    text = CTRL_CHARS.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --------------------------------------------------------------------------- #
# Per-document worker (runs in a subprocess)
# --------------------------------------------------------------------------- #
def read_metadata(folder):
    """Return (metadata dict, note). Uses the first *.jsonl file containing a record with 'symbol'."""
    problem = "no metadata jsonl with 'symbol' found"

    for jp in sorted(Path(folder).glob("*.jsonl")):
        try:
            with jp.open("r", encoding="utf-8-sig") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except Exception as e:  # noqa: BLE001
                        problem = f"bad jsonl {jp.name} line {line_no}: {e}"
                        continue

                    if isinstance(data, dict) and "symbol" in data:
                        meta = {}
                        for k in META_FIELDS:
                            v = data.get(k, "")
                            if isinstance(v, (list, tuple)):
                                v = "; ".join(map(str, v))
                            meta[k] = "" if v is None else str(v)

                        missing = [k for k in META_FIELDS if k not in data]
                        return meta, (
                            f"missing fields: {missing}" if missing else ""
                        )

        except Exception as e:  # noqa: BLE001
            problem = f"error reading {jp.name}: {e}"

    return {k: "" for k in META_FIELDS}, problem


def process(args):
    pdf_path, opts = args
    meta, note = read_metadata(Path(pdf_path).parent)

    log = {
        "symbol": meta["symbol"],
        "pdf_path": pdf_path,
        "status": "ok",
        "n_pages": 0,
        "n_two_col_pages": 0,
        "n_chars": 0,
        "note": note,
    }

    try:
        doc = pymupdf.open(pdf_path)
        log["n_pages"] = doc.page_count

        parts = []
        for page in doc:
            parts.append(
                page_text(
                    page,
                    opts["trim_top"],
                    opts["trim_bottom"],
                )
            )

        content = clean("\n\n".join(p for p in parts if p))
        log["n_chars"] = len(content)

        if len(content) < 50:
            log["status"] = "no_text"  # most likely a scanned PDF -> needs OCR
            return None, log

        return {**meta, "content": content}, log

    except Exception as e:  # noqa: BLE001
        log["status"] = "error"
        log["note"] = f"{type(e).__name__}: {e}"
        return None, log

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="root folder that contains one sub-folder per document")
    ap.add_argument("-o", "--output", default="corpus.csv")
    ap.add_argument("--log", default=None, help="per-document log CSV (default: <output>_log.csv)")
    ap.add_argument("--suffix", default="en.pdf", help="only PDFs whose name ends with this (default: en.pdf)")
    ap.add_argument("--doc-threshold", type=float, default=0.25,
                    help="skip a document when at least this share of its pages are two-column (default 0.25)")
    ap.add_argument("--keep-two-column", action="store_true", help="disable two-column detection")
    ap.add_argument("--trim-top", type=float, default=0.0, help="drop text in the top X fraction of each page (e.g. 0.07)")
    ap.add_argument("--trim-bottom", type=float, default=0.0, help="drop text in the bottom X fraction (e.g. 0.06)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args()

    root = Path(a.data_dir)
    pdfs = sorted(str(p) for p in root.rglob("*") if p.is_file() and p.name.lower().endswith(a.suffix.lower()))
    if not pdfs:
        sys.exit(f"No files ending with '{a.suffix}' found under {root}")
    print(f"Found {len(pdfs)} PDFs", file=sys.stderr)

    opts = {"doc_threshold": a.doc_threshold, "keep_two_column": a.keep_two_column,
            "trim_top": a.trim_top, "trim_bottom": a.trim_bottom}
    log_path = a.log or str(Path(a.output).with_suffix("")) + "_log.csv"
    counts = {}

    with open(a.output, "w", newline="", encoding="utf-8") as fo, \
         open(log_path, "w", newline="", encoding="utf-8") as fl, \
         ProcessPoolExecutor(max_workers=a.workers) as ex:
        out = csv.DictWriter(fo, fieldnames=CSV_FIELDS)
        lg = csv.DictWriter(fl, fieldnames=LOG_FIELDS)
        out.writeheader()
        lg.writeheader()
        jobs = ((p, opts) for p in pdfs)
        for i, (row, log) in enumerate(ex.map(process, jobs, chunksize=4), 1):
            if row:
                out.writerow(row)
            lg.writerow(log)
            counts[log["status"]] = counts.get(log["status"], 0) + 1
            if i % 100 == 0 or i == len(pdfs):
                print(f"  {i}/{len(pdfs)}  {counts}", file=sys.stderr)

    print(f"\nDone. {counts.get('ok', 0)} documents -> {a.output}\nPer-document log -> {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
