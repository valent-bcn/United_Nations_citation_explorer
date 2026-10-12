"""
Inference counterpart to `token_train_noctua.py`.

Runs the fine-tuned INSTRUMENT token classifier over EVERY resolution in
ga_resolutions_1946_2025.csv (no KNOWN/UNKNOWN split, no reference tables,
no spaCy) and saves all detected instrument mentions to a CSV. Detections are
NOT printed to the terminal.

Output columns: doc_id, doc_year, text, score, start, end, label

Typical use:
    python ner_inference_all.py
    python ner_inference_all.py --min-score 0.7
    python ner_inference_all.py --model ./checkpoints_bert-base-uncased/checkpoint-1061
    python ner_inference_all.py -n 20 --output-csv /tmp/test.csv   # quick trial run
    python ner_inference_all.py --doc-id A/RES/45/158              # one document

Notes:
    * `--model` must point straight at one saved checkpoint folder (the one
      containing config.json and model.safetensors).
    * These are full fine-tuned models, not PEFT/LoRA adapters: each
      checkpoint carries a complete `model.safetensors` and its own
      tokenizer, so nothing is loaded on top of a base model.
    * Resolution texts are usually longer than the model's 512-token window,
      so each document is split into overlapping character windows; the
      predicted offsets are shifted back to the original text and
      overlapping duplicates are merged.
    * The year is parsed from the `date_c` column ("%d %B %Y"); rows whose
      date cannot be parsed get an empty doc_year.
"""

import argparse
import csv
import os
import sys

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

DEFAULT_DOC_DF = "/home/user/branes/NER-data/ga_resolutions_1946_2025.csv"
DEFAULT_MODEL = "./checkpoints-bert-large-uncased/checkpoint-9144"
DEFAULT_MIN_SCORE = 0.5
DEFAULT_OUTPUT_CSV = "/home/user/branes/NER-data/NER-inference_1946-2025.csv"
DEFAULT_BATCH_SIZE = 16

OUTPUT_COLUMNS = ["doc_id", "doc_year", "text", "score", "start", "end", "label"]


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_ner_pipeline(model_path: str, device: int):
    """Build an aggregated NER pipeline from a saved checkpoint folder."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # same strategy as evaluate_generalization()
        device=device,
    )


# ---------------------------------------------------------------------------
# LONG-DOCUMENT HANDLING
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200):
    """
    Yield (chunk, offset) pairs covering `text`, cut on whitespace so a
    instrument name is unlikely to be split, with `overlap` characters of
    context repeated between consecutive chunks.
    """
    if len(text) <= max_chars:
        yield text, 0
        return

    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            space = text.rfind(" ", start + max_chars // 2, end)
            if space != -1:
                end = space
        yield text[start:end], start
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)


def merge_spans(spans):
    """
    Drop overlapping / duplicated predictions produced by the chunk overlap,
    keeping the longest span and, for equal length, the highest score.
    """
    ordered = sorted(spans, key=lambda s: (s["end"] - s["start"], s["score"]), reverse=True)
    kept = []
    for s in ordered:
        if any(s["start"] < k["end"] and s["end"] > k["start"] for k in kept):
            continue
        kept.append(s)
    return sorted(kept, key=lambda s: s["start"])


def predict_instruments(ner_pipe, text: str, min_score: float, batch_size: int):
    """Run the pipeline over a full document and return merged INSTRUMENT spans."""
    if not text.strip():
        return []

    chunks = list(chunk_text(text))
    # Passing a list returns one list of predictions per chunk.
    outputs = ner_pipe([chunk for chunk, _ in chunks], batch_size=batch_size)

    raw = []
    for (_chunk, offset), preds in zip(chunks, outputs):
        for pred in preds:
            if pred["score"] < min_score:
                continue
            raw.append({
                "text": text[pred["start"] + offset: pred["end"] + offset],
                "start": pred["start"] + offset,
                "end": pred["end"] + offset,
                "score": float(pred["score"]),
                "label": pred.get("entity_group", "INSTRUMENT"),
            })
    return merge_spans(raw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the fine-tuned INSTRUMENT NER model over every resolution in a CSV "
                    "and save the detected instrument mentions to a CSV (nothing is printed per detection).",
    )
    p.add_argument("--doc-df", default=DEFAULT_DOC_DF,
                   help=f"CSV of documents to run (default: {DEFAULT_DOC_DF})")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Path to a saved fine-tuned checkpoint folder (default: {DEFAULT_MODEL})")
    p.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV,
                   help=f"CSV file for detected findings (default: {DEFAULT_OUTPUT_CSV})")
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                   help=f"Minimum span score to keep (default: {DEFAULT_MIN_SCORE})")
    p.add_argument("--id-col", default="res_id2",
                   help="Id column in the CSV, renamed to 'id' (default: res_id2)")
    p.add_argument("--text-col", default="content",
                   help="Text column in the CSV (default: content)")
    p.add_argument("--date-col", default="date_c",
                   help="Date column used to derive the year, format '%%d %%B %%Y' (default: date_c)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Chunks per forward pass (default: {DEFAULT_BATCH_SIZE})")
    p.add_argument("--device", type=int, default=None,
                   help="CUDA device index, or -1 for CPU (default: auto)")
    p.add_argument("-n", "--n-samples", type=int, default=None,
                   help="Optional: run only N random documents (default: all documents)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed used with --n-samples (default: 42)")
    p.add_argument("--doc-id", default=None,
                   help="Optional: run one specific document id only")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable the progress bar")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(os.path.join(args.model, "config.json")):
        sys.exit(f"--model {args.model!r} is not a saved model folder (no config.json found)")

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    # ---- Load every resolution (no KNOWN/UNKNOWN filtering) ----------------
    df = pd.read_csv(args.doc_df)
    if args.id_col in df.columns:
        df = df.rename(columns={args.id_col: "id"})
    if "id" not in df.columns:
        sys.exit(f"No id column: {args.id_col!r} not found in {list(df.columns)}")
    if args.text_col not in df.columns:
        sys.exit(f"No text column: {args.text_col!r} not found in {list(df.columns)}")
    if args.date_col not in df.columns:
        sys.exit(f"No date column: {args.date_col!r} not found in {list(df.columns)}")

    df = df[df[args.text_col].notna()].copy()
    df["year"] = pd.to_datetime(
        df[args.date_col], format="%d %B %Y", errors="coerce"
    ).dt.year.astype("Int64")

    if args.doc_id is not None:
        df = df[df["id"].astype(str) == str(args.doc_id)]
        if df.empty:
            sys.exit(f"Document id {args.doc_id!r} not found in {args.doc_df}")
    elif args.n_samples is not None:
        df = df.sample(n=min(args.n_samples, len(df)), random_state=args.seed)

    n_no_year = int(df["year"].isna().sum())

    ner_pipe = load_ner_pipeline(args.model, device)

    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Corpus     : {args.doc_df}  ({len(df)} documents)")
    print(f"Model      : {args.model}")
    print(f"Min score  : {args.min_score}")
    print(f"Output     : {args.output_csv}")
    if n_no_year:
        print(f"[warn] {n_no_year} document(s) have an unparsable {args.date_col!r}; doc_year left empty")

    # ---- Inference: rows are written as each document finishes -------------
    total = 0
    docs_with_hits = 0
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        rows = zip(df["id"], df["year"], df[args.text_col])
        for doc_id, year, text in tqdm(rows, total=len(df), unit="doc",
                                       disable=args.no_progress):
            spans = predict_instruments(ner_pipe, str(text), args.min_score, args.batch_size)
            if spans:
                docs_with_hits += 1
            doc_year = None if pd.isna(year) else int(year)

            for s in spans:
                writer.writerow({
                    "doc_id": doc_id,
                    "doc_year": doc_year,
                    "text": s["text"],
                    "score": s["score"],
                    "start": s["start"],
                    "end": s["end"],
                    "label": s["label"],
                })
            total += len(spans)
            fh.flush()

    print(f"{total} instrument mention(s) in {docs_with_hits}/{len(df)} document(s).")
    print(f"Findings CSV : {args.output_csv}")


if __name__ == "__main__":
    main()