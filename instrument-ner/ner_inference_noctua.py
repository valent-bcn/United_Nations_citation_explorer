"""
Inference counterpart to `token_train_noctua.py`.

Takes N random resolutions from the same CSV used for training, runs them
through the fine-tuned TREATY token classifier, and prints the detected
treaty names to stdout.

Typical use:
    python ner_inference_sample.py
    python ner_inference_sample.py --seed 7 -n 10
    python ner_inference_sample.py --doc-df ./my_resolutions.csv --model-dir ./treaty-ner
    python ner_inference_sample.py --doc-id A/RES/45/158        # inspect one doc
    python ner_inference_sample.py --list-checkpoints           # what's in ./treaty-ner
    python ner_inference_sample.py --checkpoint 1568            # pin one training step

Notes:
    * A Trainer `output_dir` is not itself a model: it holds one
      `checkpoint-N` per saved step (plus eval_metrics.csv). `--model-dir`
      names the run, `--checkpoint` picks the step within it: "best"
      (default, read from trainer_state.json), "last", a step number, a
      folder name, or a full path. `--model-dir` pointing straight at a
      saved model still works.
    * These are full fine-tuned models, not PEFT/LoRA adapters: each
      checkpoint carries a complete `model.safetensors` and its own
      tokenizer, so nothing is loaded on top of a base model.
    * Resolution texts are usually longer than the model's 512-token window,
      so each document is split into overlapping character windows; the
      predicted offsets are shifted back to the original text and
      overlapping duplicates are merged.
"""

import argparse
import json
import os
import re
import sys

import pandas as pd
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

DEFAULT_DOC_DF = "/home/user/branes/NER-data/ga_resolutions_1946_2019.csv"
DEFAULT_MODEL_DIR = "./treaty-ner"


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def list_checkpoints(model_dir: str):
    """Return [(step, path), ...] for every checkpoint-N in model_dir, by step."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"{model_dir!r} is not a directory")
    found = []
    for name in os.listdir(model_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        path = os.path.join(model_dir, name)
        if m and os.path.isdir(path):
            found.append((int(m.group(1)), path))
    return sorted(found)


def read_best_checkpoint(model_dir: str, checkpoints):
    """
    Look up the checkpoint the Trainer itself considered best.

    `trainer_state.json` records `best_model_checkpoint` / `best_metric`
    (populated because training ran with load_best_model_at_end=True and
    metric_for_best_model="f1"). The recorded path comes from the training
    machine, so only its basename is trusted and re-joined to model_dir.
    Returns (path, metric) or (None, None).
    """
    for _step, path in reversed(checkpoints):
        state_path = os.path.join(path, "trainer_state.json")
        if not os.path.isfile(state_path):
            continue
        try:
            with open(state_path) as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        best = state.get("best_model_checkpoint")
        if not best:
            continue
        local = os.path.join(model_dir, os.path.basename(str(best).rstrip("/\\")))
        if os.path.isdir(local):
            return local, state.get("best_metric")
    return None, None


def resolve_model_dir(model_dir: str, checkpoint: str = "best") -> str:
    """
    Turn (model_dir, checkpoint) into a single loadable model directory.

    `checkpoint` accepts:
        "best"  - the checkpoint recorded as best in trainer_state.json
                  (falls back to the last one if that is unavailable)
        "last"  - the highest training step present
        "1568"  - that exact step
        a path  - used as-is
    A model_dir that is itself a saved model (has config.json) is returned
    unchanged unless a specific step or path was requested.
    """
    # An explicit path wins over everything else.
    if checkpoint and (os.sep in checkpoint or checkpoint.startswith("checkpoint-")):
        candidate = checkpoint if os.path.isdir(checkpoint) else os.path.join(model_dir, checkpoint)
        if not os.path.isfile(os.path.join(candidate, "config.json")):
            raise FileNotFoundError(f"No config.json in {candidate!r}")
        return candidate

    is_model_itself = os.path.isfile(os.path.join(model_dir, "config.json"))
    if is_model_itself and checkpoint in (None, "best", "last"):
        return model_dir

    checkpoints = list_checkpoints(model_dir)
    if not checkpoints:
        raise FileNotFoundError(
            f"No config.json and no checkpoint-N directory found in {model_dir!r}. "
            "Point --model-dir at a training output_dir or a saved model."
        )
    available = ", ".join(str(s) for s, _ in checkpoints)

    if checkpoint and checkpoint.isdigit():
        step = int(checkpoint)
        for s, path in checkpoints:
            if s == step:
                print(f"[info] using checkpoint-{s} (requested)")
                return path
        raise FileNotFoundError(
            f"checkpoint-{step} not found in {model_dir!r}. Available steps: {available}"
        )

    if checkpoint == "last":
        step, path = checkpoints[-1]
        print(f"[info] using checkpoint-{step} (last of: {available})")
        return path

    best_path, best_metric = read_best_checkpoint(model_dir, checkpoints)
    if best_path:
        metric = f", f1={best_metric:.4f}" if isinstance(best_metric, float) else ""
        print(f"[info] using {os.path.basename(best_path)} "
              f"(best per trainer_state.json{metric}; available: {available})")
        return best_path

    step, path = checkpoints[-1]
    print(f"[warn] no best_model_checkpoint recorded; falling back to "
          f"checkpoint-{step} (available: {available})")
    return path


def print_checkpoint_table(model_dir: str):
    """Print the checkpoints in model_dir, marking the one recorded as best."""
    checkpoints = list_checkpoints(model_dir)
    if not checkpoints:
        print(f"No checkpoint-N directory in {model_dir!r}")
        return
    best_path, best_metric = read_best_checkpoint(model_dir, checkpoints)
    print(f"Checkpoints in {model_dir}:")
    for step, path in checkpoints:
        mark = ""
        if best_path and os.path.abspath(path) == os.path.abspath(best_path):
            metric = f" f1={best_metric:.4f}" if isinstance(best_metric, float) else ""
            mark = f"   <- best{metric}"
        print(f"  checkpoint-{step}{mark}")


def load_ner_pipeline(model_dir: str, checkpoint: str, device: int):
    """Build an aggregated NER pipeline from the selected checkpoint."""
    resolved = resolve_model_dir(model_dir, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = AutoModelForTokenClassification.from_pretrained(resolved)
    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # same strategy as evaluate_generalization()
        device=device,
    )
    return ner_pipe, resolved


# ---------------------------------------------------------------------------
# LONG-DOCUMENT HANDLING
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200):
    """
    Yield (chunk, offset) pairs covering `text`, cut on whitespace so a
    treaty name is unlikely to be split, with `overlap` characters of
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


def predict_treaties(ner_pipe, text: str, min_score: float):
    """Run the pipeline over a full document and return merged TREATY spans."""
    raw = []
    for chunk, offset in chunk_text(text):
        for pred in ner_pipe(chunk):
            if pred["score"] < min_score:
                continue
            raw.append({
                "text": text[pred["start"] + offset: pred["end"] + offset],
                "start": pred["start"] + offset,
                "end": pred["end"] + offset,
                "score": float(pred["score"]),
                "label": pred.get("entity_group", "TREATY"),
            })
    return merge_spans(raw)


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def context_snippet(text: str, start: int, end: int, window: int = 60) -> str:
    """One-line snippet of surrounding text with the span in [brackets]."""
    left = text[max(0, start - window): start].replace("\n", " ")
    right = text[end: end + window].replace("\n", " ")
    snippet = f"...{left}[{text[start:end]}]{right}..."
    return re.sub(r"\s+", " ", snippet).strip()


def print_document(doc_id, text, spans, show_context: bool):
    print("=" * 78)
    print(f"DOC {doc_id}   ({len(text)} chars)")
    print("-" * 78)

    if not spans:
        print("  (no treaty detected)")
        print()
        return

    for s in spans:
        print(f"  • {s['text']}")
        print(f"      score={s['score']:.3f}  chars={s['start']}–{s['end']}  label={s['label']}")
        if show_context:
            print(f"      {context_snippet(text, s['start'], s['end'])}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Sample resolutions from a CSV and print the treaties detected by the fine-tuned NER model.",
    )
    p.add_argument("--doc-df", default=DEFAULT_DOC_DF,
                   help=f"CSV of documents to sample from (default: {DEFAULT_DOC_DF})")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                   help=f"Fine-tuned model or Trainer output_dir (default: {DEFAULT_MODEL_DIR})")
    p.add_argument("--checkpoint", default="best",
                   help="Which checkpoint inside --model-dir to load: 'best' (per "
                        "trainer_state.json), 'last', a step number like 1568, a folder "
                        "name like checkpoint-1568, or a full path (default: best)")
    p.add_argument("--list-checkpoints", action="store_true",
                   help="List the checkpoints in --model-dir and exit")
    p.add_argument("-n", "--n-samples", type=int, default=5,
                   help="Number of random documents to run (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for the sample (default: 42)")
    p.add_argument("--doc-id", default=None,
                   help="Run one specific document id instead of a random sample")
    p.add_argument("--id-col", default="res_id2",
                   help="Id column in the CSV, renamed to 'id' (default: res_id2)")
    p.add_argument("--text-col", default="content",
                   help="Text column in the CSV (default: content)")
    p.add_argument("--min-score", type=float, default=0.5,
                   help="Minimum span score to report (default: 0.5)")
    p.add_argument("--no-context", action="store_true",
                   help="Hide the surrounding-text snippet for each detection")
    p.add_argument("--device", type=int, default=None,
                   help="CUDA device index, or -1 for CPU (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_checkpoints:
        print_checkpoint_table(args.model_dir)
        return

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    df = pd.read_csv(args.doc_df)
    if args.id_col in df.columns:
        df = df.rename(columns={args.id_col: "id"})
    if "id" not in df.columns:
        sys.exit(f"No id column: {args.id_col!r} not found in {list(df.columns)}")
    if args.text_col not in df.columns:
        sys.exit(f"No text column: {args.text_col!r} not found in {list(df.columns)}")

    df = df[df[args.text_col].notna()]

    if args.doc_id is not None:
        sample = df[df["id"].astype(str) == str(args.doc_id)]
        if sample.empty:
            sys.exit(f"Document id {args.doc_id!r} not found in {args.doc_df}")
    else:
        n = min(args.n_samples, len(df))
        sample = df.sample(n=n, random_state=args.seed)

    ner_pipe, resolved = load_ner_pipeline(args.model_dir, args.checkpoint, device)

    print()
    print(f"Corpus     : {args.doc_df}  ({len(df)} documents)")
    print(f"Checkpoint : {resolved}")
    print(f"Sample     : {len(sample)} document(s), seed={args.seed}, min_score={args.min_score}")
    print()

    total = 0
    for _, row in sample.iterrows():
        text = str(row[args.text_col])
        spans = predict_treaties(ner_pipe, text, args.min_score)
        total += len(spans)
        print_document(row["id"], text, spans, show_context=not args.no_context)

    print("=" * 78)
    print(f"{total} treaty mention(s) detected across {len(sample)} document(s).")


if __name__ == "__main__":
    main()