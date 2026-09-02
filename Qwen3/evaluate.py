"""
Full-dataset evaluation for the fine-tuned LoRA adapter.
Runs inference over every record in a JSONL file and reports aggregate
precision / recall / F1 (set-based, over the extracted resolution codes)
plus exact-match accuracy.

Usage:
    python evaluate.py \
        --base_model Qwen/Qwen3-0.6B \
        --adapter    checkpoints/final \
        --data       data/resolution_test.jsonl \
        --max_new_tokens 256

    python evaluate.py --all --checkpoints old_checkpoints
"""

import argparse
import json
import re
import os
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(base_model: str, adapter: str, device: str):
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device,
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model, tokenizer, messages: list[dict], max_new_tokens: int, device: str) -> str:
    """
    Build the prompt from all turns EXCEPT the last assistant turn,
    then let the model generate.
    """
    # drop the final assistant turn so the model has to generate it
    prompt_messages = [m for m in messages if not (m["role"] == "assistant" and m is messages[-1])]

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,   # appends the <|im_start|>assistant token
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy — deterministic for eval
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Metrics: precision / recall over sets of extracted codes
# ---------------------------------------------------------------------------

def parse_code_set(text: str) -> set[str]:
    """
    Split a "code1; code2, code3" style string into a normalized set.
    Accepts both ';' and ',' as separators (gold data in this project uses
    ', ' while the model may emit ';' — we don't want formatting choice
    to affect the metric), and strips whitespace/case differences.
    """
    if not text or not text.strip():
        return set()

    raw_parts = re.split(r"[;,]", text)
    codes = {p.strip().upper() for p in raw_parts if p.strip()}
    return codes


def compute_precision_recall(gold_set: set[str], pred_set: set[str]) -> dict:
    """
    Standard set-based precision / recall / F1.
    Edge cases:
      - both empty                 -> precision = recall = f1 = 1.0 (correctly predicted "nothing")
      - pred empty, gold non-empty -> precision = 0.0, recall = 0.0 (missed everything)
      - gold empty, pred non-empty -> precision = 0.0, recall = 0.0 (hallucinated codes)
    """
    tp = len(gold_set & pred_set)

    if not gold_set and not pred_set:
        precision = recall = f1 = 1.0
    else:
        precision = tp / len(pred_set) if pred_set else (1.0 if not gold_set else 0.0)
        recall    = tp / len(gold_set) if gold_set else (1.0 if not pred_set else 0.0)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": len(pred_set - gold_set),
        "fn": len(gold_set - pred_set),
    }

def evaluate_checkpoint(model, tokenizer, records, max_new_tokens, device):
    # running totals for two flavors of aggregate metrics:
    #   - micro: pool TP/FP/FN across all samples, then compute one P/R/F1
    #   - macro: average each sample's own P/R/F1
    micro_tp = micro_fp = micro_fn = 0
    macro_precisions, macro_recalls, macro_f1s = [], [], []
    exact_matches = 0

    for record in tqdm(records, leave=False):
        messages = record["messages"]

        gold = next(
            (m["content"] for m in reversed(messages)
             if m["role"] == "assistant"),
            ""
        )

        prediction = predict(
            model,
            tokenizer,
            messages,
            max_new_tokens,
            device,
        )

        if gold.strip() == prediction.strip():
            exact_matches += 1

        gold_set = parse_code_set(gold)
        pred_set = parse_code_set(prediction)

        metrics = compute_precision_recall(gold_set, pred_set)

        micro_tp += metrics["tp"]
        micro_fp += metrics["fp"]
        micro_fn += metrics["fn"]

        macro_precisions.append(metrics["precision"])
        macro_recalls.append(metrics["recall"])
        macro_f1s.append(metrics["f1"])

    n = len(records)

    micro_precision = (
        micro_tp / (micro_tp + micro_fp)
        if (micro_tp + micro_fp) > 0 else 1.0
    )

    micro_recall = (
        micro_tp / (micro_tp + micro_fn)
        if (micro_tp + micro_fn) > 0 else 1.0
    )

    micro_f1 = (
        2 * micro_precision * micro_recall /
        (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    macro_precision = sum(macro_precisions) / n
    macro_recall = sum(macro_recalls) / n
    macro_f1 = sum(macro_f1s) / n

    exact_match_accuracy = exact_matches / n

    return {
        "exact_match_accuracy": exact_match_accuracy,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "tp": micro_tp,
        "fp": micro_fp,
        "fn": micro_fn,
    }

def evaluate_all(args: argparse.Namespace) -> None:
    print(f"\n{'=' * 72}")
    print(f"Evaluating all checkpoints evolution")
    print(f"{'=' * 72}")

    results = []

    # load full dataset
    with open(args.data) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # checkpoint directory is now configurable via --checkpoints
    checkpoint_root = args.checkpoints

    if not os.path.isdir(checkpoint_root):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_root!r}. "
            f"Pass the correct path with --checkpoints."
        )

    # discover checkpoint directories
    checkpoint_steps = sorted(
        [
            int(d)
            for d in os.listdir(checkpoint_root)
            if os.path.isdir(os.path.join(checkpoint_root, d))
               and d.isdigit()
        ]
    )

    # evaluate numbered checkpoints
    for step in checkpoint_steps:
        adapter_path = os.path.join(checkpoint_root, str(step))

        print(f"\nEvaluating checkpoint {step}")

        model, tokenizer = load(
            args.base_model,
            adapter_path,
            args.device
        )

        metrics = evaluate_checkpoint(
            model,
            tokenizer,
            records,
            args.max_new_tokens,
            args.device
        )

        metrics["step"] = step
        results.append(metrics)
        print(f"F1 score: {metrics['micro_f1']:.3f}")

        del model
        torch.cuda.empty_cache()

    # evaluate final adapter
    print("\nEvaluating final checkpoint")

    final_adapter_path = os.path.join(checkpoint_root, "final")

    model, tokenizer = load(
        args.base_model,
        final_adapter_path,
        args.device
    )

    metrics = evaluate_checkpoint(
        model,
        tokenizer,
        records,
        args.max_new_tokens,
        args.device
    )

    # Obtaining the step integer:
    latest_checkpoint_path = os.path.join(
        checkpoint_root, "latest_checkpoint.pt")

    if os.path.exists(latest_checkpoint_path):
        ckpt = torch.load(
            latest_checkpoint_path,
            map_location="cpu",
            weights_only=False
        )
        final_step = ckpt["global_step"]
    else:
        print(
            f"Warning: {latest_checkpoint_path!r} not found — "
            f"labeling the final checkpoint's step as 'final' instead of "
            f"a global step number."
        )
        final_step = "final"

    metrics["step"] = final_step

    # append, just like the prior checkpoints.

    results.append(metrics)
    print(f"F1 score: {metrics['micro_f1']}")
    df = pd.DataFrame(results)

    del model
    torch.cuda.empty_cache()

    output_path = os.path.join(checkpoint_root, "evaluation_all.csv")
    df.to_csv(output_path, index=False)

    print(f"\nSaved results to {output_path}")
    print(df)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",         default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter",            default=None,
                         help="Path to a single adapter dir. Defaults to "
                              "'<checkpoints>/final' if not set. Ignored when --all is used.")
    parser.add_argument("--checkpoints",        default="./checkpoints",
                         help="Root directory containing numbered checkpoint "
                              "subdirs, a 'final' subdir, and ideally "
                              "'latest_checkpoint.pt'. Used by --all, and as "
                              "the default base for --adapter.")
    parser.add_argument("--data",               default="data/resolution_test.jsonl")
    parser.add_argument("--max_new_tokens",     type=int, default=298)
    parser.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--all",                action="store_true", help="Process all checkpoints")
    args = parser.parse_args()

    if args.adapter is None:
        args.adapter = os.path.join(args.checkpoints, "final")

    if args.all:
        evaluate_all(args)
    else:
        # load full dataset
        with open(args.data) as f:
            records = [json.loads(l) for l in f if l.strip()]

        print(f"Loading base model:   {args.base_model}")
        print(f"Loading LoRA adapter: {args.adapter}")
        model, tokenizer = load(args.base_model, args.adapter, args.device)
        n = len(records)
        print(f"Model ready. Evaluating on {n} records.\n")

        evaluation_results = evaluate_checkpoint(model, tokenizer, records, args.max_new_tokens, args.device)

        print(f"\n{'='*72}")
        print(f"RESULTS over {n} records")
        print(f"{'='*72}")
        print()
        print("Micro-averaged (pool all TP/FP/FN, then compute):")
        print(f"  Precision: {evaluation_results['micro_precision']:.3f}")
        print(f"  Recall:    {evaluation_results['micro_recall']:.3f}")
        print(f"  F1:        {evaluation_results['micro_f1']:.3f}")
        print(f"  (TP={evaluation_results['tp']}, "
              f"FP={evaluation_results['fp']}, "
              f"FN={evaluation_results['fn']})"
             )
        print()
        print("Macro-averaged (average each sample's own P/R/F1):")
        print(f"  Precision: {evaluation_results['macro_precision']:.3f}")
        print(f"  Recall:    {evaluation_results['macro_recall']:.3f}")
        print(f"  F1:        {evaluation_results['macro_f1']:.3f}")
        print(f"{'='*72}")

if __name__ == "__main__":
    main()
