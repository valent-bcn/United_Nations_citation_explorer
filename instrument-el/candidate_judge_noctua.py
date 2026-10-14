"""
Judge NER-entity -> candidate-title matches with Qwen3 on the extremes of a CSV.

From the input CSV it keeps the N rows with the highest `rerank_score` and the
N rows with the lowest (default N=100 -> 200 rows). For each row it shows the
LLM the `query_context` plus the top 3 candidates from `top_k_matches`
(shuffled, so position 1 is not always the best one) and stores the model's
reasoning and its answer.

Usage:
    python candidate_judge_csv.py --input entity_candidates_alias.csv --output alias_judged.csv
    python candidate_judge_csv.py --input entity_candidates_no_match.csv --output no_match_judged.csv

Columns appended to the output:
    selection           "top" or "bottom" (which end of rerank_score the row came from)
    source_row          row index in the input CSV
    shown_candidates    JSON list of the 3 candidates in the order the LLM saw them
    thinking_reasoning  Qwen3's <think> content
    LLM_answer          number the LLM returned (1-3 = position in shown_candidates, 0 = none)
    LLM_answer_title    title/year that LLM_answer points to (empty when 0 / unparseable)
    LLM_answer_year
"""
import argparse
import ast
import json
import os
import random
import re
import sys
from tqdm import tqdm

import pandas as pd

THINK_END_TOKEN_ID = 151668  # </think> in the Qwen3 tokenizer
N_SHOWN = 3
REQUIRED_COLUMNS = {"query_context", "top_k_matches", "rerank_score"}

SYSTEM_PROMPT = """
You are a title-matching classifier.

Determine which candidate title matches the title enclosed in square brackets
in the provided content.

Return ONLY the number of the matching candidate:
- 1, 2, 3, etc. if there is a match
- 0 if none match
- Do not provide explanations or any other text.
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Path to the candidates CSV")
    p.add_argument("--output", required=True, help="Path of the CSV to write")
    p.add_argument("--n", type=int, default=100, help="Rows to take from each end of rerank_score (default 100)")
    p.add_argument("--model", default="Qwen/Qwen3-14B", help="HF model name (default Qwen/Qwen3-14B)")
    p.add_argument("--seed", type=int, default=42, help="Seed for the candidate shuffle (default 42)")
    p.add_argument("--max-new-tokens", type=int, default=32768, help="Generation limit (default 32768)")
    p.add_argument("--save-every", type=int, default=10, help="Write a partial output every K rows (default 10)")
    
    p.add_argument("--temperature", type=float, default=0,
               help="Sampling temperature; lower = more deterministic (default 0)")
    return p.parse_args()


def select_extremes(df, n):
    """Top-n and bottom-n rows by rerank_score, without duplicates."""
    df = df.dropna(subset=["rerank_score"])
    top = df.nlargest(n, "rerank_score").assign(selection="top")
    bottom = df.nsmallest(n, "rerank_score").assign(selection="bottom")
    selected = pd.concat([top, bottom])
    selected = selected[~selected.index.duplicated(keep="first")]
    selected.index.name = "source_row"
    return selected.reset_index()


def get_candidates(raw, k=N_SHOWN):
    """First k distinct (title, year) candidates from the top_k_matches string, best score first."""
    try:
        items = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
    except (ValueError, SyntaxError, TypeError):
        return []
    items = sorted(items, key=lambda c: c.get("score", float("-inf")), reverse=True)
    seen, out = set(), []
    for item in items:
        key = (item.get("title"), item.get("year"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": item.get("title"), "year": item.get("year")})
        if len(out) == k:
            break
    return out


def build_content_prompt(context, shown):
    lines = "\n".join(
        f"{i}: 'title': '{c['title']}', 'year': {c['year']}" for i, c in enumerate(shown, start=1)
    )
    return f"{str(context).strip()}\n\n{lines}"


def parse_answer(content, n_shown):
    """First integer in the model output, if it is a valid option (0..n_shown), else None."""
    m = re.search(r"\d+", content)
    if not m:
        return None
    value = int(m.group())
    return value if 0 <= value <= n_shown else None


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    return tokenizer, model


def judge(tokenizer, model, content_prompt, max_new_tokens, temperature):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    if temperature > 0:
        gen_kwargs = dict(do_sample=True, temperature=temperature, top_p=0.95, top_k=20)
    else:
        gen_kwargs = dict(do_sample=False)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        # position right after the last </think>
        index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        index = len(output_ids)  # never finished thinking -> empty content -> no answer

    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return thinking, content


def attach_results(selected, shown_all, thinking, answers, titles, years):
    out = selected.iloc[: len(thinking)].copy()
    out["shown_candidates"] = shown_all
    out["thinking_reasoning"] = thinking
    out["LLM_answer"] = pd.array(answers, dtype="Int64")
    out["LLM_answer_title"] = titles
    try:
        out["LLM_answer_year"] = pd.array(years, dtype="Int64")  # keeps 1991, not 1991.0
    except (TypeError, ValueError):
        out["LLM_answer_year"] = years
    return out


def save(df, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)


def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(f"Input CSV is missing required columns: {sorted(missing)}")

    selected = select_extremes(df, args.n)
    print(f"Selected {len(selected)} rows ({args.n} top + {args.n} bottom by rerank_score)", flush=True)
    if len(selected) != 2 * args.n:
        print(f"WARNING: expected {2 * args.n} rows but got {len(selected)} (not enough scored rows in the input)")

    tokenizer, model = load_model(args.model)
    rng = random.Random(args.seed)

    shown_all, thinking_all, answers, titles, years = [], [], [], [], []
    invalid_rows = []
    total = len(selected)

    for i, row in enumerate(
        tqdm(
            selected.itertuples(index=False),
            total=total,
            desc="Judging",
            unit="row",
        ), start=1):
        candidates = get_candidates(row.top_k_matches)

        if not candidates:
            shown, thinking, answer = [], "", None
        else:
            shown = candidates[:]
            rng.shuffle(shown)  # avoid always putting the best candidate first
            prompt = build_content_prompt(row.query_context, shown)
            thinking, content = judge(tokenizer, model, prompt, args.max_new_tokens, args.temperature)
            answer = parse_answer(content, len(shown))
            if answer is None:
                invalid_rows.append((row.source_row, content[:80]))

        chosen = shown[answer - 1] if answer else None
        shown_all.append(json.dumps(shown, ensure_ascii=False))
        thinking_all.append(thinking)
        answers.append(answer)
        titles.append(chosen["title"] if chosen else None)
        years.append(chosen["year"] if chosen else None)


        if i % args.save_every == 0 and i < total:
            save(attach_results(selected, shown_all, thinking_all, answers, titles, years), args.output)

    result = attach_results(selected, shown_all, thinking_all, answers, titles, years)
    save(result, args.output)
    print(f"Saved {len(result)} rows to {args.output}")

    if invalid_rows:
        print(f"WARNING: {len(invalid_rows)} rows had no valid number in the LLM output (LLM_answer left empty):")
        for src, snippet in invalid_rows:
            print(f"  source_row={src}: {snippet!r}")


if __name__ == "__main__":
    main()
