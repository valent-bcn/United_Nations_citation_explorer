"""
Resolves NER-extracted entity mentions (e.g. label == "INSTRUMENT") against a
canonical catalogue of legal instruments (treaties, conventions, resolutions,
recommendations, ...) using a Qwen3 CrossEncoder reranker.

Pipeline
--------
1. Build the candidate catalogue `df_instrument` exactly as in the notebook
   snippet (wiki treaties + OHCHR + UNO treaties + UNESCO + conv/prot/rec).
2. Load the resolutions dataframe `df_res`, which holds the *full text* of
   each document, keyed by `id`.
3. Load the entities CSV (schema: id,text,score,start,end,label).
4. For every entity, pull the resolution text by `id`, slice a window of
   `--window` characters before/after the [start, end) span, and wrap the
   entity itself in square brackets, e.g.:
       "...adoption of the [Geneva Protocol] concerning the..."
5. Rerank the candidate catalogue documents (title + year) against that
   context string with Qwen3-Reranker, and keep the top-k matches.
6. Write a result CSV with the best match (and optionally the top-k) per
   entity.

Assumptions you should double check
------------------------------------
* `df_res` is assumed to have a text column. Its name is NOT specified in
  the prompt, so it's controlled by `--res-text-col` (default: "text").
  If your resolutions CSV uses a different column (e.g. "full_text",
  "resolution_text", "content"), pass it explicitly.
* `df_res["id"]` and the entities CSV's `id` column are both cast to `str`
  before joining, since ids like "3319 (xxix)" are not numeric.
* The catalogue document string is built as
      f"{title} Year: {year}"
  (interpreted from "as string + Year : {title}" in the request, which
  looks like a typo for `{year}`; change `build_document_text` if you meant
  something else).
* Only entities whose `label` matches `--label-filter` (default
  "INSTRUMENT") are resolved; everything else is skipped. Pass
  `--label-filter ""` to disable filtering.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from sentence_transformers import CrossEncoder

import pandas as pd
from tqdm import tqdm


# --------------------------------------------------------------------------
# 1. Candidate catalogue (df_instrument) -- same logic as the notebook cell
# --------------------------------------------------------------------------
def build_instrument_catalogue(args: argparse.Namespace) -> pd.DataFrame:
    cols = ["title", "year", "alternative_name"]

    df_wiki = pd.read_csv(args.wiki_treaties)
    df_wiki = df_wiki.rename(
        columns={"name": "title", "cleaned_note": "alternative_name"}
    )[cols]

    df_ohchr = pd.read_csv(args.ohchr)
    df_ohchr["year"] = pd.to_datetime(
        df_ohchr["adoption_date"], format="%d %B %Y", errors="coerce"
    ).dt.year
    df_ohchr["alternative_name"] = ""
    df_ohchr = df_ohchr[cols]

    df_uno = pd.read_csv(args.uno_treaties)
    df_uno["year"] = pd.to_datetime(
        df_uno["date"], format="%d %B %Y", errors="coerce"
    ).dt.year
    df_uno["alternative_name"] = ""
    df_uno = df_uno[cols]

    df_unesco = pd.read_csv(args.unesco)
    df_unesco["year"] = pd.to_datetime(
        df_unesco["date"], format="%d %B %Y", errors="coerce"
    ).dt.year
    df_unesco["alternative_name"] = ""
    df_unesco = df_unesco[cols]

    df_conv_prot_rec = pd.read_csv(args.conv_prot_rec)
    df_conv_prot_rec["alternative_name"] = ""
    df_conv_prot_rec = df_conv_prot_rec[cols]

    df_instrument = pd.concat(
        [df_wiki, df_ohchr, df_uno, df_unesco, df_conv_prot_rec],
        ignore_index=True,
    )

    # Drop rows with no title at all, they can't be matched to anything.
    df_instrument = df_instrument.dropna(subset=["title"]).reset_index(drop=True)
    return df_instrument


def build_document_text(row: pd.Series) -> str:
    """Text representation of a candidate instrument fed to the reranker."""
    title = str(row["title"]).strip()
    year = row.get("year", "")
    year = "" if pd.isna(year) else str(int(year)) if str(year) != "" else ""
    return f"{title} Year: {year}" if year else title


# --------------------------------------------------------------------------
# 2. Resolutions dataframe (df_res) -- source of full text per id
# --------------------------------------------------------------------------
def load_resolutions(args: argparse.Namespace) -> pd.DataFrame:
    df_res = pd.read_csv(args.ga_resolutions)
    df_res = df_res.rename(columns={"res_id2": "id"})

    if args.res_text_col not in df_res.columns:
        raise ValueError(
            f"Column '{args.res_text_col}' not found in {args.ga_resolutions}. "
            f"Available columns: {list(df_res.columns)}. "
            f"Pass the correct column via --res-text-col."
        )

    df_res["id"] = df_res["id"].astype(str).str.strip()
    return df_res


# --------------------------------------------------------------------------
# 3. Entities CSV
# --------------------------------------------------------------------------
def load_entities(args: argparse.Namespace) -> pd.DataFrame:
    df_ent = pd.read_csv(args.entities)
    required = {"id", "text", "score", "start", "end", "label"}
    missing = required - set(df_ent.columns)
    if missing:
        raise ValueError(f"Entities CSV is missing columns: {missing}")

    df_ent["id"] = df_ent["id"].astype(str).str.strip()
    df_ent["start"] = df_ent["start"].astype(int)
    df_ent["end"] = df_ent["end"].astype(int)

    return df_ent


# --------------------------------------------------------------------------
# 4. Context window builder
# --------------------------------------------------------------------------
def build_context_query(res_text: str, start: int, end: int, window: int) -> str:
    """Return the text around [start, end) with the entity wrapped in [ ]."""
    n = len(res_text)
    start = max(0, min(start, n))
    end = max(start, min(end, n))

    ctx_start = max(0, start - window)
    ctx_end = min(n, end + window)

    before = res_text[ctx_start:start]
    entity = res_text[start:end]
    after = res_text[end:ctx_end]

    return f"{before}[{entity}]{after}"


# --------------------------------------------------------------------------
# 5. Reranking
# --------------------------------------------------------------------------
def load_reranker(model_name: str):

    model = CrossEncoder(
        model_name,
        prompts={
            "classification": "Classify: which document is being mentioned in the query. The titles is inside [], watch out the year and city."},
        default_prompt_name="classification",
    )
    return model


def resolve_entities(
    df_ent: pd.DataFrame,
    df_res: pd.DataFrame,
    df_instrument: pd.DataFrame,
    model,
    window: int,
    top_k: int,
    batch_size: int,
) -> pd.DataFrame:
    documents = [build_document_text(row) for _, row in df_instrument.iterrows()]

    # Index resolutions by id once, avoid re-filtering df_res per row.
    res_lookup = df_res.set_index("id")[args_global.res_text_col]

    results = []
    missing_ids = set()

    for _, ent in tqdm(df_ent.iterrows(), total=len(df_ent), desc="Resolving entities"):
        res_id = ent["id"]
        if res_id not in res_lookup.index:
            missing_ids.add(res_id)
            results.append(
                {
                    **ent.to_dict(),
                    "query_context": None,
                    "matched_title": None,
                    "matched_year": None,
                    "rerank_score": None,
                    "top_k_matches": None,
                }
            )
            continue

        res_text = res_lookup.loc[res_id]
        if isinstance(res_text, pd.Series):
            # Duplicate ids in df_res: just take the first non-null text.
            res_text = res_text.dropna().iloc[0] if not res_text.dropna().empty else ""
        res_text = "" if pd.isna(res_text) else str(res_text)

        query = build_context_query(res_text, ent["start"], ent["end"], window)

        rankings = model.rank(
            query,
            documents,
            top_k=top_k,
            batch_size=batch_size,
        )

        top = rankings[0]
        best_row = df_instrument.iloc[top["corpus_id"]]

        top_k_matches = [
            {
                "title": df_instrument.iloc[r["corpus_id"]]["title"],
                "year": df_instrument.iloc[r["corpus_id"]]["year"],
                "score": float(r["score"]),
            }
            for r in rankings
        ]

        results.append(
            {
                **ent.to_dict(),
                "query_context": query,
                "matched_title": best_row["title"],
                "matched_year": best_row["year"],
                "rerank_score": float(top["score"]),
                "top_k_matches": top_k_matches,
            }
        )

    if missing_ids:
        print(
            f"WARNING: {len(missing_ids)} distinct resolution id(s) from the entities CSV "
            f"were not found in df_res and were left unresolved, e.g. "
            f"{list(missing_ids)[:5]}"
        )

    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--entities",
        default="/home/user/branes/NER-data/ner_entities.csv",
        help="CSV of NER entities (id,text,score,start,end,label)",
    )

    p.add_argument(
        "--ga-resolutions",
        default="/home/user/branes/NER-data/ga_resolutions_1946_2019.csv",
        help="CSV with full resolution texts (df_res source)",
    )
    p.add_argument(
        "--res-text-col",
        default="content",
        help="Column in --ga-resolutions holding the full text",
    )

    p.add_argument(
        "--wiki-treaties",
        default="/home/user/branes/NER-data/wiki-treaties_formatted.csv",
        help="CSV of Wikipedia-formatted treaties",
    )
    p.add_argument(
        "--ohchr",
        default="/home/user/branes/NER-data/ohchr_instruments_detailed-instit.csv",
        help="CSV of OHCHR instruments",
    )
    p.add_argument(
        "--uno-treaties",
        default="/home/user/branes/NER-data/UNO-Treaties.csv",
        help="CSV of UNO treaties",
    )
    p.add_argument(
        "--unesco",
        default="/home/user/branes/NER-data/UNESCO_legal_instruments_detail.csv",
        help="CSV of UNESCO legal instruments",
    )
    p.add_argument(
        "--conv-prot-rec",
        default="/home/user/branes/NER-data/conventions-protocols-recommendations.csv",
        help="CSV of conventions, protocols, and recommendations",
    )

    p.add_argument(
        "--window",
        type=int,
        default=100,
        help="Chars of context before/after the entity span",
    )

    p.add_argument(
        "--model",
        default="Qwen/Qwen3-Reranker-8B",
        help="CrossEncoder model name",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidate instruments to keep per entity",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    p.add_argument(
        "--output",
        default="/home/user/branes/NER-data/resolved_instruments.csv",
    )

    return p.parse_args()


def main():
    global args_global
    args = parse_args()
    args_global = args

    print("Building instrument catalogue (df_instrument)...")
    df_instrument = build_instrument_catalogue(args)
    print(f"  {len(df_instrument)} candidate instruments")

    print("Loading resolutions (df_res)...")
    df_res = load_resolutions(args)
    print(f"  {len(df_res)} resolutions, text column = {args.res_text_col!r}")

    print("Loading entities...")
    df_ent = load_entities(args)
    print(f"  {len(df_ent)} entities to resolve")
    #TODO: true resize when debugged
    df_ent = df_ent[df_ent["score"]>0.9]
    df_ent = df_ent.tail(10)

    if df_ent.empty:
        print("Nothing to resolve, exiting.")
        sys.exit(0)

    print(f"Loading reranker: {args.model} ...")
    model = load_reranker(args.model)

    df_out = resolve_entities(
        df_ent=df_ent,
        df_res=df_res,
        df_instrument=df_instrument,
        model=model,
        window=args.window,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    out_path = Path(args.output)
    df_out.to_csv(out_path, index=False)
    print(f"Wrote {len(df_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
