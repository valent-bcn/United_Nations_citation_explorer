#!/usr/bin/env python3
"""
Run the trained instrument NER model on documents that contain NO known
instrument/convention/protocol alias according to the same PhraseMatcher logic used
in token_train.py.

Defaults:
    MODEL_DIR = "./checkpoints/checkpoint-1792"
    SCORE_THRESHOLD = 0.88 #Empirical
    OUTPUT_CSV = "./unknown_instrument_predictions.csv"

The script:
1. Loads the same resolution corpus and the same instrument/reference tables.
2. Builds the same alias dictionary.
3. Re-runs PhraseMatcher to split documents into KNOWN and UNKNOWN.
4. Runs the trained NER model on UNKNOWN documents only.
5. Keeps INSTRUMENT predictions with score >= SCORE_THRESHOLD.
6. Prints found titles/spans to the screen.
7. Saves the predictions to CSV.

Output columns:
    doc_id
    predicted_title
    score
    start_char
    end_char
"""

import argparse
import os
import re

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = "./checkpoints/checkpoint-1792"
DEFAULT_SCORE_THRESHOLD = 0.70
DEFAULT_OUTPUT_CSV = "./unknown_instrument_predictions.csv"

RESOLUTION_CSV = "../resolutions/ga_resolutions_1946_2019.csv"
WIKI_CSV = "../treaties/wiki-treaties_formatted.csv"
OHCHR_CSV = "../ohchr_instruments/ohchr_instruments_detailed-instit.csv"
UNO_CSV = "../treaties/UNO-Treaties.csv"
UNESCO_CSV = "../unesco_instruments/UNESCO_legal_instruments_detail.csv"
CONV_PROT_REC_CSV = "../conv-prot-rec/conventions-protocols-recommendations.csv"


# ---------------------------------------------------------------------------
# spaCy
# ---------------------------------------------------------------------------

nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# 1. Same alias generation as token_train.py
# ---------------------------------------------------------------------------

def build_aliases(df_instrument: pd.DataFrame) -> dict:
    """Build {lowercase_alias: entity_id} from the instrument reference tables."""
    aliases = {}

    for idx, row in (df_instrument.iterrows()):
        entity_id = idx
        variants = set()

        title = row.get("title")
        if pd.notna(title):
            title = str(title).strip()
            if title:
                variants.add(title)

        alt = row.get("alternative_name")
        if pd.notna(alt):
            for part in str(alt).split(";"):
                v = part.strip()
                if v:
                    variants.add(v)

        # Same year-removal logic as token_train.py
        for v in list(variants):
            no_year = re.sub(r"\s*\(?\b(18|19|20)\d{2}\b\)?", "", v).strip()
            if no_year and no_year != v:
                variants.add(no_year)

        for v in variants:
            aliases[v.lower().strip()] = entity_id

    return aliases


# ---------------------------------------------------------------------------
# 2. Load the same reference tables as token_train.py
# ---------------------------------------------------------------------------

def load_instrument_reference_tables() -> pd.DataFrame:
    cols = ["title", "year", "alternative_name"]

    df_wiki = pd.read_csv(WIKI_CSV)
    df_wiki = df_wiki.rename(
        columns={
            "name": "title",
            "cleaned_note": "alternative_name",
        }
    )[cols]

    df_ohchr = pd.read_csv(OHCHR_CSV)
    df_ohchr["year"] = pd.to_datetime(
        df_ohchr["adoption_date"],
        format="%d %B %Y"
    ).dt.year
    df_ohchr["alternative_name"] = ""
    df_ohchr = df_ohchr[cols]

    df_uno = pd.read_csv(UNO_CSV)
    df_uno["year"] = pd.to_datetime(
        df_uno["date"],
        format="%d %B %Y"
    ).dt.year
    df_uno["alternative_name"] = ""
    df_uno = df_uno[cols]

    df_unesco = pd.read_csv(UNESCO_CSV)
    df_unesco["year"] = pd.to_datetime(
        df_unesco["date"],
        format="%d %B %Y"
    ).dt.year
    df_unesco["alternative_name"] = ""
    df_unesco = df_unesco[cols]

    df_conv_prot_rec = pd.read_csv(CONV_PROT_REC_CSV)
    df_conv_prot_rec["alternative_name"] = ""

    df_conv_prot_rec = df_conv_prot_rec[cols]

    df_instrument = pd.concat(
        [
            df_wiki,
            df_ohchr,
            df_uno,
            df_unesco,
            df_conv_prot_rec,
        ],
        ignore_index=True,
    )

    df_instrument.drop_duplicates(inplace=True)
    return df_instrument


# ---------------------------------------------------------------------------
# 3. Re-run PhraseMatcher and split documents into KNOWN / UNKNOWN
# ---------------------------------------------------------------------------

def split_known_unknown(
    df_res: pd.DataFrame,
    aliases: dict,
):
    """
    Reproduce the same document-level KNOWN/UNKNOWN split as token_train.py.

    KNOWN:
        at least one alias is found in the document.

    UNKNOWN:
        no alias is found anywhere in the document.

    Returns:
        known_docs, unknown_docs
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(alias) for alias in aliases.keys()]
    matcher.add("INSTRUMENT", patterns)

    known_docs = []
    unknown_docs = []

    for _, row in df_res.iterrows():
        text = str(row["content"])
        doc = nlp(text)
        matches = matcher(doc)

        if not matches:
            unknown_docs.append(
                {
                    "doc_id": row["id"],
                    "text": text,
                }
            )
        else:
            known_docs.append(
                {
                    "doc_id": row["id"],
                    "text": text,
                }
            )

    return known_docs, unknown_docs


# ---------------------------------------------------------------------------
# 4. Load model
# ---------------------------------------------------------------------------

def load_ner_pipeline(model_dir: str):
    """Load the trained token-classification model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

    return ner_pipe


# ---------------------------------------------------------------------------
# 5. Run model on UNKNOWN documents
# ---------------------------------------------------------------------------

def infer_unknown(
    unknown_docs,
    ner_pipe,
    score_threshold: float,
):
    """
    Run the model over UNKNOWN documents.

    Keeps only predictions:
        entity_group == "INSTRUMENT"
        score >= score_threshold

    Returns a DataFrame of predicted instrument spans.
    """
    rows = []

    total = len(unknown_docs)

    for i, d in enumerate(unknown_docs, start=1):
        predictions = ner_pipe(d["text"])

        for p in predictions:
            entity_group = str(p.get("entity_group", ""))
            score = float(p.get("score", 0.0))

            if entity_group != "INSTRUMENT":
                continue

            if score < score_threshold:
                continue

            start_char = int(p["start"])
            end_char = int(p["end"])
            predicted_title = d["text"][start_char:end_char].strip()

            if not predicted_title:
                continue

            rows.append(
                {
                    "doc_id": d["doc_id"],
                    "predicted_title": predicted_title,
                    "score": score,
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )

        if i % 25 == 0 or i == total:
            print(f"Processed UNKNOWN documents: {i}/{total}")

    if not rows:
        return pd.DataFrame(
            columns=[
                "doc_id",
                "predicted_title",
                "score",
                "start_char",
                "end_char",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=["score", "doc_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run instrument NER on the UNKNOWN documents."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_DIR,
        help=f"Model/checkpoint directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=f"Minimum model score (default: {DEFAULT_SCORE_THRESHOLD})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--all-docs",
        action="store_true",
        help=(
            "Use the full resolutions CSV. Without this flag, reproduce "
            "token_train.py exactly and use df_res.tail(500)."
        ),
    )

    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0.0 and 1.0")

    if not os.path.isdir(args.model):
        raise FileNotFoundError(
            f"Model checkpoint not found: {args.model}"
        )

    print("Loading resolution corpus...")
    df_res = pd.read_csv(RESOLUTION_CSV)
    df_res.rename(columns={"res_id2": "id"}, inplace=True)

    # Reproduce the original token_train.py behavior by default.
    if not args.all_docs:
        df_res = df_res.sample(n=1000) #TODO: change this when run in server

    print(f"Resolution documents loaded: {len(df_res)}")

    print("Loading instrument/reference tables...")
    df_instrument = load_instrument_reference_tables()
    print(f"Reference rows after concatenation/deduplication: {len(df_instrument)}")

    print("Building aliases...")
    aliases = build_aliases(df_instrument)
    print(f"Aliases: {len(aliases)}")

    print("Running PhraseMatcher and rebuilding KNOWN / UNKNOWN...")
    known_docs, unknown_docs = split_known_unknown(df_res, aliases)

    print(f"KNOWN documents:   {len(known_docs)}")
    print(f"UNKNOWN documents: {len(unknown_docs)}")

    if not unknown_docs:
        print("No UNKNOWN documents found. Nothing to infer.")
        return

    print(f"Loading model: {args.model}")
    ner_pipe = load_ner_pipeline(args.model)

    print(
        f"Running NER on UNKNOWN documents "
        f"(score threshold = {args.threshold:.2f})..."
    )

    result_df = infer_unknown(
        unknown_docs=unknown_docs,
        ner_pipe=ner_pipe,
        score_threshold=args.threshold,
    )

    print("\n" + "=" * 80)
    print("FOUND TITLES")
    print("=" * 80)

    if result_df.empty:
        print("No instruments titles found above the threshold.")
    else:
        for _, row in result_df.iterrows():
            print(
                f"[score={row['score']:.4f}] "
                f"doc_id={row['doc_id']}  "
                f"{row['predicted_title']}"
            )

    result_df.to_csv(args.output, index=False)

    print("\n" + "=" * 80)
    print(f"Saved {len(result_df)} predictions to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
