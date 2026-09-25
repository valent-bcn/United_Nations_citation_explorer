"""
GLiNER vs fine-tuned BERT NER: false-negative discovery on the known partition.

Goal
----
The BERT token classifier (fine-tuned on the weakly-labeled INSTRUMENT spans,
see token_train_gliner.py) is run over `partition_known.csv`. Separately,
GLiNER-large (English, zero-shot) is run over the same texts with a fixed set
of instrument-like tags. For every GLiNER entity that does NOT overlap
(character-span-wise) with any BERT prediction scoring >= 0.90, we keep it as
a *candidate false negative*: something the fine-tuned model likely missed,
that GLiNER caught at >= 0.80.

We do NOT dump every GLiNER entity — only the ones with no BERT counterpart.

Usage
-----
    pip install gliner
    python gliner_infer_candidates.py

Requires the fine-tuned checkpoint at ./checkpoints/checkpoint-1792 and the
partition produced by token_train_gliner.py's get_or_build_partition() at
./data/partition/partition_known.csv.
"""

import os

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from gliner import GLiNER

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BERT_CHECKPOINT_DIR = "./checkpoints/checkpoint-1792"
BERT_SCORE_THRESHOLD = 0.90

GLINER_MODEL_NAME = "urchade/gliner_small-v2.1"
GLINER_SCORE_THRESHOLD = 0.85
TAGS = ["treaty", "convention", "agreement", "pact", "protocol", "charter", "declaration"]

KNOWN_PARTITION_PATH = "./data/partition/partition_known.csv"
OUT_DIR = "./data/gliner"
OUT_FILENAME = "bert_false_negatives.csv"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_known_docs(path=KNOWN_PARTITION_PATH):
    """
    Load partition_known.csv as produced by save_partition() in
    token_train_gliner.py. We only need doc_id/text here — the weak-label
    `spans` column is not used, since we compare *model inference* (BERT vs
    GLiNER), not the original silver labels.
    """
    df = pd.read_csv(path)
    return [{"doc_id": row["doc_id"], "text": row["text"]} for _, row in df.iterrows()]


def load_bert_ner(checkpoint_dir=BERT_CHECKPOINT_DIR):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )


def load_gliner(model_name=GLINER_MODEL_NAME):
    return GLiNER.from_pretrained(model_name)


# ---------------------------------------------------------------------------
# Per-doc inference
# ---------------------------------------------------------------------------

def bert_predict_spans(ner_pipe, text, score_threshold=BERT_SCORE_THRESHOLD, max_length=512):
    """
    Returns [(start_char, end_char, score, word), ...] for BERT entities
    scoring >= score_threshold.

    Note: the pipeline truncates to max_length tokens by default, so very
    long resolutions may have their tail unscanned. If that matters for your
    corpus, chunk the text yourself before calling this.
    """
    try:
        ents = ner_pipe(text, tokenizer_kwargs={"truncation": True, "max_length": max_length})
    except TypeError:
        # older transformers versions don't accept tokenizer_kwargs on __call__
        ents = ner_pipe(text)

    return [
        (e["start"], e["end"], e["score"], e["word"])
        for e in ents
        if e["score"] >= score_threshold
    ]


def gliner_predict_spans(gliner_model, text, labels=TAGS, threshold=GLINER_SCORE_THRESHOLD):
    """Returns [(start_char, end_char, score, label, text), ...] from GLiNER."""
    ents = gliner_model.predict_entities(text, labels, threshold=threshold)
    return [(e["start"], e["end"], e["score"], e["label"], e["text"]) for e in ents]


def _spans_overlap(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def find_gliner_only_entities(bert_spans, gliner_spans):
    """
    Keep only the GLiNER entities that don't character-overlap any BERT span.
    These are the candidate false negatives of the fine-tuned BERT model.
    """
    bert_ranges = [(s, e) for s, e, *_ in bert_spans]
    missed = []
    for g_start, g_end, g_score, g_label, g_text in gliner_spans:
        if not any(_spans_overlap(g_start, g_end, b_start, b_end) for b_start, b_end in bert_ranges):
            missed.append({
                "start": g_start,
                "end": g_end,
                "score": g_score,
                "label": g_label,
                "text": g_text,
            })
    return missed


# ---------------------------------------------------------------------------
# Corpus-level run
# ---------------------------------------------------------------------------

def run_gliner_vs_bert(known_docs, ner_pipe, gliner_model,
                        bert_threshold=BERT_SCORE_THRESHOLD,
                        gliner_threshold=GLINER_SCORE_THRESHOLD,
                        tags=TAGS):
    rows = []
    for d in tqdm(known_docs, desc="BERT vs GLiNER"):
        doc_id, text = d["doc_id"], d["text"]
        if not isinstance(text, str) or not text.strip():
            continue

        bert_spans = bert_predict_spans(ner_pipe, text, score_threshold=bert_threshold)
        gliner_spans = gliner_predict_spans(gliner_model, text, labels=tags, threshold=gliner_threshold)
        missed = find_gliner_only_entities(bert_spans, gliner_spans)

        for m in missed:
            rows.append({
                "doc_id": doc_id,
                "start": m["start"],
                "end": m["end"],
                "gliner_text": m["text"],
                "gliner_label": m["label"],
                "gliner_score": m["score"],
            })

    return pd.DataFrame(
        rows, columns=["doc_id", "start", "end", "gliner_text", "gliner_label", "gliner_score"]
    )


def save_candidates(df, out_dir=OUT_DIR, filename=OUT_FILENAME):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    known_docs = load_known_docs()
    print(f"Loaded {len(known_docs)} known docs from {KNOWN_PARTITION_PATH}")

    print(f"Loading fine-tuned BERT NER from {BERT_CHECKPOINT_DIR} ...")
    ner_pipe = load_bert_ner()

    print(f"Loading GLiNER ({GLINER_MODEL_NAME}) zero-shot with tags={TAGS} ...")
    gliner_model = load_gliner()

    candidates_df = run_gliner_vs_bert(known_docs, ner_pipe, gliner_model)
    print(
        f"GLiNER entities (score>={GLINER_SCORE_THRESHOLD}) with no overlapping "
        f"BERT prediction (score>={BERT_SCORE_THRESHOLD}): {len(candidates_df)}"
    )

    out_path = save_candidates(candidates_df)
    print(f"Saved candidate false negatives to {out_path}")