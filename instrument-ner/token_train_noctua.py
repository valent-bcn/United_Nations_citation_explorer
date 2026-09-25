"""
Weakly-supervised NER pipeline for detecting instrument/convention/protocol names in UN
General Assembly resolution text.

Pipeline:
    1. Build a lowercase alias -> entity_id lookup table from several instrument
       reference tables (Wikipedia, OHCHR, UNO, UNESCO, conventions/protocols).
    2. Use a spaCy PhraseMatcher over the resolution corpus to weakly label
       spans that match a known alias ("known_docs"); resolutions with no
       match are kept separately ("unknown_docs").
    3. Convert the character-level spans from step 2 into token-level BIO
       tags (B-INSTRUMENT / I-INSTRUMENT / O) for HF token classification.
    4. Fine-tune a token classifier (default: bert-base-uncased) on the BIO
       data with the HF Trainer, logging seqeval precision/recall/F1 to
       stdout and to a CSV file in the training output directory at the end
       of every epoch.
    5. Build "silver" candidate spans on the unknown_docs using a rule-based
       spaCy Matcher (lexical patterns like "treaty/convention/pact of ...",
       then check how many of those candidates the fine-tuned model
       also detects, as a rough proxy for how well it generalises beyond the
       original alias list.
"""

import csv
import os
import re
from functools import partial

import numpy as np
import pandas as pd
import spacy
from datasets import Dataset
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from spacy.matcher import Matcher, PhraseMatcher
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    pipeline,
)
import json
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

LABEL_LIST = ["O", "B-INSTRUMENT", "I-INSTRUMENT"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


# ---------------------------------------------------------------------------
# 1. ALIAS GENERATION PER ENTITY
# ---------------------------------------------------------------------------

def build_aliases(df_instrument: pd.DataFrame) -> dict:
    """
    Build a {lowercase alias: alias_id} lookup from the instrument table.

    Each alias are considered as different forms of titles. Here aliases are
    not variation such as just changing the year position of the title or
    just a switch of two words.
    In any case, this helper serves the NER, not the Entity Linking,
    Which belongs to a further phase.
    """
    aliases = {}
    for idx, row in df_instrument.iterrows():
        variants = list(dict.fromkeys(
            [row["title"]] +
            (
                [p.strip() for p in row["alternative_name"].split(";") if p.strip()]
                if pd.notna(row.get("alternative_name")) else []
            )
        ))

        for i, v in enumerate(variants):
            alias_id = f"{idx}::{i}"
            aliases[v.lower().strip()] = alias_id

    return aliases

# ---------------------------------------------------------------------------
# 2. WEAK LABELING WITH PhraseMatcher
# ---------------------------------------------------------------------------

def weak_label_corpus(df_res: pd.DataFrame, aliases: dict):
    """
    df_res: columns 'id', 'content'.

    Returns:
        known_docs: list of dicts {doc_id, text, spans: [(start_char, end_char, entity_id)]}
        unknown_docs: list of dicts {doc_id, text}  (no alias detected at all)
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(alias) for alias in aliases.keys()]
    matcher.add("INSTRUMENT", patterns)

    known_docs, unknown_docs = [], []

    for _, row in df_res.iterrows():
        doc = nlp(row["content"])
        matches = matcher(doc)

        if not matches:
            unknown_docs.append({"doc_id": row["id"], "text": row["content"]})
            continue

        spans = []
        seen = set()
        matches_sorted = sorted(
            matches,
            key=lambda m: doc[m[1]:m[2]].end_char - doc[m[1]:m[2]].start_char,
            reverse=True,
        )
        for match_id, start, end in matches_sorted:
            span = doc[start:end]
            if any(span.start_char < s[1] and span.end_char > s[0] for s in seen):
                continue  # overlaps with an already accepted match
            seen.add((span.start_char, span.end_char))
            alias_text = span.text.lower().strip()
            alias_id = aliases.get(alias_text)
            spans.append((span.start_char, span.end_char, alias_id))

        known_docs.append({"doc_id": row["id"], "text": row["content"], "spans": spans})

    return known_docs, unknown_docs


def save_partition(known_docs, unknown_docs, out_dir="./data/partition"):
    os.makedirs(out_dir, exist_ok=True)

    # known docs: serialize spans (list of tuples) as JSON so they survive the round trip
    known_rows = [
        {
            "doc_id": d["doc_id"],
            "text": d["text"],
            "spans": json.dumps(d["spans"]),
        }
        for d in known_docs
    ]

    known_df = pd.DataFrame(known_rows, columns=["doc_id", "text", "spans"])
    known_path = os.path.join(out_dir, "partition_known.csv")
    known_df.to_csv(known_path, index=False)

    # unknown docs: no spans, just id + text
    unknown_rows = [{"doc_id": d["doc_id"], "text": d["text"]} for d in unknown_docs]
    unknown_df = pd.DataFrame(unknown_rows, columns=["doc_id", "text"])
    unknown_path = os.path.join(out_dir, "partition_unkown.csv")
    unknown_df.to_csv(unknown_path, index=False)

    return known_path, unknown_path

def partition_exists(out_dir="./data/partition"):
    known_path = os.path.join(out_dir, "partition_known.csv")
    unknown_path = os.path.join(out_dir, "partition_unkown.csv")
    return os.path.exists(known_path) and os.path.exists(unknown_path)


def load_partition(out_dir="./data/partition"):
    known_path = os.path.join(out_dir, "partition_known.csv")
    unknown_path = os.path.join(out_dir, "partition_unkown.csv")

    known_df = pd.read_csv(known_path)
    unknown_df = pd.read_csv(unknown_path)

    # spans were json.dumps'd from a list of tuples -> come back as list of lists,
    # so cast each span back to a tuple to match what weak_label_corpus originally produced
    known_docs = [
        {
            "doc_id": row["doc_id"],
            "text": row["text"],
            "spans": [tuple(span) for span in json.loads(row["spans"])],
        }
        for _, row in known_df.iterrows()
    ]

    unknown_docs = [
        {"doc_id": row["doc_id"], "text": row["text"]}
        for _, row in unknown_df.iterrows()
    ]

    return known_docs, unknown_docs

def get_or_build_partition(df_res, aliases, out_dir="./data/partition"):
    if partition_exists(out_dir):
        print(f"Found existing partition in {out_dir}, loading from disk...")
        known_docs, unknown_docs = load_partition(out_dir)
    else:
        print("No existing partition found, running weak_label_corpus...")
        known_docs, unknown_docs = weak_label_corpus(df_res, aliases)
        save_partition(known_docs, unknown_docs, out_dir)
    return known_docs, unknown_docs

# ---------------------------------------------------------------------------
# 3. SETTING BIO TAGS (formatting HF token classification)
# ---------------------------------------------------------------------------

def to_bio_examples(known_docs):
    """
    Convert each doc's character-level spans into tokens + BIO labels,
    using spaCy's tokenizer for the split (this gets realigned to the
    model's own tokenizer later, in align_labels / Dataset.map).
    """
    examples = []
    for d in known_docs:
        doc = nlp(d["text"])
        labels = ["O"] * len(doc)

        for start, end, _entity_id in d["spans"]:
            span = doc.char_span(start, end, alignment_mode="expand")
            if span is None:
                continue
            labels[span.start] = "B-INSTRUMENT"
            for i in range(span.start + 1, span.end):
                labels[i] = "I-INSTRUMENT"

        examples.append({
            "doc_id": d["doc_id"],
            "tokens": [t.text for t in doc],
            "ner_tags": labels,
        })
    return examples


# ---------------------------------------------------------------------------
# 4. TOKEN CLASSIFIER FINE-TUNING (HF Trainer)
# ---------------------------------------------------------------------------

def align_labels(example, tokenizer, label2id=LABEL2ID):
    """
    Tokenize with the model's tokenizer and realign word-level BIO labels
    to subword tokens. Subword continuations keep I-INSTRUMENT if the parent
    word was tagged, otherwise O; special tokens get -100 so they're
    ignored by the loss and by seqeval.
    """
    tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True)
    word_ids = tokenized.word_ids()
    aligned = []
    prev_word = None
    for wid in word_ids:
        if wid is None:
            aligned.append(-100)
        elif wid != prev_word:
            aligned.append(label2id[example["ner_tags"][wid]])
        else:
            tag = example["ner_tags"][wid]
            aligned.append(label2id["I-INSTRUMENT"] if tag != "O" else label2id["O"])
        prev_word = wid
    tokenized["labels"] = aligned
    return tokenized


def compute_metrics(eval_pred, id2label=ID2LABEL):
    """Compute seqeval precision/recall/F1 from an HF EvalPrediction."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=2)

    true_labels, true_preds = [], []
    for pred_row, label_row in zip(preds, labels):
        seq_labels, seq_preds = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:  # ignore padding / subword-continuation tokens
                continue
            seq_labels.append(id2label[l])
            seq_preds.append(id2label[p])
        true_labels.append(seq_labels)
        true_preds.append(seq_preds)

    # printed to stdout at every eval call (i.e. every epoch), not just returned
    print(classification_report(true_labels, true_preds))

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


class MetricsCSVLogger(TrainerCallback):
    """
    Trainer callback that appends the eval metrics dict to a CSV file inside
    the run's output directory every time evaluation runs (i.e. at the end
    of each epoch, given eval_strategy="epoch").
    """

    def __init__(self, output_dir, filename="eval_metrics.csv"):
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, filename)
        self._header_written = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        row = {"epoch": state.epoch, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


def train_token_classifier(bio_examples,
                           model_name="bert-base-uncased",
                           output_dir="./checkpoints/"):
    """
    Fine-tune a token classification model to detect INSTRUMENT spans with the
    HF Trainer. Per-epoch seqeval metrics are printed to stdout and appended
    to <output_dir>/eval_metrics.csv via MetricsCSVLogger.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds = Dataset.from_list(bio_examples)
    ds = ds.train_test_split(test_size=0.15, seed=42, shuffle=True)
    ds = ds.map(
        partial(align_labels, tokenizer=tokenizer),
        remove_columns=["tokens", "ner_tags", "doc_id"],
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID
    )
    collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",       # print train loss at each epoch too, for comparison
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        num_train_epochs=10,
        weight_decay=0.005,
        load_best_model_at_end=True,
        metric_for_best_model="f1",     # which of the compute_metrics keys decides "best"
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,        # <-- triggers seqeval metrics each epoch
        callbacks=[MetricsCSVLogger(output_dir)],  # <-- writes those metrics to CSV each epoch
    )
    trainer.train()

    # trainer.state.log_history also has a per-epoch record of eval_precision/eval_recall/eval_f1
    # if you want to plot the evolution afterward in-memory instead of re-reading the CSV, e.g.:
    #   hist = pd.DataFrame(trainer.state.log_history)
    #   hist[hist["eval_f1"].notna()][["epoch", "eval_precision", "eval_recall", "eval_f1"]]

    return trainer, tokenizer, ID2LABEL


# ---------------------------------------------------------------------------
# DATA LOAD & RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_res = pd.read_csv("/home/user/branes/NER-data/ga_resolutions_1946_2019.csv")
    df_res.rename(columns={"res_id2": "id"}, inplace=True)

    cols = ["title", "year", "alternative_name"]

    df_wiki = pd.read_csv("/home/user/branes/NER-data/wiki-treaties_formatted.csv")
    df_wiki = df_wiki.rename(columns={"name": "title", "cleaned_note": "alternative_name"})[cols]

    df_ohchr = pd.read_csv("/home/user/branes/NER-data/ohchr_instruments_detailed-instit.csv")
    df_ohchr["year"] = pd.to_datetime(df_ohchr["adoption_date"], format="%d %B %Y").dt.year
    df_ohchr["alternative_name"] = ""  # it does not have an alias column, we set this as null
    df_ohchr = df_ohchr[cols]

    df_uno = pd.read_csv("/home/user/branes/NER-data/UNO-Treaties.csv")
    df_uno["year"] = pd.to_datetime(df_uno["date"], format="%d %B %Y").dt.year
    df_uno["alternative_name"] = ""
    df_uno = df_uno[cols]

    df_unesco = pd.read_csv("/home/user/branes/NER-data/UNESCO_legal_instruments_detail.csv")
    df_unesco["year"] = pd.to_datetime(df_unesco["date"], format="%d %B %Y").dt.year
    df_unesco["alternative_name"] = ""
    df_unesco = df_unesco[cols]

    df_conv_prot_rec = pd.read_csv("/home/user/branes/NER-data/conventions-protocols-recommendations.csv")
    df_conv_prot_rec["alternative_name"] = ""
    df_conv_prot_rec = df_conv_prot_rec[cols]

    df_instrument = pd.concat(
        [df_wiki, df_ohchr, df_uno, df_unesco, df_conv_prot_rec],
        ignore_index=True
    )

    aliases = build_aliases(df_instrument)
    known_docs, unknown_docs = get_or_build_partition(df_res, aliases)
    print(f"Docs with a known match: {len(known_docs)} | without match: {len(unknown_docs)}")

    bio_examples = to_bio_examples(known_docs)
    trainer, tokenizer, id2label = train_token_classifier(bio_examples)
