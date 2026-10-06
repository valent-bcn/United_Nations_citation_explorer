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
import argparse

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

nlp = spacy.blank("en")

# Number of texts per nlp.pipe() batch.
BATCH_SIZE = 64

LABEL_LIST = ["O", "B-INSTRUMENT", "I-INSTRUMENT"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


# ---------------------------------------------------------------------------
# 1. LIST GENERATION ACROSS ALL ENTITIES
# ---------------------------------------------------------------------------
def build_golden_list(df_instrument: pd.DataFrame) -> list[str]:
    """
    Build the backbone of the weak labelling.

    Official titles and aliases receive the same treatment. Either the
    official name or an alias can be used in training. The train/eval
    partition should therefore be based on this list rather than on
    individual entities.

    Parameters
    ----------
    df_instrument:
        DataFrame containing at least ["title", "alternative_name", ...].

    Returns
    -------
    list[str]
        Names (official titles and alternative names) that the
        PhraseMatcher should detect.
    """
    if df_instrument.empty:
        return []

    golden_list = []

    # Official titles
    titles = [
        t.strip()
        for t in df_instrument["title"].dropna()
        if t.strip()
    ]

    # Alternative names
    aliases = []
    for value in df_instrument["alternative_name"].dropna():
        aliases.extend(
            p.strip()
            for p in value.split(";")
            if p.strip()
        )

    golden_list.extend(titles)
    golden_list.extend(aliases)

    return golden_list

# ---------------------------------------------------------------------------
# 2. WEAK LABELING WITH PhraseMatcher
# ---------------------------------------------------------------------------

def weak_label_corpus(df_res: pd.DataFrame, golden_list: list[str], batch_size: int = BATCH_SIZE):
    """
    Weakly label documents using a list of instrument names.

    Parameters
    ----------
    df_res:
        DataFrame containing columns ["id", "content"].

    golden_list:
        List of official instrument names and aliases returned by
        build_golden_list().

    batch_size:
        Number of texts per nlp.pipe() batch.

    Returns
    -------
    known_docs:
        List of dictionaries:
        {
            "doc_id": ...,
            "text": ...,
            "spans": [(start_char, end_char, matched_text)]
        }

    unknown_docs:
        List of dictionaries:
        {
            "doc_id": ...,
            "text": ...
        }

    Notes
    -----
    Unlike the previous dictionary-based implementation, there is no
    entity/alias ID associated with a match. The matched text itself is
    returned.
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    # Build patterns directly from the list
    patterns = [
        nlp.make_doc(name)
        for name in golden_list
        if isinstance(name, str) and name.strip()
    ]

    matcher.add("INSTRUMENT", patterns)

    known_docs = []
    unknown_docs = []

    # Stream the corpus through nlp.pipe(). It yields docs in input order, so
    # zipping against the (id, text) rows keeps each doc paired with its id.
    rows = list(
        df_res[["id", "content"]].itertuples(index=False, name=None)
    )
    texts = (text for _, text in rows)

    for (doc_id, text), doc in zip(
        rows,
        nlp.pipe(texts, batch_size=batch_size),
    ):
        matches = matcher(doc)

        if not matches:
            unknown_docs.append({
                "doc_id": doc_id,
                "text": text,
            })
            continue

        # Sort longest matches first so that, in case of overlap,
        # the more specific/longer match is kept.
        matches_sorted = sorted(
            matches,
            key=lambda match: (
                doc[match[1]:match[2]].end_char
                - doc[match[1]:match[2]].start_char
            ),
            reverse=True,
        )

        spans = []
        seen = []

        for _, start, end in matches_sorted:
            span = doc[start:end]

            # Skip overlapping spans
            if any(
                span.start_char < seen_end
                and span.end_char > seen_start
                for seen_start, seen_end in seen
            ):
                continue

            seen.append(
                (span.start_char, span.end_char)
            )

            spans.append(
                (
                    span.start_char,
                    span.end_char,
                    span.text,
                )
            )

        known_docs.append({
            "doc_id": doc_id,
            "text": text,
            "spans": spans,
        })

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
    unknown_path = os.path.join(out_dir, "partition_unknown.csv")
    unknown_df.to_csv(unknown_path, index=False)

    return known_path, unknown_path

def partition_exists(out_dir="./data/partition"):
    known_path = os.path.join(out_dir, "partition_known.csv")
    unknown_path = os.path.join(out_dir, "partition_unknown.csv")
    return os.path.exists(known_path) and os.path.exists(unknown_path)


def load_partition(out_dir="./data/partition"):
    known_path = os.path.join(out_dir, "partition_known.csv")
    unknown_path = os.path.join(out_dir, "partition_unknown.csv")

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

def to_bio_examples(known_docs, batch_size: int = BATCH_SIZE):
    """
    Convert each doc's character-level spans into tokens + BIO labels,
    using spaCy's tokenizer for the split (this gets realigned to the
    model's own tokenizer later, in align_labels / Dataset.map).
    """
    examples = []

    # Stream the texts through nlp.pipe(); docs come back in input order.
    docs = nlp.pipe((d["text"] for d in known_docs), batch_size=batch_size)

    for d, doc in zip(known_docs, docs):
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
                           doc_years,
                           eval_from_year=2015,
                           model_name="bert-base-uncased",
                           output_dir="./checkpoints/"):
    """
    Fine-tune a token classification model to detect INSTRUMENT spans with the
    HF Trainer. Per-epoch seqeval metrics are printed to stdout and appended
    to <output_dir>/eval_metrics.csv via MetricsCSVLogger.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ex = [ex for ex in bio_examples if doc_years[ex["doc_id"]] < eval_from_year]
    eval_ex = [ex for ex in bio_examples if doc_years[ex["doc_id"]] >= eval_from_year]
    print(f"train: {len(train_ex)} | eval (>= {eval_from_year}): {len(eval_ex)}")

    align = partial(align_labels, tokenizer=tokenizer)
    cols = ["tokens", "ner_tags", "doc_id"]
    train_ds = Dataset.from_list(train_ex).map(align, remove_columns=cols)
    eval_ds = Dataset.from_list(eval_ex).map(align, remove_columns=cols)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID
    )
    collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.005,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[MetricsCSVLogger(output_dir)],
    )
    trainer.train()

    return trainer, tokenizer, ID2LABEL


# ---------------------------------------------------------------------------
# DATA LOAD & RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_res = pd.read_csv("/home/user/branes/NER-data/ga_resolutions_1946_2019.csv")
    df_res.rename(columns={"res_id2": "id"}, inplace=True)
    df_res["year"] = pd.to_datetime(
        df_res["date_c"], format="%d %B %Y", errors="coerce"
    ).dt.year

    doc_years = dict(zip(df_res["id"], df_res["year"]))

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

    aliases = build_golden_list(df_instrument)
    known_docs, unknown_docs = get_or_build_partition(df_res, aliases)
    print(f"Docs with a known match: {len(known_docs)} | without match: {len(unknown_docs)}")

    bio_examples = to_bio_examples(known_docs)
    trainer, tokenizer, id2label = train_token_classifier(bio_examples, doc_years)