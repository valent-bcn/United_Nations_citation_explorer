"""
ICU server version.
Weakly-supervised NER pipeline for detecting agreement/treaty/convention aka.
Instrument names in UN General Assembly resolution text.

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
       spaCy Matcher (lexical patterns like "treaty of ...", "convention on
       ..."), then check how many of those candidates the fine-tuned model
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
            entity_id = aliases.get(alias_text)
            spans.append((span.start_char, span.end_char, entity_id))

        known_docs.append({"doc_id": row["id"], "text": row["content"], "spans": spans})

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


def train_token_classifier(bio_examples, model_name="bert-base-uncased", output_dir="./treaty-ner"):
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
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,  # eval doesn't need grads, can go higher than train
        gradient_accumulation_steps=2,  # raise this instead of batch size if you hit OOM
        dataloader_num_workers=4,  # parallel batch loading, keeps GPU fed
        group_by_length=True,  # buckets similar-length sequences -> less padding waste
        fp16=True,  # or bf16=True on Ampere+ (A100, RTX 30/40xx, H100)
        num_train_epochs=10,
        weight_decay=0.005,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
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
# 5. "SILVER" TEST SET ON DOCS WITHOUT ANY KNOWN MATCH (regex/rules)
# ---------------------------------------------------------------------------

def build_silver_candidates(unknown_docs):
    """
    Detect instrument candidates via lexical pattern in docs where
    the gazetteer found NO match. Used to measure whether the model
    generalises to names outside the original alias list.
    """
    matcher = Matcher(nlp.vocab)
    keywords = ["treaty", "convention", "agreement", "pact", "protocol", "charter"]
    pattern = [
        {"LOWER": {"IN": keywords}},
        {"LOWER": {"IN": ["of", "on", "of the"]}, "OP": "?"},
        {"IS_ALPHA": True, "OP": "+"},
        {"LOWER": {"IN": ["in", "of"]}, "OP": "?"},
        {"POS": "PROPN", "OP": "+"},
    ]
    matcher.add("INSTRUMENT_LIKE", [pattern])

    silver = []
    for d in unknown_docs:
        doc = nlp(d["text"])

        for sent in doc.sents:
            matches = matcher(sent)

            for match_id, start, end in matches:
                span = sent[start:end]

                silver.append({
                    "doc_id": d["doc_id"],
                    "text": d["text"],
                    "candidate_span": span.text,
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                })
    return pd.DataFrame(silver)


# ---------------------------------------------------------------------------
# 6. EVALUATE THE GENERALISATION: Run the model on unknown_docs and check the predictions against the silver set.
# ---------------------------------------------------------------------------

def evaluate_generalization(trainer, tokenizer, id2label, unknown_docs, silver_df):
    """For each silver candidate span, check whether the fine-tuned model's
    NER pipeline predicted an overlapping span, and report the hit rate."""
    ner_pipe = pipeline(
        "ner", model=trainer.model, tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

    rows = []
    for d in unknown_docs:
        preds = ner_pipe(d["text"])
        gold_spans = silver_df[silver_df["doc_id"] == d["doc_id"]]
        for _, g in gold_spans.iterrows():
            hit = any(
                p["start"] < g["end_char"] and p["end"] > g["start_char"]
                for p in preds
            )
            rows.append({"doc_id": d["doc_id"], "candidate": g["candidate_span"], "detected_by_model": hit})

    result_df = pd.DataFrame(rows)
    recall_proxy = result_df["detected_by_model"].mean() if len(result_df) else float("nan")
    print(f"Recall proxy over silver candidates (not seen in training): {recall_proxy:.2%}")
    return result_df


def evaluate_test_split(trainer, tokenizer, id2label, test_dataset, raw_test_examples):
    """Run trainer.predict on the held-out test split and print/return seqeval metrics."""
    predictions, labels, _ = trainer.predict(test_dataset)
    preds = np.argmax(predictions, axis=2)

    true_labels, true_preds = [], []
    for pred_row, label_row in zip(preds, labels):
        seq_labels, seq_preds = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            seq_labels.append(id2label[l])
            seq_preds.append(id2label[p])
        true_labels.append(seq_labels)
        true_preds.append(seq_preds)

    print(classification_report(true_labels, true_preds))
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


# ---------------------------------------------------------------------------
# DATA LOAD & RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_res = pd.read_csv("/home/user/branes/NER-data/ga_resolutions_1946_2019.csv")
    df_res.rename(columns={"res_id2": "id"}, inplace=True)

    cols = ["title", "year", "alternative_name"]

    df_wiki = pd.read_csv("/home/user/branes/NER-data/wiki-treaties_formatted.csv")

    df_ohchr = pd.read_csv("/home/user/branes/NER-data/ohchr_instruments_detailed-instit.csv")
    df_ohchr["year"] = pd.to_datetime(df_ohchr["adoption_date"], format="%d %B %Y").dt.year

    df_wiki = df_wiki.rename(columns={"name": "title", "cleaned_note": "alternative_name"})[cols]

    df_ohchr["alternative_name"] = ""  # it does not have an alias column, we set this as null
    df_ohchr = df_ohchr[cols]

    df_uno = pd.read_csv("/home/user/branes/NER-data/UNO-Treaties.csv")
    df_uno["year"] = pd.to_datetime(df_uno["date"], format="%d %B %Y").dt.year
    df_uno["alternative_name"] = ""

    df_unesco = pd.read_csv("/home/user/branes/NER-data/UNESCO_legal_instruments_detail.csv")
    df_unesco["year"] = pd.to_datetime(df_unesco["date"], format="%d %B %Y").dt.year
    df_unesco["alternative_name"] = ""

    df_conv_prot_rec = pd.read_csv("/home/user/branes/NER-data/conventions-protocols-recommendations.csv")
    df_conv_prot_rec["alternative_name"] = ""

    df_instrument = pd.concat([df_wiki, df_ohchr, df_uno, df_unesco, df_conv_prot_rec], ignore_index=True)
    df_instrument.drop_duplicates(inplace=True)

    aliases = build_aliases(df_instrument)
    known_docs, unknown_docs = weak_label_corpus(df_res, aliases)
    print(f"Docs with a known match: {len(known_docs)} | without match: {len(unknown_docs)}")

    bio_examples = to_bio_examples(known_docs)
    trainer, tokenizer, id2label = train_token_classifier(bio_examples)

    silver_df = build_silver_candidates(unknown_docs)
    if not silver_df.empty:
        evaluate_generalization(trainer, tokenizer, id2label, unknown_docs, silver_df)
