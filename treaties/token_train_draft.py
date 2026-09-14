import re
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------------
# 1. ALIAS GENERATION PER ENTITY
# ---------------------------------------------------------------------------

def build_aliases(df_treaty: pd.DataFrame) -> dict:
    aliases = {}
    for idx, row in df_treaty.iterrows():
        entity_id = idx
        variants = {row["title"]}

        alt = row.get("alternative_name")
        if pd.notna(alt):
            for part in alt.split(";"):
                v = part.strip()
                if v:
                    variants.add(v)
        for v in list(variants):
            no_year = re.sub(r"\s*\(?\b(18|19|20)\d{2}\b\)?", "", v).strip()
            if no_year and no_year != v:
                variants.add(no_year)
        for v in variants:
            aliases[v.lower().strip()] = entity_id
    return aliases


# ---------------------------------------------------------------------------
# 2. WEAK LABELING WITH PhraseMatcher
# ---------------------------------------------------------------------------

def weak_label_corpus(df_res: pd.DataFrame, aliases: dict):
    """
    df_res: columnas 'id', 'content'
    Devuelve:
      known_docs: lista de dicts {doc_id, text, spans: [(start_char, end_char, entity_id)]}
      unknown_docs: lista de dicts {doc_id, text}  (sin ningún alias detectado)
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(alias) for alias in aliases.keys()]
    alias_list = list(aliases.keys())
    matcher.add("TREATY", patterns)

    known_docs, unknown_docs = [], []

    for _, row in df_res.iterrows():
        doc = nlp(row["content"])
        matches = matcher(doc)

        if not matches:
            unknown_docs.append({"doc_id": row["id"], "text": row["content"]})
            continue

        spans = []
        seen = set()
        matches_sorted = sorted(matches, key=lambda m: doc[m[1]:m[2]].end_char - doc[m[1]:m[2]].start_char, reverse=True)
        for match_id, start, end in matches_sorted:
            span = doc[start:end]
            if any(span.start_char < s[1] and span.end_char > s[0] for s in seen):
                continue  # se solapa con un match ya aceptado
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
    Convierte cada doc con spans de caracteres a tokens + labels BIO,
    usando el tokenizador de spaCy para el split (luego se realinea con
    el tokenizador del modelo en el Dataset.map).
    """
    examples = []
    for d in known_docs:
        doc = nlp(d["text"])
        labels = ["O"] * len(doc)
        char_to_tok = {tok.idx: tok.i for tok in doc}

        for start, end, _entity_id in d["spans"]:
            span = doc.char_span(start, end, alignment_mode="expand")
            if span is None:
                continue
            labels[span.start] = "B-TREATY"
            for i in range(span.start + 1, span.end):
                labels[i] = "I-TREATY"

        examples.append({
            "doc_id": d["doc_id"],
            "tokens": [t.text for t in doc],
            "ner_tags": labels,
        })
    return examples


# ---------------------------------------------------------------------------
# 4. TOKEN CLASSIFIER FINE-TUNING (HF Trainer)
# ---------------------------------------------------------------------------

def train_token_classifier(bio_examples, model_name="bert-base-uncased"):
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        DataCollatorForTokenClassification, TrainingArguments, Trainer,
    )

    label_list = ["O", "B-TREATY", "I-TREATY"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def align_labels(example):
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
                # subword continuation: mantiene I- si corresponde, o -100 si preferís ignorarlo
                tag = example["ner_tags"][wid]
                aligned.append(label2id["I-TREATY"] if tag != "O" else label2id["O"])
            prev_word = wid
        tokenized["labels"] = aligned
        return tokenized

    ds = Dataset.from_list(bio_examples)

    ds = ds.train_test_split(test_size=0.15, seed=42)
    ds = ds.map(align_labels, remove_columns=["tokens", "ner_tags", "doc_id"])

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )
    collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir="./treaty-ner",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds["train"], eval_dataset=ds["test"],
        data_collator=collator, processing_class=tokenizer,
    )
    trainer.train()
    return trainer, tokenizer, id2label


# ---------------------------------------------------------------------------
# 5. "SILVER" TEST SET ON DOCS WITHOUT ANY KNOWN MATCH (regex/reglas)
# ---------------------------------------------------------------------------

def build_silver_candidates(unknown_docs):
    """
    Detecta candidatos a tratado/convención por patrón léxico en docs donde
    el gazetteer NO encontró nada. Sirve para medir si el modelo generaliza
    a nombres fuera de la lista original.
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
    matcher.add("TREATY_LIKE", [pattern])

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
    from transformers import pipeline

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
    print(f"Recall proxy sobre candidatos silver (no vistos en train): {recall_proxy:.2%}")
    return result_df

def evaluate_test_split(trainer, tokenizer, id2label, test_dataset, raw_test_examples):
    import numpy as np
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
    df_res = pd.read_csv("../resolutions/ga_resolutions_1946_2019.csv")
    df_res.rename(columns={"res_id2": "id"}, inplace=True)
    df_res = df_res.tail(100)

    cols = ["title", "year", "alternative_name"]

    df_wiki = pd.read_csv("./wiki-treaties_formatted.csv")

    df_ohchr = pd.read_csv("../ohchr_instruments/ohchr_instruments_detailed-instit.csv")
    df_ohchr["year"] = pd.to_datetime(df_ohchr["adoption_date"], format="%d %B %Y").dt.year

    df_wiki = df_wiki.rename(columns={"name": "title", "cleaned_note": "alternative_name"})[cols]

    df_ohchr["alternative_name"] = ""  #It does not have an alias column, we set this as null
    df_ohchr = df_ohchr[cols]

    df_uno = pd.read_csv("../treaties/UNO-Treaties.csv")
    df_uno["year"] = pd.to_datetime(df_uno["date"], format="%d %B %Y").dt.year
    df_uno["alternative_name"] = ""

    df_unesco = pd.read_csv("../unesco_instruments/UNESCO_legal_instruments_detail.csv")
    df_unesco["year"] = pd.to_datetime(df_unesco["date"], format="%d %B %Y").dt.year
    df_unesco["alternative_name"] = ""

    df_conv_prot_rec = pd.read_csv("../conv-prot-rec/conventions-protocols-recommendations.csv")
    df_conv_prot_rec["alternative_name"] = ""

    df_treaty = pd.concat([df_wiki, df_ohchr, df_uno, df_unesco, df_conv_prot_rec], ignore_index=True)
    df_treaty.drop_duplicates(inplace=True)

    aliases = build_aliases(df_treaty)
    known_docs, unknown_docs = weak_label_corpus(df_res, aliases)
    print(f"Docs with a known match: {len(known_docs)} | without match: {len(unknown_docs)}")

    bio_examples = to_bio_examples(known_docs)
    trainer, tokenizer, id2label = train_token_classifier(bio_examples)

    silver_df = build_silver_candidates(unknown_docs)
    evaluate_generalization(trainer, tokenizer, id2label, unknown_docs, silver_df)

