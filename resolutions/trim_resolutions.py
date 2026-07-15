import pandas as pd
import os
import json
import re

# ----------------------------
# SETTINGS
# ----------------------------
RESOLUTIONS_CSV = "./ga_resolutions_1946_2019.csv"
MAX_CHAR_LENGTH = 3100

res_df = pd.read_csv(RESOLUTIONS_CSV)


# ----------------------------
# SENTENCE SPLITTING
# ----------------------------
def split_into_sentences(text):
    """
    Split text into sentences on '.', '!', ';' or '?', but:
      - never split while inside parentheses ( ... )
      - never split on a '.' that is part of a decimal number (e.g. 1.0)
    Returns a list of sentence strings (whitespace-trimmed), each still
    ending with its terminal punctuation when applicable.
    """
    if not isinstance(text, str) or text == "":
        return [""]

    sentences = []
    buf = []
    depth = 0  # parenthesis nesting depth
    n = len(text)

    for i, ch in enumerate(text):
        buf.append(ch)

        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)

        if ch in ".!?;" and depth == 0:
            prev_ch = text[i - 1] if i > 0 else ""
            next_ch = text[i + 1] if i + 1 < n else ""

            # Skip decimal numbers like "1.0" -> digit '.' digit
            if ch == "." and prev_ch.isdigit() and next_ch.isdigit():
                continue

            # Skip abbreviation-like single-letter dots, e.g. "U.S." pattern
            # (only treat as sentence end if next non-space char is empty,
            # uppercase, a digit-start of new sentence, or quote)
            # Look ahead to next non-space character
            j = i + 1
            while j < n and text[j] == " ":
                j += 1
            next_non_space = text[j] if j < n else ""

            is_end_of_text = (j >= n)
            next_looks_like_new_sentence = (
                next_non_space == "" or
                next_non_space.isupper() or
                next_non_space in "\"'“”‘’(" or
                next_non_space.isdigit()
            )

            if is_end_of_text or next_looks_like_new_sentence:
                sentence = "".join(buf).strip()
                if sentence:
                    sentences.append(sentence)
                buf = []

    # Any trailing leftover text (no terminal punctuation)
    remainder = "".join(buf).strip()
    if remainder:
        sentences.append(remainder)

    if not sentences:
        sentences = [text.strip()]

    return sentences


def hard_split(text, max_len):
    """
    Fallback: split an over-long single sentence into <= max_len chunks
    on whitespace boundaries, so we never cut mid-word if avoidable.
    """
    chunks = []
    while len(text) > max_len:
        cut = text.rfind(" ", 0, max_len)
        if cut == -1 or cut == 0:
            cut = max_len  # no space found, hard cut
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def chunk_content(text, max_len):
    """
    Pack sentences into chunks whose length does not exceed max_len.
    A sentence longer than max_len on its own is hard-split as a fallback.
    """
    sentences = split_into_sentences(text)

    chunks = []
    current = ""

    for sent in sentences:
        if len(sent) > max_len:
            # flush current buffer first
            if current:
                chunks.append(current.strip())
                current = ""
            # hard split the oversized sentence itself
            chunks.extend(hard_split(sent, max_len))
            continue

        candidate = (current + " " + sent).strip() if current else sent

        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = sent

    if current:
        chunks.append(current.strip())

    if not chunks:
        chunks = [text if isinstance(text, str) else ""]

    return chunks


# ----------------------------
# BUILD TRIMMED DATAFRAME
# ----------------------------
rows = []

for _, row in res_df.iterrows():
    res_id = row["res_id2"]
    content = row["content"]

    if len(content) <= MAX_CHAR_LENGTH:
        new_row = row.to_dict()
        new_row["res_id2"] = res_id
        new_row["content"] = content
        new_row["part"] = 1
        rows.append(new_row)
    else:
        parts = chunk_content(content, MAX_CHAR_LENGTH)
        for idx, part_text in enumerate(parts, start=1):
            new_row = row.to_dict()   # copies all original columns, incl. res_id
            new_row["res_id2"] = res_id
            new_row["content"] = part_text
            new_row["part"] = idx
            rows.append(new_row)

trimmed_df = pd.DataFrame(rows)

# Optional: reorder columns so res_id / part / content are easy to inspect
cols = list(trimmed_df.columns)
if "part" in cols:
    cols.remove("part")
    # place 'part' right after 'res_id' if present, else at the end
    if "res_id2" in cols:
        insert_at = cols.index("res_id2") + 1
    else:
        insert_at = len(cols)
    cols.insert(insert_at, "part")
    trimmed_df = trimmed_df[cols]

# ----------------------------
# SAVE
# ----------------------------
OUTPUT_CSV = f"./ga_resolutions_1946_2019_{MAX_CHAR_LENGTH}-trim.csv"
trimmed_df.to_csv(OUTPUT_CSV, index=False)
print(trimmed_df.tail(20))

print(f"Original rows: {len(res_df)}")
print(f"Trimmed rows:  {len(trimmed_df)}")
print(f"Saved to: {OUTPUT_CSV}")