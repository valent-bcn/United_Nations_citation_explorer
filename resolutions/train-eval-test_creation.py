import pandas as pd
import os
import json
import re
from transformers import AutoTokenizer

# ----------------------------
# SETTINGS
# ----------------------------
CITATIONS_CSV = "./ga_citations_1946_2019.csv"
RESOLUTIONS_CSV = "./ga_resolutions_1946_2019.csv"
OUTPUT_PATH = "../Qwen3/data/"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
MAX_TOKEN_LENGTH = 2826
# ----------------------------
# HELPERS
# ----------------------------
def trim_contents(text, tokenizer,
                     target_tokens: int,
                     max_tokens: int):

    # Total length
    total_tokens = len(tokenizer.encode(text, add_special_tokens=False))

    # Already short enough
    if total_tokens <= target_tokens:
        return text

    # Too long no matter what
    if total_tokens > max_tokens:
        return None

    # Find sentence endings
    sentence_endings = [m.end() for m in re.finditer(r'[.!?]', text)]

    # Character position corresponding approximately to 3000 tokens
    first_part = tokenizer.decode(
        tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    )
    char_pos_3000 = len(first_part)

    # First sentence ending after token 3000
    next_end = next((pos for pos in sentence_endings if pos >= char_pos_3000), None)

    if next_end is None:
        return None

    candidate = text[:next_end]

    candidate_tokens = len(
        tokenizer.encode(candidate, add_special_tokens=False)
    )

    if candidate_tokens <= max_tokens:
        return candidate

    return None

# ----------------------------
# SORT & SPLIT DATA
# ----------------------------
def sort_and_split(resolutions_dataframe: pd.DataFrame, citations_dataframe: pd.DataFrame):
    res_df = resolutions_dataframe.copy()
    cit_df = citations_dataframe.copy()

    # We prepare the contents forhand
    detail = res_df[["res_id2", "res_id2_unlet", "date_p", "content"]].copy()  # different for each lettered resolution (resolution sectionned by letter)
    detail["res_id2"] = detail["res_id2"].str.strip()

    # First we take the resolution ids of the ones we want to take for train+val and for test
    res_df = res_df[["res_id2_unlet", "date_p"]].drop_duplicates().reset_index(
        drop=True)  # res_id has letters, we want the whole resolution (unlet)
    res_df["res_id2_unlet"] = res_df["res_id2_unlet"].str.strip()
    len_res = len(res_df)

    # We sort in ascendent way, so we have the test set as the newest resolutions
    res_df.sort_values(by=["date_p"], inplace=True, ascending=True)

    # Filter the ids by proportion. We follow 80 for train 10 for val and 10 for test.
    train = int(len_res*0.8)
    val = int(len_res*0.9)
    train_ids = res_df[train:]["res_id2_unlet"]
    val_ids = res_df[val:]["res_id2_unlet"]
    test_ids = res_df[val:len_res]["res_id2_unlet"]

    # Work on the actual df columns that contains the recognized cites
    cit_df = cit_df[["res_id2_doc_giving_cite", "res_id2_unlet_giv", "date_p_giv", "content_giv", "res_id2_doc_receiv_cite"]]
    cit_df["res_id2_doc_receiv_cite"] = cit_df["res_id2_doc_receiv_cite"].str.strip()

    # Add the None cite cases
    no_cite = detail[~detail["res_id2"].isin(cit_df["res_id2_doc_giving_cite"])].copy()
    no_cite["cites"] = "None"

    no_cite = no_cite[["res_id2", "res_id2_unlet", "date_p", "content", "cites"]] # Following the same column order as cit_df
    no_cite.rename(columns={"res_id2": "res_id2_doc_giving_cite", "res_id2_unlet": "res_id2_unlet_giv", "date_p": "date_p_giv", "content": "content_giv", "cites": "res_id2_doc_receiv_cite"})
    cit_df = pd.concat([cit_df, no_cite])

    cit_df.sort_values(by=["date_p_giv", "res_id2_doc_giving_cite", "res_id2_doc_receiv_cite"], inplace=True, ascending=True)
    cit_df.reset_index(drop=True, inplace=True)

    # Format the id in a more formal way, including A/RES/[code]
    cit_df["res_id2_doc_giving_cite"] = "A/RES/" + cit_df["res_id2_doc_giving_cite"].str.replace(" ", "").str.upper()
    cit_df["res_id2_doc_receiv_cite"] = "A/RES/" + cit_df["res_id2_doc_receiv_cite"].str.replace(" ", "").str.upper()
    cit_df = (
        cit_df.groupby(["res_id2_doc_giving_cite", "res_id2_unlet_giv", "content_giv"], as_index=False)
        .agg(answer=("res_id2_doc_receiv_cite", lambda x: ", ".join(x)))
    )

    print(f"Citation dataframe shape: {cit_df.shape}")

    train_df = cit_df[cit_df["res_id2_unlet_giv"].isin(train_ids)]
    val_df = cit_df[cit_df["res_id2_unlet_giv"].isin(val_ids)]
    test_df = cit_df[cit_df["res_id2_unlet_giv"].isin(test_ids)]

    return train_df, val_df, test_df


def format_message_line(row):
    doc_name = getattr(row, "res_id2_doc_giving_cite", "")
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict information extraction system. "
                    "Extract only United Nations general assembly resolution codes from the input. "
                    "Do not explain, do not add commentary, return only results."
                    f"Ignore the name of the current document, which is {doc_name}."
                )
            },
            {
                "role": "user",
                "content": getattr(row, "content_giv", "")
            },
            {
                "role": "assistant",
                "content": getattr(row, "answer", "")
            }
        ]
    }


def main():
    # ----------------------------
    # LOAD DATA
    # ----------------------------
    cit_df = pd.read_csv(CITATIONS_CSV)
    res_df = pd.read_csv(RESOLUTIONS_CSV)

    new_df = res_df.copy()

    new_df["content"] = new_df["content"].apply(
        lambda x: trim_contents(x, tokenizer, target_tokens=MAX_TOKEN_LENGTH, max_tokens=MAX_TOKEN_LENGTH+30)
    )

    # Remove discarded documents
    new_df = new_df.dropna(subset=["content"])

    # Keep id and content
    new_df = new_df[["res_id2", "content"]]

    # Filter the res_df and the cit_df to be use exclusively with the contents shorter than 3K tokens
    res_df = res_df[res_df['res_id2'].isin(new_df["res_id2"])].copy()
    cit_df = cit_df[cit_df['res_id2_doc_giving_cite'].isin(new_df["res_id2"])].copy()

    # Apply sort and split
    train_df, val_df, test_df = sort_and_split(res_df, cit_df)

    output_files = [OUTPUT_PATH + "resolution_train.jsonl",
                    OUTPUT_PATH + "resolution_val.jsonl",
                    OUTPUT_PATH + "resolution_test.jsonl"]

    for path, df in zip(output_files, [train_df, val_df, test_df]):
        with (open(path, "w", encoding="utf-8") as f):
            for row in df.itertuples(index=False):
                f.write(json.dumps(format_message_line(row), ensure_ascii=False) + "\n")

    print("JSONL written!")

if __name__ == "__main__":
    main()
