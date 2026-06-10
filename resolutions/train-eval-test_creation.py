import pandas as pd
import os
import json

# ----------------------------
# SETTINGS
# ----------------------------
CITATIONS_CSV = "./ga_citations_1946_2019.csv"
RESOLUTIONS_CSV = "./ga_resolutions_1946_2019.csv"
OUTPUT_PATH = "../Qwen3/data/"

# ----------------------------
# LOAD DATA
# ----------------------------
cit_df = pd.read_csv(CITATIONS_CSV)
res_df = pd.read_csv(RESOLUTIONS_CSV)

# ----------------------------
# MESSAGE TEMPLATE
# ----------------------------


# ----------------------------
# SORT & SPLIT DATA
# ----------------------------

# We prepare the contents forhand
detail = res_df[["res_id2", "res_id2_unlet", "date_p", "content"]].copy() #different for each lettered resolution (resolution sectionned by letter)
detail["res_id2"] = detail["res_id2"].str.strip()

# First we take the resolution ids of the ones we want to take for train+val and for test
res_df = res_df[["res_id2_unlet", "date_p"]].drop_duplicates().reset_index(drop=True) #res_id has letters, we want the whole resolution (unlet)
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

print(cit_df.shape)

train_df = cit_df[cit_df["res_id2_unlet_giv"].isin(train_ids)]
val_df = cit_df[cit_df["res_id2_unlet_giv"].isin(val_ids)]
test_df = cit_df[cit_df["res_id2_unlet_giv"].isin(test_ids)]


# ----------------------------
# HELPERS
# ----------------------------
def format_message_line(row):
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict information extraction system. "
                    "Extract only United Nations general assembly resolution codes from the input. "
                    "Do not explain, do not add commentary, return only results."
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

output_files = [OUTPUT_PATH + "resolution_train.jsonl",
                OUTPUT_PATH + "resolution_val.jsonl",
                OUTPUT_PATH + "resolution_test.jsonl"]
'''
existing_file = [f for f in output_files if os.path.exists(f)]

if existing_file:
    resp = input(f"A JSONL File already exists. Overwrite? (y/n): ").strip().lower()
    if resp != "y":
        print("Aborted. File not overwritten.")
        exit()
    else:
        print("Overwriting file...")
'''

for path, df in zip(output_files, [train_df, val_df, test_df]):
    with (open(path, "w", encoding="utf-8") as f):
        for row in df.itertuples(index=False):
            f.write(json.dumps(format_message_line(row), ensure_ascii=False) + "\n")

print("JSONL files written.")