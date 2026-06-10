import pandas as pd
import os
import re

# ----------------------------
# SETTINGS
# ----------------------------
CONV_PROT_REC_TXT = "./conventions-protocols-recommendations.txt"
RESOLUTIONS_CSV = "../resolutions/ga_resolutions_1946_2019.csv"

# ----------------------------
# LOAD DATA
# ----------------------------
df_c_p_r = None
if not os.path.isfile(CONV_PROT_REC_TXT):
    print(f"Error: File '{CONV_PROT_REC_TXT}' does not exist.")

else:
    print("File found. Loading...")

    try:
        # Read all lines
        with open(CONV_PROT_REC_TXT, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data = []

        for line in lines:

            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Split only on the first dash
            code, title = line.split(' - ', 1)

            # Extract year (last 4-digit number after last comma)
            year_match = re.search(r',\s*(\d{4})', title)
            year = year_match.group(1) if year_match else None

            # Extract number inside (No. X)
            number_match = re.search(r'\(No\.\s*([^)]+)\)', title)
            number = number_match.group(1) if number_match else None

            data.append({
                'code': code.strip(),
                'title': title.strip(),
                'year': year,
                'number': number
            })
        # Create dataframe
        df_c_p_r = pd.DataFrame(data)

        print("TXT file loaded successfully.")
        # Assertion on uniqueness
        assert len(df_c_p_r['title'].unique()) == len(df_c_p_r)
    except Exception as e:
        print(f"Error loading TXT file: {e}")

df = None
if not os.path.isfile(RESOLUTIONS_CSV):
    print(f"Error: File '{RESOLUTIONS_CSV}' does not exist.")
else:
    print("File found. Loading...")

    try:
        # Load in chunks to avoid memory issues
        chunksize = 200000
        chunks = []

        for chunk in pd.read_csv(RESOLUTIONS_CSV, chunksize=chunksize):
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        print("CSV file loaded successfully.")

    except Exception as e:
        print(f"Error loading CSV file: {e}")

# ----------------------------
# PROCESS RESOLUTIONS
# ----------------------------
cites = pd.DataFrame(columns=['res_id2', 'conv_prot_rec'], dtype='object')

for title, idx in zip(df_c_p_r['title'], range(len(df_c_p_r))):
  if idx%10 == 0:
    print(idx, 'out of', len(df_c_p_r))
  for res_id, cont in zip(df['res_id2'], df['content']):
    if title.lower() in cont.lower():
      cites = pd.concat([cites, pd.DataFrame({'res_id2': [res_id], 'conv_prot_rec': [title]})], ignore_index=True)

print('Now our dataframe of cites has', len(cites), 'new registers')

# ----------------------------
# SAVE DATA
# ----------------------------
cites.to_csv("./citations_conv_prot_rec.csv", index=False)
df_c_p_r.to_csv("./conventions-protocols-recommendations.csv", index=False)