import json

with open("doc_vector_database.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# remove broken widget metadata
nb.get("metadata", {}).pop("widgets", None)

with open("doc_vector_database.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)