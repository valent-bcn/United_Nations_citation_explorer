"""
Query a previously-built FAISS document index for the most similar
document to a given concept (e.g. "gender equality").

Expects the two files produced by the indexing notebook to be in the
current working directory ("./"):
    - faiss_docs_resolutions.bin   (FAISS index, IndexFlatIP)
    - faiss_docs_resolutions.npy   (metadata: array of {"doc_id","content"})

Usage:
    python search_similar_doc.py
    python search_similar_doc.py "climate change" --k 5
"""

import argparse
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# ----------------------------
# SETTINGS
# ----------------------------
BASE_DIR = "./document_embeddings/"
INDEX_PATH = BASE_DIR + "faiss_docs_resolutions.bin"
METADATA_PATH = BASE_DIR + "faiss_docs_resolutions.npy"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def load_model() -> SentenceTransformer:
    """Load the same embedding model used to build the index.

    Falls back to CPU/float32 automatically if no GPU is available
    (the original notebook ran on a Colab T4 GPU with float16).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # "device_map": "auto" is for splitting a model across multiple
    # GPUs (or GPU+disk) - on a CPU-only machine it leaves some weights
    # on a "meta" device with no real data, which crashes on .to(device).
    # So we only use it when CUDA is actually available, and otherwise
    # let SentenceTransformer's own `device=` argument place the model.
    model_kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
    )
    return model


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Encode a single query string the same way documents were encoded
    (plain encode + manual L2 normalization, so it stays comparable to
    the doc embeddings that were built with the same recipe)."""
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    return l2_normalize(emb)


def search(index: faiss.Index, query_emb: np.ndarray, k: int = 1):
    scores, indices = index.search(query_emb, k)
    return scores[0], indices[0]


def main():
    parser = argparse.ArgumentParser(description="Find the most similar document(s) to a concept.")
    parser.add_argument("query", nargs="?", default="gender equality", help="Concept/query text.")
    parser.add_argument("--k", type=int, default=1, help="Number of top results to show.")
    args = parser.parse_args()

    print(f"Loading FAISS index from {INDEX_PATH} ...")
    index = faiss.read_index(INDEX_PATH)
    print(f"Index loaded: {index.ntotal} documents, dim={index.d}")

    print(f"Loading metadata from {METADATA_PATH} ...")
    metadata = np.load(METADATA_PATH, allow_pickle=True)
    print(f"Metadata loaded: {len(metadata)} entries")

    if index.ntotal != len(metadata):
        print(
            f"WARNING: index has {index.ntotal} vectors but metadata has "
            f"{len(metadata)} entries — they may be out of sync."
        )

    print("Loading embedding model (this can take a moment)...")
    model = load_model()

    query_emb = encode_query(model, args.query)
    scores, idxs = search(index, query_emb, k=args.k)

    print(f"\nTop {args.k} result(s) for: \"{args.query}\"\n" + "-" * 50)
    for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
        if idx == -1:
            continue
        doc = metadata[idx]
        content = str(doc["content"])
        preview = content if len(content) <= 1000 else content[:1000] + " [...]"
        print(f"\n#{rank}  doc_id: {doc['doc_id']}   similarity: {score:.4f}")
        print(preview)


if __name__ == "__main__":
    main()