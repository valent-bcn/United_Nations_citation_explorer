"""
Subject histogram + "do embeddings agree with the subject tags?" check.

1. Reads the resolutions CSV (columns: `symbol`, `subject`), where `subject`
   is a comma-separated list of subject tags for each document. A document
   with several subjects counts toward each of them.
2. Plots a histogram (bar chart) of the most common subjects.
3. For each of the most common subjects, embeds the subject text as a query
   and scores it against EVERY document in the FAISS index. From those scores:
     a) precision@K : how many of the top-K documents carry the subject tag
     b) score distribution : histogram of the similarity scores of the
        documents that carry the tag (vs. those that don't), per subject

Requires similarity_doc_topic.py in the same folder: we reuse its index paths,
model loading, query encoding and search, so the query is embedded exactly
the same way as in that script.

Usage:
    python subject_vs_embedding.py
    python subject_vs_embedding.py --top-subjects 20 --k 20 --query-case sentence
    python subject_vs_embedding.py --score-range 0.2 0.99 --bins 40
    python subject_vs_embedding.py --hist-only        # histogram only, no model
"""

import argparse
import re
from collections import Counter
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEFAULT_CSV = "../resolutions/ga_resolutions_1946_2019-subjects.csv"
BLUE, GREY = "#1f77b4", "#9e9e9e"


# ----------------------------
# ID / SUBJECT NORMALIZATION
# ----------------------------
def norm_id(x) -> str:
    """Make CSV `symbol` and FAISS `doc_id` comparable.

    Steps: lowercase -> drop file extension -> drop leading 'A/RES/' ->
    remove ALL whitespace -> collapse runs of non-alphanumerics to '_'.

        'A/RES/996(ES-IANDII)'  -> '996_es_iandii'
        '996 (es-i and ii)'     -> '996_es_iandii'
        'A/RES/47/1'            -> '47_1'
        '47/1'                  -> '47_1'

    Separators become '_' rather than being deleted, so '4/71' ('4_71')
    never collides with '47/1' ('47_1').
    """
    s = str(x).strip().lower()
    s = re.sub(r"\.(txt|pdf|json|md|html?)$", "", s)
    s = re.sub(r"^a[\s/_\-]*res[\s/_\-]*", "", s)  # leading 'A/RES/' (any separator)
    s = re.sub(r"\s+", "", s)  # '996 (es-i and ii)' == '996(es-iandii)'
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def safe_name(s: str, n: int = 60) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()[:n]


def apply_case(text: str, mode: str) -> str:
    if mode == "lower":
        return text.lower()
    if mode == "title":
        return text.title()
    if mode == "sentence":
        return text.capitalize()
    return text


# ----------------------------
# CSV LOADING
# ----------------------------
def load_subjects(csv_path: str):
    """Returns:
    doc_subjects : {normalized doc id -> set of subject keys (lowercased)}
    raw_symbol   : {normalized doc id -> original symbol string}
    display_name : {subject key -> most common original spelling}
    """
    df = pd.read_csv(csv_path)
    missing = {"symbol", "subject"} - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing column(s) {sorted(missing)}. Found: {list(df.columns)}")

    doc_subjects, raw_symbol, spellings = {}, {}, {}
    for symbol, subj in zip(df["symbol"], df["subject"]):
        if pd.isna(symbol):
            continue
        sid = norm_id(symbol)
        keys = doc_subjects.setdefault(sid, set())  # duplicate symbols are merged
        raw_symbol.setdefault(sid, str(symbol))
        if pd.isna(subj):
            continue
        for part in str(subj).split(","):  # several subjects per document
            part = clean_ws(part)
            if not part:
                continue
            key = part.lower()
            keys.add(key)
            spellings.setdefault(key, Counter())[part] += 1

    display_name = {k: c.most_common(1)[0][0] for k, c in spellings.items()}
    return doc_subjects, raw_symbol, display_name


# ----------------------------
# STATS
# ----------------------------
def auc_tagged_vs_rest(tagged: np.ndarray, other: np.ndarray) -> float:
    """P(random tagged doc scores higher than random untagged doc).
    0.5 = embedding doesn't separate them, 1.0 = perfect separation."""
    n1, n0 = len(tagged), len(other)
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([tagged, other])).rank().to_numpy()
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------
# PLOTS
# ----------------------------
def plot_histogram(counts: Counter, display_name: dict, n: int, n_docs: int, out_path: Path):
    top = counts.most_common(n)
    labels = [display_name[k] for k, _ in top][::-1]
    values = [v for _, v in top][::-1]

    fig, ax = plt.subplots(figsize=(10, 0.32 * len(top) + 1.5))
    bars = ax.barh(labels, values)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("Number of documents")
    ax.set_title(f"Top {len(top)} most common subjects ({n_docs} documents)")
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_precision(summary: pd.DataFrame, k: int, out_path: Path):
    s = summary.iloc[::-1]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(s) + 2))
    ax.barh(y + 0.2, s["precision_at_k"], height=0.4, label=f"Precision@{k} (embedding search)")
    ax.barh(y - 0.2, s["base_rate"], height=0.4, label="Base rate (share of all docs with the tag)")
    ax.set_yticks(y)
    ax.set_yticklabels(s["subject"], fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Fraction of documents carrying the subject tag")
    ax.set_title(f"Do the top-{k} embedding matches carry the subject tag?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(name, tagged, other, bins, auc, out_path: Path):
    """Left: raw counts of tagged docs per score bin (the shape asked for).
    Right: tagged vs. not-tagged, each normalized to area 1, to compare shapes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.hist(tagged, bins=bins, color=BLUE, edgecolor="white", linewidth=0.4)
    if len(tagged):
        med = float(np.median(tagged))
        ax.axvline(med, color="k", ls="--", lw=1, label=f"median {med:.3f}")
        ax.legend(frameon=False)
    ax.set_title(f'"{name}": documents tagged with it (n={len(tagged)})', fontsize=10)
    ax.set_xlabel("cosine similarity: subject query vs. document")
    ax.set_ylabel("number of documents")

    ax = axes[1]
    if len(other):
        ax.hist(other, bins=bins, density=True, alpha=0.55, color=GREY, label=f"not tagged (n={len(other)})")
    if len(tagged):
        ax.hist(tagged, bins=bins, density=True, alpha=0.55, color=BLUE, label=f"tagged (n={len(tagged)})")
    ax.set_title(f"tagged vs. not tagged, normalized  (AUC = {auc:.3f})", fontsize=10)
    ax.set_xlabel("cosine similarity: subject query vs. document")
    ax.set_ylabel("density")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_score_grid(entries, bins, cols: int, out_path: Path):
    """Small multiples: one panel per subject, same x-axis everywhere."""
    rows = ceil(len(entries) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.5 * rows), sharex=True, squeeze=False)
    for ax, (name, tagged, other, auc) in zip(axes.ravel(), entries):
        if len(other):
            ax.hist(other, bins=bins, density=True, histtype="stepfilled", color=GREY, alpha=0.5)
        if len(tagged):
            ax.hist(tagged, bins=bins, density=True, histtype="step", color=BLUE, linewidth=1.6)
        ax.set_title(f"{name[:30]} (n={len(tagged)}, AUC {auc:.2f})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
    for ax in axes.ravel()[len(entries):]:
        ax.axis("off")
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=1.6, label="documents tagged with the subject"),
            Patch(color=GREY, alpha=0.5, label="documents without the tag"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    fig.supxlabel("cosine similarity: subject query vs. document", fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------
# EMBEDDING vs TAGS
# ----------------------------
def run_embedding_check(args, top_keys, doc_subjects, raw_symbol, display_name, out_dir: Path):
    # Imported here so --hist-only doesn't need torch / faiss / the model.
    import faiss
    from similarity_doc_topic import (
        INDEX_PATH,
        METADATA_PATH,
        encode_query,
        load_model,
        search,
    )
    INDEX_PATH = "faiss_docs_resolutions.bin"
    METADATA_PATH = "faiss_docs_resolutions.npy"

    print(f"\nLoading FAISS index from {INDEX_PATH} ...")
    index = faiss.read_index(INDEX_PATH)
    metadata = np.load(METADATA_PATH, allow_pickle=True)
    n_total = index.ntotal
    print(f"Index: {n_total} vectors | metadata: {len(metadata)} entries")
    if n_total != len(metadata):
        print("WARNING: index and metadata sizes differ; results may be misaligned.")

    # --- match FAISS doc_id <-> CSV symbol ---
    idx_ids = [norm_id(m["doc_id"]) for m in metadata]
    known = np.array([i in doc_subjects for i in idx_ids])  # indexed doc present in the CSV
    print(f"ID match: {int(known.sum())}/{len(idx_ids)} indexed docs found in the CSV (by symbol)")
    if not known.all():
        un = np.flatnonzero(~known)
        print("  Unmatched doc_ids (raw -> normalized):")
        for j in un[:8]:
            print(f"    {metadata[j]['doc_id']!r:35} -> {idx_ids[j]!r}")
        in_index = set(idx_ids)
        csv_only = [s for s in doc_subjects if s not in in_index]
        print(f"  CSV symbols not in the index: {len(csv_only)} (raw -> normalized):")
        for s in csv_only[:8]:
            print(f"    {raw_symbol[s]!r:35} -> {s!r}")
    if not known.any():
        raise SystemExit("No doc_id matched any CSV symbol - adjust norm_id() for your ID format.")

    # Base rate among indexed, labelled docs: what a random document would score.
    n_lab = int(known.sum())
    base_counts = Counter(k for i, ok in zip(idx_ids, known) if ok for k in doc_subjects[i])

    print("Loading embedding model ...")
    model = load_model()

    # ---- 1. score EVERY indexed document against each subject query ----
    print(f"Scoring {len(top_keys)} subject queries against all {n_total} documents ...")
    sims, ranked = {}, {}
    for key in top_keys:
        query = apply_case(display_name[key], args.query_case)
        scores, idxs = search(index, encode_query(model, query), k=n_total)  # flat index: exact
        valid = idxs != -1
        s = np.full(n_total, np.nan, dtype=np.float32)
        s[idxs[valid]] = scores[valid]  # similarity indexed by document position
        sims[key] = s
        ranked[key] = (idxs[valid], scores[valid])  # already sorted best-first

    # ---- 2. precision@K (does the top-K carry the tag?) ----
    detail_rows, summary_rows = [], []
    for key in top_keys:
        name = display_name[key]
        top_idx, top_scores = ranked[key][0][: args.k], ranked[key][1][: args.k]
        hits = labelled_k = unlabelled_k = 0
        first_miss = None
        rows = []
        for rank, (ix, score) in enumerate(zip(top_idx, top_scores), start=1):
            ix = int(ix)
            did = idx_ids[ix]
            is_known = bool(known[ix])
            subs = doc_subjects.get(did, set())
            has = key in subs  # ANY of the document's subjects
            if is_known:
                labelled_k += 1
                hits += has
                if not has and first_miss is None:
                    first_miss = rank
            else:
                unlabelled_k += 1
            rows.append(
                {
                    "subject": name,
                    "rank": rank,
                    "doc_id": metadata[ix]["doc_id"],
                    "symbol": raw_symbol.get(did, ""),
                    "score": float(score),
                    "in_csv": is_known,
                    "has_subject": has if is_known else None,
                    "all_subjects": " | ".join(sorted(display_name[s] for s in subs)),
                    "preview": clean_ws(str(metadata[ix]["content"]))[:150],
                }
            )
        detail_rows.extend(rows)

        precision = hits / labelled_k if labelled_k else float("nan")
        base_rate = base_counts[key] / n_lab
        summary_rows.append(
            {
                "subject": name,
                "docs_with_tag": base_counts[key],
                "base_rate": base_rate,
                f"hits_in_top{args.k}": hits,
                "precision_at_k": precision,
                "lift": precision / base_rate if base_rate else float("nan"),
                "all_tagged": bool(labelled_k and hits == labelled_k),
                "first_miss_rank": first_miss,
                "unlabelled_in_top_k": unlabelled_k,
                "mean_score": float(np.mean([r["score"] for r in rows])) if rows else float("nan"),
            }
        )

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)
    summary.to_csv(out_dir / "subject_vs_embedding_summary.csv", index=False)
    details.to_csv(out_dir / "subject_vs_embedding_details.csv", index=False)
    plot_precision(summary, args.k, out_dir / "precision_vs_baseline.png")

    # ---- 3. score distributions: tagged vs. not tagged ----
    all_scores = np.concatenate([sims[k][known] for k in top_keys])
    lo, hi = args.score_range if args.score_range else (float(np.nanmin(all_scores)), float(np.nanmax(all_scores)))
    bins = np.linspace(lo, hi, args.bins + 1)
    print(f"Score range used for histograms: {lo:.3f} .. {hi:.3f} ({args.bins} bins)")

    dist_dir = out_dir / "score_distributions"
    dist_dir.mkdir(exist_ok=True)
    stats_rows, bin_rows, entries = [], [], []
    for n, key in enumerate(top_keys, start=1):
        name = display_name[key]
        has = np.array([key in doc_subjects.get(i, ()) for i in idx_ids])
        tagged = sims[key][known & has]
        other = sims[key][known & ~has]
        auc = auc_tagged_vs_rest(tagged, other)
        entries.append((name, tagged, other, auc))

        q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")
        stats_rows.append(
            {
                "subject": name,
                "n_tagged": len(tagged),
                "n_not_tagged": len(other),
                "tagged_min": q(tagged, 0),
                "tagged_p05": q(tagged, 5),
                "tagged_p25": q(tagged, 25),
                "tagged_median": q(tagged, 50),
                "tagged_p75": q(tagged, 75),
                "tagged_p95": q(tagged, 95),
                "tagged_max": q(tagged, 100),
                "tagged_mean": float(tagged.mean()) if len(tagged) else float("nan"),
                "not_tagged_median": q(other, 50),
                "not_tagged_p95": q(other, 95),
                "auc_tagged_vs_rest": auc,
            }
        )
        c_tag, _ = np.histogram(tagged, bins=bins)
        c_oth, _ = np.histogram(other, bins=bins)
        for b in range(args.bins):
            bin_rows.append(
                {
                    "subject": name,
                    "bin_left": bins[b],
                    "bin_right": bins[b + 1],
                    "n_tagged": int(c_tag[b]),
                    "n_not_tagged": int(c_oth[b]),
                }
            )
        plot_score_distribution(name, tagged, other, bins, auc, dist_dir / f"{n:02d}_{safe_name(name)}.png")

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(out_dir / "score_distribution_stats.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(out_dir / "score_histogram_bins.csv", index=False)
    plot_score_grid(entries, bins, args.grid_cols, out_dir / "score_distributions_grid.png")

    # ---- console report ----
    print(f"\n=== Precision@{args.k}: share of top-{args.k} embedding matches carrying the tag ===")
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.max_colwidth", 40):
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

        print("\n=== Similarity-score distribution of documents TAGGED with each subject ===")
        cols = ["subject", "n_tagged", "tagged_p05", "tagged_median", "tagged_p95", "tagged_max",
                "not_tagged_median", "auc_tagged_vs_rest"]
        print(stats[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    show = top_keys if args.verbose else top_keys[:1]
    for key in show:
        name = display_name[key]
        print(f"\n--- Top {args.k} documents for \"{apply_case(name, args.query_case)}\" ---")
        for r in details[details["subject"] == name].itertuples():
            mark = "?" if not r.in_csv else ("Y" if r.has_subject else "N")
            print(f"  #{r.rank:<2} [{mark}] {r.score:.4f}  {r.doc_id}  ::  {r.all_subjects}")
    print("\n[Y] has the tag, [N] doesn't, [?] doc not found in the CSV")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=DEFAULT_CSV, help="CSV with `symbol` and `subject` columns.")
    p.add_argument("--out-dir", default="./subject_analysis", help="Where to write plots and CSVs.")
    p.add_argument("--hist-n", type=int, default=50, help="Subjects shown in the subject histogram.")
    p.add_argument("--top-subjects", type=int, default=50,
                   help="Most common subjects to test with embeddings (precision@K + score distributions).")
    p.add_argument("--k", type=int, default=20, help="Top-K documents for the precision@K check.")
    p.add_argument("--bins", type=int, default=40, help="Bins in the score-distribution histograms.")
    p.add_argument("--score-range", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                   help="Fixed x-range for score histograms, e.g. 0.2 0.99. Default: min..max of the data.")
    p.add_argument("--grid-cols", type=int, default=5, help="Columns in the all-subjects overview grid.")
    p.add_argument(
        "--query-case",
        choices=["asis", "lower", "title", "sentence"],
        default="asis",
        help="Casing of the subject text used as the query (CSV subjects may be ALL CAPS).",
    )
    p.add_argument("--hist-only", action="store_true", help="Only build the subject histogram; skip embeddings.")
    p.add_argument("--verbose", action="store_true", help="Print the top-K list for every subject, not just #1.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. histogram of subjects ----
    doc_subjects, raw_symbol, display_name = load_subjects(args.csv)
    counts = Counter(k for keys in doc_subjects.values() for k in keys)  # each doc counted once per subject
    n_docs = len(doc_subjects)
    print(f"{n_docs} unique documents, {len(counts)} distinct subjects")
    print("\nMost common subjects:")
    for k, c in counts.most_common(15):
        print(f"  {c:>6}  {display_name[k]}")

    pd.DataFrame(
        [(display_name[k], c, c / n_docs) for k, c in counts.most_common()],
        columns=["subject", "n_documents", "share_of_documents"],
    ).to_csv(out_dir / "subject_counts.csv", index=False)
    plot_histogram(counts, display_name, args.hist_n, n_docs, out_dir / "subject_histogram.png")
    print(f"\nSaved histogram -> {out_dir / 'subject_histogram.png'}")

    if args.hist_only:
        return

    # ---- 2. subject-as-embedding vs. subject tags ----
    top_keys = [k for k, _ in counts.most_common(args.top_subjects)]
    run_embedding_check(args, top_keys, doc_subjects, raw_symbol, display_name, out_dir)
    print(f"\nOutputs written to {out_dir}/")


if __name__ == "__main__":
    main()