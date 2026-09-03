#!/usr/bin/env python3
"""
Build train/val/test JSONL datasets for UN resolution citation extraction.

Usage:
    python build_resolution_dataset.py                 # GA citations only
    python build_resolution_dataset.py --old            # use the legacy 1946-2019 GA citation file
    python build_resolution_dataset.py --sc             # also include Security Council citations
    python build_resolution_dataset.py --ohchr          # also include OHCHR citations
    python build_resolution_dataset.py --sc --ohchr     # combine all three

Adding a new citation source (e.g. a future ICJ or treaty-body file) only
requires adding one entry to KNOWN_SOURCES below -- the CLI flag, the
"no citation" exclusion logic, the answer merging, and the system prompt
all pick it up automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(".")
DEFAULT_OUTPUT_DIR = Path("../Qwen3/data/")
DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_MAX_TOKEN_LENGTH = 2926
DEFAULT_TOKEN_SLACK = 30  # a trimmed candidate must fit within max_tokens + slack

GA_CITATIONS_CURRENT = "ga_citations_1946_2019.csv"
GA_CITATIONS_OLD = "ga_citations_1946_2019_OLD.csv"
RESOLUTIONS_FILENAME = "ga_resolutions_1946_2019.csv"


@dataclass(frozen=True)
class SourceConfig:
    """Static description of one *additional* (non-GA) citation source."""
    csv_filename: str
    resolution_column: str      # column already holding the fully-coded citation, e.g. "S/RES/1234"
    description: str            # used in the generated system prompt
    id_column: str = "res_id2"  # GA document id column that "contains" the citation


# Registry of optional sources. Add a line here + nothing else to support a new source.
KNOWN_SOURCES: Dict[str, SourceConfig] = {
    "sc": SourceConfig(
        csv_filename="sc_citations.csv",
        resolution_column="sc_resolution",
        description="security council resolution codes (S/RES/)",
    ),
    "ohchr": SourceConfig(
        csv_filename="ohchr_citations.csv",
        resolution_column="ohchr_resolution",
        description="human rights council resolution codes (A/HRC/RES/)",
    ),
}


def normalize_id(series: pd.Series) -> pd.Series:
    """Canonical 'A/RES/CODE' form used as the join key everywhere."""
    return "A/RES/" + series.str.strip().str.replace(" ", "").str.upper()


@dataclass
class CitationSource:
    """A resolved, loadable citation source (registry entry + actual path)."""
    name: str
    csv_path: Path
    resolution_column: str
    id_column: str = "res_id2"

    def load_grouped(self) -> pd.DataFrame:
        """One row per citing document, indexed by normalized id, with all of
        this source's resolution codes joined by '; ' in a column named `self.name`."""
        df = pd.read_csv(self.csv_path)
        df[self.resolution_column] = df[self.resolution_column].str.replace(" ", "")
        df[self.id_column] = normalize_id(df[self.id_column])
        return (
            df.groupby(self.id_column, as_index=True)
            .agg(**{self.name: (self.resolution_column, lambda x: "; ".join(x))})
        )


# ----------------------------------------------------------------------------
# TOKEN TRIMMING
# ----------------------------------------------------------------------------

def trim_contents(text: str, tokenizer, target_tokens: int, max_tokens: int) -> str | None:
    """Shorten `text` to end on a sentence boundary at/after `target_tokens`.
    Returns the original text unchanged if already short enough, a trimmed
    candidate if it fits within `max_tokens`, or None if it can't be brought
    under budget at all (caller should drop the document)."""
    total_tokens = len(tokenizer.encode(text, add_special_tokens=False))

    if total_tokens <= target_tokens:
        return text
    if total_tokens > max_tokens:
        return None

    sentence_endings = [m.end() for m in re.finditer(r"[.!?]", text)]

    first_part = tokenizer.decode(
        tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    )
    char_pos = len(first_part)

    next_end = next((pos for pos in sentence_endings if pos >= char_pos), None)
    if next_end is None:
        return None

    candidate = text[:next_end]
    if len(tokenizer.encode(candidate, add_special_tokens=False)) <= max_tokens:
        return candidate
    return None


# ----------------------------------------------------------------------------
# DATASET BUILDER
# ----------------------------------------------------------------------------

class ResolutionDatasetBuilder:
    def __init__(
        self,
        resolutions_csv: Path,
        ga_citations_csv: Path,
        extra_sources: List[CitationSource],
        tokenizer,
        max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH,
        token_slack: int = DEFAULT_TOKEN_SLACK,
    ):
        self.resolutions_csv = resolutions_csv
        self.ga_citations_csv = ga_citations_csv
        self.extra_sources = extra_sources
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.token_slack = token_slack

        self.res_df: pd.DataFrame | None = None
        self.cit_df: pd.DataFrame | None = None

    # -- loading -----------------------------------------------------------
    def load(self) -> None:
        logger.info("Loading %s and %s", self.resolutions_csv, self.ga_citations_csv)
        self.res_df = pd.read_csv(self.resolutions_csv)
        self.cit_df = pd.read_csv(self.ga_citations_csv)

    # -- length filtering ----------------------------------------------------
    def filter_by_length(self) -> None:
        """Drop resolutions whose content can't fit the token budget, and -
        importantly - replace `content` with the *trimmed* text for the ones
        kept, so the trimming actually reaches the training examples."""
        trimmed = self.res_df.copy()
        trimmed["content"] = trimmed["content"].apply(
            lambda x: trim_contents(
                x, self.tokenizer, self.max_token_length,
                self.max_token_length + self.token_slack,
            )
        )
        trimmed = trimmed.dropna(subset=["content"])
        keep_ids = set(trimmed["res_id2"])

        self.res_df = self.res_df[self.res_df["res_id2"].isin(keep_ids)].copy()
        self.res_df = self.res_df.merge(
            trimmed[["res_id2", "content"]].rename(columns={"content": "content_trimmed"}),
            on="res_id2", how="left",
        )
        self.res_df["content"] = self.res_df["content_trimmed"]
        self.res_df.drop(columns=["content_trimmed"], inplace=True)

        # Keep a normalized-id -> trimmed-content lookup so the *separate*
        # content_giv column in the citations CSV can be corrected too.
        trimmed["res_id2_norm"] = normalize_id(trimmed["res_id2"])
        self.content_by_id = trimmed.set_index("res_id2_norm")["content"]

        self.cit_df = self.cit_df[self.cit_df["res_id2_doc_giving_cite"].isin(keep_ids)].copy()
        logger.info("%d resolutions remain after length filtering", len(self.res_df))

    # -- split boundaries ------------------------------------------------
    def _train_val_test_ids(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ids = self.res_df[["res_id2_unlet", "date_p"]].drop_duplicates().reset_index(drop=True)
        ids["res_id2_unlet"] = ids["res_id2_unlet"].str.strip()
        ids.sort_values(by="date_p", ascending=True, inplace=True)

        n = len(ids)
        train_cut = int(n * 0.8)
        val_cut = int(n * 0.9)
        # Oldest 80% -> train, next 10% -> val, newest 10% -> test.
        train_ids = ids[:train_cut]["res_id2_unlet"]
        val_ids = ids[train_cut:val_cut]["res_id2_unlet"]
        test_ids = ids[val_cut:]["res_id2_unlet"]
        return train_ids, val_ids, test_ids

    # -- main assembly -----------------------------------------------------
    def sort_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        detail = self.res_df[["res_id2", "res_id2_unlet", "date_p", "content"]].copy()
        detail["res_id2"] = normalize_id(detail["res_id2"])

        cit_df = self.cit_df[
            ["res_id2_doc_giving_cite", "res_id2_unlet_giv", "date_p_giv", "content_giv", "res_id2_doc_receiv_cite"]
        ].copy()
        cit_df["res_id2_doc_giving_cite"] = normalize_id(cit_df["res_id2_doc_giving_cite"])
        cit_df["res_id2_doc_receiv_cite"] = normalize_id(cit_df["res_id2_doc_receiv_cite"])
        cit_df["content_giv"] = (
            cit_df["res_id2_doc_giving_cite"].map(self.content_by_id).fillna(cit_df["content_giv"])
        )

        # Load & group every extra source once; reuse for both exclusion and merging.
        grouped_sources: Dict[str, pd.DataFrame] = {}
        cited_ids = set(cit_df["res_id2_doc_giving_cite"])
        for src in self.extra_sources:
            grouped = src.load_grouped()
            grouped_sources[src.name] = grouped
            cited_ids |= set(grouped.index)

        no_cite = detail[~detail["res_id2"].isin(cited_ids)].copy()
        no_cite["res_id2_doc_receiv_cite"] = "None"
        no_cite = no_cite[["res_id2", "res_id2_unlet", "date_p", "content", "res_id2_doc_receiv_cite"]]
        no_cite.columns = [
            "res_id2_doc_giving_cite", "res_id2_unlet_giv",
            "date_p_giv", "content_giv", "res_id2_doc_receiv_cite",
        ]

        cit_df = pd.concat([cit_df, no_cite], ignore_index=True)
        cit_df.sort_values(
            by=["date_p_giv", "res_id2_doc_giving_cite", "res_id2_doc_receiv_cite"],
            inplace=True,
        )

        cit_df = (
            cit_df.groupby(["res_id2_doc_giving_cite", "res_id2_unlet_giv", "content_giv"], as_index=False)
            .agg(answer=("res_id2_doc_receiv_cite", lambda x: "; ".join(x)))
        )

        for name, grouped in grouped_sources.items():
            cit_df = cit_df.merge(grouped, how="left", left_on="res_id2_doc_giving_cite", right_index=True)
            cit_df["answer"] = cit_df["answer"].fillna("") + "; " + cit_df[name].fillna("")
            cit_df.drop(columns=[name], inplace=True)

        if grouped_sources:
            cit_df["answer"] = (
                cit_df["answer"].str.replace(r"(;\s*)+", "; ", regex=True).str.strip("; ")
            )

        logger.info("Citation dataframe shape: %s", cit_df.shape)

        train_ids, val_ids, test_ids = self._train_val_test_ids()
        train_df = cit_df[cit_df["res_id2_unlet_giv"].isin(train_ids)]
        val_df = cit_df[cit_df["res_id2_unlet_giv"].isin(val_ids)]
        test_df = cit_df[cit_df["res_id2_unlet_giv"].isin(test_ids)]
        return train_df, val_df, test_df

    # -- formatting ----------------------------------------------------------
    def _system_prompt(self, doc_name: str) -> str:
        parts = ["general assembly resolution codes (A/RES/)"]
        order = ["general assembly"]
        for src in self.extra_sources:
            parts.append(KNOWN_SOURCES[src.name].description)
            order.append(src.name)
        return (
            "You are a strict information extraction system. "
            f"Extract from the text only {', '.join(parts)}. "
            "Do not explain, do not add commentary, return only results. "
            f"Use ; as delimiter and follow this order: {', '.join(order)}. "
            "Exclude the current document's self code/title."
        )

    def format_message_line(self, row) -> dict:
        doc_name = getattr(row, "res_id2_doc_giving_cite", "")
        return {
            "messages": [
                {"role": "system", "content": self._system_prompt(doc_name)},
                {"role": "user", "content": getattr(row, "content_giv", "")},
                {"role": "assistant", "content": getattr(row, "answer", "")},
            ]
        }

    # -- orchestration ---------------------------------------------------
    def run(self, output_dir: Path) -> None:
        self.load()
        self.filter_by_length()
        train_df, val_df, test_df = self.sort_and_split()

        output_dir.mkdir(parents=True, exist_ok=True)
        splits = {
            "train": (output_dir / "resolution_train.jsonl", train_df),
            "val": (output_dir / "resolution_val.jsonl", val_df),
            "test": (output_dir / "resolution_test.jsonl", test_df),
        }
        for name, (path, df) in splits.items():
            with open(path, "w", encoding="utf-8") as f:
                for row in df.itertuples(index=False):
                    f.write(json.dumps(self.format_message_line(row), ensure_ascii=False) + "\n")
            logger.info("Wrote %d examples to %s (%s)", len(df), path, name)

        logger.info("JSONL written!")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val/test JSONL datasets for UN resolution citation extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--old", action="store_true",
        help=f"Use {GA_CITATIONS_OLD} instead of {GA_CITATIONS_CURRENT}",
    )
    for name, cfg in KNOWN_SOURCES.items():
        parser.add_argument(f"--{name}", action="store_true", help=f"Also include {name.upper()} citations")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory holding the source CSVs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write the JSONL splits to")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Tokenizer used for length trimming")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKEN_LENGTH, help="Target token length per example")
    parser.add_argument("--token-slack", type=int, default=DEFAULT_TOKEN_SLACK, help="Extra tokens allowed beyond the target before a candidate is rejected")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ga_citations_csv = args.data_dir / (GA_CITATIONS_OLD if args.old else GA_CITATIONS_CURRENT)
    resolutions_csv = args.data_dir / RESOLUTIONS_FILENAME

    extra_sources = [
        CitationSource(name, args.data_dir / cfg.csv_filename, cfg.resolution_column, cfg.id_column)
        for name, cfg in KNOWN_SOURCES.items()
        if getattr(args, name)
    ]
    if extra_sources:
        logger.info("Including extra sources: %s", [s.name for s in extra_sources])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    builder = ResolutionDatasetBuilder(
        resolutions_csv=resolutions_csv,
        ga_citations_csv=ga_citations_csv,
        extra_sources=extra_sources,
        tokenizer=tokenizer,
        max_token_length=args.max_tokens,
        token_slack=args.token_slack,
    )
    builder.run(args.output_dir)


if __name__ == "__main__":
    main()
