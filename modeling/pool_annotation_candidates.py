"""
Build a pooled candidate set from four retrieval top-20 outputs for manual annotation.

Reads:
  {output_dir}/{tfidf,bm25,scibert,sbert}/*_top20.csv

Writes (under output_dir/annotation/ by default):
  pooled_for_annotation.csv       — unique (query_id, pmid) + which models + best rank
  relevance_labels_template.csv   — same pairs with empty relevance / notes / annotator
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODELS = ("tfidf", "bm25", "scibert", "sbert")


def read_top20(output_dir: Path, model: str) -> pd.DataFrame:
    path = output_dir / model / f"{model}_top20.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    df = pd.read_csv(path)
    required = {"query_id", "pmid", "rank", "model_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["query_id"] = df["query_id"].astype(str).str.strip()
    df["pmid"] = df["pmid"].astype(str).str.strip()
    df["source_model"] = model
    return df[["query_id", "pmid", "rank", "score", "model_name", "source_model"]]


def build_pool(output_dir: Path, template_sort: str = "pmid_query") -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [read_top20(output_dir, m) for m in MODELS]
    all_rows = pd.concat(frames, ignore_index=True)

    def models_agg(series: pd.Series) -> str:
        return "|".join(sorted(series.unique().tolist()))

    pooled = (
        all_rows.groupby(["query_id", "pmid"], as_index=False)
        .agg(
            best_rank=("rank", "min"),
            max_score=("score", "max"),
            models_retrieved=("source_model", models_agg),
            n_models=("source_model", "nunique"),
        )
        .sort_values(["query_id", "best_rank", "pmid"])
        .reset_index(drop=True)
    )

    template = pooled[["query_id", "pmid"]].copy()
    if template_sort == "pmid_query":
        template = template.sort_values(["pmid", "query_id"]).reset_index(drop=True)
    elif template_sort == "query_pmid":
        template = template.sort_values(["query_id", "pmid"]).reset_index(drop=True)
    else:
        raise ValueError("template_sort must be 'pmid_query' or 'query_pmid'")
    template["relevance"] = ""
    template["notes"] = ""
    template["annotator"] = ""

    return pooled, template


def main() -> None:
    parser = argparse.ArgumentParser(description="Pool top-20 retrieval outputs for annotation.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("modeling/outputs/dev"),
        help="Directory containing tfidf/, bm25/, scibert/, sbert/ subfolders.",
    )
    parser.add_argument(
        "--annotation-subdir",
        type=str,
        default="annotation",
        help="Subfolder under output-dir for pooled CSVs (default: annotation).",
    )
    parser.add_argument(
        "--template-sort",
        choices=["pmid_query", "query_pmid"],
        default="pmid_query",
        help="Sort order for relevance_labels_template.csv (default: pmid_query).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    ann_dir = output_dir / args.annotation_subdir
    ann_dir.mkdir(parents=True, exist_ok=True)

    pooled, template = build_pool(output_dir, template_sort=args.template_sort)

    pool_path = ann_dir / "pooled_for_annotation.csv"
    tmpl_path = ann_dir / "relevance_labels_template.csv"
    pooled.to_csv(pool_path, index=False)
    template.to_csv(tmpl_path, index=False)

    print(f"Wrote {len(pooled)} rows -> {pool_path}")
    print(f"Wrote {len(template)} rows -> {tmpl_path}")


if __name__ == "__main__":
    main()
