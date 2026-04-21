from __future__ import annotations

import pandas as pd


def topk_overlap(results_a: pd.DataFrame, results_b: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    overlaps = []
    for query_id in sorted(set(results_a["query_id"]).intersection(set(results_b["query_id"]))):
        a_docs = set(results_a[(results_a["query_id"] == query_id) & (results_a["rank"] <= top_k)]["pmid"])
        b_docs = set(results_b[(results_b["query_id"] == query_id) & (results_b["rank"] <= top_k)]["pmid"])
        union = a_docs.union(b_docs)
        overlap = len(a_docs.intersection(b_docs))
        jaccard = overlap / len(union) if union else 0.0
        overlaps.append({"query_id": query_id, "top_k": top_k, "overlap_count": overlap, "jaccard": jaccard})
    return pd.DataFrame(overlaps)


def score_distribution_summary(results: pd.DataFrame) -> dict:
    return {
        "min": float(results["score"].min()),
        "max": float(results["score"].max()),
        "mean": float(results["score"].mean()),
        "median": float(results["score"].median()),
        "std": float(results["score"].std(ddof=0)),
    }

