from pathlib import Path
import pandas as pd
import math

MODELS = ["tfidf", "bm25", "scibert", "sbert"]

def load_ranked(path):
    df = pd.read_csv(path)
    df["query_id"] = df["query_id"].astype(str).str.strip()
    df["pmid"] = df["pmid"].astype(str).str.strip()
    return df

def load_labels(path):
    df = pd.read_csv(path)
    df["query_id"] = df["query_id"].astype(str).str.strip()
    df["pmid"] = df["pmid"].astype(str).str.strip()
    df["relevance"] = pd.to_numeric(df["relevance"])
    df["binary_relevance"] = (df["relevance"] >= 1).astype(int)
    return df

def precision_at_k(merged, k):
    scores = []
    for qid in merged["query_id"].unique():
        sub = merged[merged["query_id"] == qid].sort_values("rank").head(k)
        if len(sub) == 0:
            continue
        scores.append(sub["binary_relevance"].sum() / k)
    return sum(scores) / len(scores) if scores else 0.0

def recall_at_k(merged, labels, k):
    scores = []
    for qid in merged["query_id"].unique():
        sub = merged[merged["query_id"] == qid].sort_values("rank").head(k)
        total_relevant = labels[labels["query_id"] == qid]["binary_relevance"].sum()
        if total_relevant == 0:
            continue
        found_relevant = sub["binary_relevance"].sum()
        scores.append(found_relevant / total_relevant)
    return sum(scores) / len(scores) if scores else 0.0

def dcg_at_k(rels, k):
    rels = rels[:k]
    total = 0.0
    for i, rel in enumerate(rels, start=1):
        total += (2**rel - 1) / math.log2(i + 1)
    return total

def ndcg_at_k(merged, labels, k):
    scores = []
    for qid in merged["query_id"].unique():
        sub = merged[merged["query_id"] == qid].sort_values("rank").head(k)
        actual_rels = sub["relevance"].tolist()

        ideal_rels = labels[labels["query_id"] == qid]["relevance"].sort_values(ascending=False).tolist()[:k]

        dcg = dcg_at_k(actual_rels, k)
        idcg = dcg_at_k(ideal_rels, k)

        if idcg == 0:
            continue
        scores.append(dcg / idcg)
    return sum(scores) / len(scores) if scores else 0.0

def evaluate_model(run_dir, model, labels):
    ranked_path = Path(run_dir) / model / f"{model}_top20.csv"
    ranked = load_ranked(ranked_path)

    merged = ranked.merge(
        labels[["query_id", "pmid", "relevance", "binary_relevance"]],
        on=["query_id", "pmid"],
        how="left"
    )

    merged["relevance"] = merged["relevance"].fillna(0)
    merged["binary_relevance"] = merged["binary_relevance"].fillna(0)

    return {
        "model": model,
        "P@5": round(precision_at_k(merged, 5), 4),
        "R@10": round(recall_at_k(merged, labels, 10), 4),
        "nDCG@10": round(ndcg_at_k(merged, labels, 10), 4),
    }

def main():
    run_dir = "modeling/outputs/dev"
    labels_path = "modeling/outputs/dev/annotation/relevance_labels.csv"

    labels = load_labels(labels_path)

    results = []
    for model in MODELS:
        results.append(evaluate_model(run_dir, model, labels))

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv("modeling/outputs/dev/evaluation/evaluation_results.csv", index=False)


if __name__ == "__main__":
    main()