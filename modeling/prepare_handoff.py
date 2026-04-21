"""
Prepare evaluator handoff artifacts after manual labeling.

Usage example:
  & ".venv\\Scripts\\python.exe" modeling/prepare_handoff.py ^
      --run-dir modeling/outputs/dev ^
      --labels modeling/outputs/dev/annotation/relevance_labels.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


MODELS = ("tfidf", "bm25", "scibert", "sbert")
RANKED_COLUMNS = {"query_id", "pmid", "rank", "score", "model_name"}
LABEL_COLUMNS = {"query_id", "pmid", "relevance"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _normalize_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["query_id"] = out["query_id"].astype(str).str.strip()
    out["pmid"] = out["pmid"].astype(str).str.strip()
    return out


def validate_labels(labels_path: Path) -> pd.DataFrame:
    labels = _read_csv(labels_path)
    missing = LABEL_COLUMNS - set(labels.columns)
    if missing:
        raise ValueError(f"Labels file missing columns: {sorted(missing)}")
    labels = _normalize_pair_columns(labels)

    empty_relevance = labels["relevance"].isna() | (labels["relevance"].astype(str).str.strip() == "")
    if empty_relevance.any():
        raise ValueError(
            f"Labels contain {int(empty_relevance.sum())} empty relevance values. "
            "Fill all relevance entries before handoff."
        )

    # Force numeric relevance where possible; raises if impossible.
    labels["relevance"] = pd.to_numeric(labels["relevance"], errors="raise")

    duplicated = labels.duplicated(subset=["query_id", "pmid"], keep=False)
    if duplicated.any():
        dup_count = int(duplicated.sum())
        raise ValueError(f"Labels contain duplicated (query_id, pmid) rows: {dup_count}")

    return labels


def validate_ranked_outputs(run_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for model in MODELS:
        path = run_dir / model / f"{model}_top20.csv"
        ranked = _read_csv(path)
        missing = RANKED_COLUMNS - set(ranked.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        files[model] = path
    return files


def build_manifest(run_dir: Path, labels: pd.DataFrame, ranked_files: dict[str, Path]) -> dict:
    query_count = int(labels["query_id"].nunique())
    judged_pairs = int(len(labels))
    relevant_pairs = int((labels["relevance"] > 0).sum())

    return {
        "run_dir": str(run_dir),
        "models": list(MODELS),
        "query_count_in_labels": query_count,
        "judged_pairs": judged_pairs,
        "relevant_pairs": relevant_pairs,
        "ranked_files": {k: str(v) for k, v in ranked_files.items()},
        "artifacts": {
            "labels": "relevance_labels.csv",
            "notes": "HANDOFF_NOTES.md",
            "run_log_if_exists": "run_log.jsonl",
        },
    }


def copy_handoff_package(run_dir: Path, labels_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(labels_path, output_dir / "relevance_labels.csv")

    notes_path = Path("modeling/outputs/HANDOFF_NOTES.md")
    if notes_path.exists():
        shutil.copy2(notes_path, output_dir / "HANDOFF_NOTES.md")

    run_log = run_dir / "run_log.jsonl"
    if run_log.exists():
        shutil.copy2(run_log, output_dir / "run_log.jsonl")

    for model in MODELS:
        src_dir = run_dir / model
        dst_dir = output_dir / model
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in (f"{model}_top10.csv", f"{model}_top20.csv"):
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, dst_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and package retrieval handoff artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run output directory, e.g. modeling/outputs/dev")
    parser.add_argument("--labels", type=Path, required=True, help="Final labeled CSV file")
    parser.add_argument(
        "--handoff-dir",
        type=Path,
        default=Path("modeling/handoff"),
        help="Output handoff package directory",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    labels_path = args.labels.resolve()
    handoff_dir = args.handoff_dir.resolve()

    labels = validate_labels(labels_path)
    ranked_files = validate_ranked_outputs(run_dir)
    copy_handoff_package(run_dir, labels_path, handoff_dir)

    manifest = build_manifest(run_dir, labels, ranked_files)
    manifest_path = handoff_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Handoff package prepared at: {handoff_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Judged pairs: {manifest['judged_pairs']} | Relevant pairs: {manifest['relevant_pairs']}")


if __name__ == "__main__":
    main()

