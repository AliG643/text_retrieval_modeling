# Retrieval Modeling (Ali Guo Scope)

This module implements four retrieval methods over PubMed-style CSV input:

- TF-IDF
- BM25
- SciBERT dense retrieval
- SBERT dense retrieval

## Input Contracts

- Corpus CSV: `pmid`, `title`, `abstract`
- Query CSV: `query_id`, `query_text`, optional `topic_label`

## Output Contract

Every model exports ranked rows with:

- `query_id`
- `pmid`
- `rank`
- `score`
- `model_name`

## Quickstart

1. Install dependencies:
   - `pip install -r modeling/requirements.txt`
2. Update config if needed:
   - `modeling/src/config/defaults.yaml`
3. Run one model:
   - `python modeling/run_retrieval.py --model tfidf`
4. Run all models:
   - `python modeling/run_all_models.py`

## Artifacts

- Standard outputs: `modeling/outputs/<model_name>/<model_name>_top{k}.csv`
- Run log: `modeling/outputs/run_log.jsonl`
- Embedding cache: `modeling/cache/embeddings/`

