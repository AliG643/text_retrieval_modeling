# Evaluation Handoff Notes

## Output Files

Each model writes:

- `modeling/outputs/<model_name>/<model_name>_top10.csv`
- `modeling/outputs/<model_name>/<model_name>_top20.csv`

Schema:

- `query_id`
- `pmid`
- `rank`
- `score`
- `model_name`

## Run Metadata

Per-run metadata is appended to:

- `modeling/outputs/run_log.jsonl`

Includes corpus fingerprint, query count, corpus size, document mode, and data-quality summary.

## Preprocessing Notes

- Lexical (TF-IDF/BM25): lowercase, punctuation cleanup, conservative tokenization, optional stopword filtering.
- Semantic (SciBERT/SBERT): light text cleanup only; biomedical terminology preserved.

## Known Failure Modes To Watch

- Lexical methods may miss semantically-related documents when query vocabulary differs from abstracts.
- Dense methods can drift on short/highly-specific acronym queries.
- Quality of retrieval depends on corpus CSV completeness (`pmid`, `title`, `abstract` all required).

## Pooled annotation (one label file for all models)

After you have all four `*_top20.csv` files under the same run folder (e.g. `modeling/outputs/dev/`), build the pooled pool and an empty label template:

```text
& ".venv\Scripts\python.exe" modeling/pool_annotation_candidates.py --output-dir modeling/outputs/dev
```

This writes:

- `modeling/outputs/dev/annotation/pooled_for_annotation.csv` — unique `(query_id, pmid)` with `best_rank`, `models_retrieved`, etc.
- `modeling/outputs/dev/annotation/relevance_labels_template.csv` — same pairs with empty `relevance`, `notes`, `annotator` columns to fill in.

Fill `relevance` (e.g. `0` / `1`) in one file; evaluation can join that file to each model’s ranked CSV.

## Final packaging for evaluation lead

After labeling is complete and saved as `relevance_labels.csv`, prepare a clean handoff package:

```text
& ".venv\Scripts\python.exe" modeling/prepare_handoff.py --run-dir modeling/outputs/dev --labels modeling/outputs/dev/annotation/relevance_labels.csv
```

This validates label completeness and creates:

- `modeling/handoff/relevance_labels.csv`
- `modeling/handoff/{tfidf,bm25,scibert,sbert}/...top10/top20...`
- `modeling/handoff/HANDOFF_NOTES.md`
- `modeling/handoff/run_log.jsonl` (if present)
- `modeling/handoff/manifest.json`

