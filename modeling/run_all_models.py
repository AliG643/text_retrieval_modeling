from __future__ import annotations

import argparse

from run_retrieval import load_config, run_single_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all retrieval models in sequence.")
    parser.add_argument("--config", default="modeling/src/config/defaults.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    for model_name in ["tfidf", "bm25", "scibert", "sbert"]:
        run_single_model(model_name=model_name, config=config)


if __name__ == "__main__":
    main()

