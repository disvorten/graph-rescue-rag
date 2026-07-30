from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_rescue.config import ExperimentConfig
from graph_rescue.experiment import Experiment
from graph_rescue.official_export import export_official_predictions
from graph_rescue.official_runner import score_official_predictions
from graph_rescue.reproducibility import freeze_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("hotpot", "2wiki", "musique"), required=True
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--output-root", default="outputs/final_v1_reader"
    )
    args = parser.parse_args()

    base_path = Path(args.base_config).resolve()
    base = ExperimentConfig.load(base_path).resolve_paths(base_path.parent)
    value = asdict(base)
    dataset_root = (
        Path("work/final_protocol") / args.dataset
    ).resolve()
    value["eval_queries_path"] = str(
        dataset_root / f"reader_eval_{args.size}.jsonl"
    )
    value["model_dir"] = str(
        (
            dataset_root
            / "models"
            / "qwen3-embedding_0.6b"
            / f"seed_{args.seed}"
        ).resolve()
    )
    output_dir = (
        Path(args.output_root) / args.dataset / f"seed_{args.seed}"
    ).resolve()
    value["output_dir"] = str(output_dir)
    value["ollama"]["reader_policies"] = ["hybrid", "mrv_gated"]
    value["evaluation"]["include_classic_baselines"] = False
    value["evaluation"]["include_oracle_upper_bound"] = False
    value["evaluation"]["bootstrap_samples"] = 5000
    config = ExperimentConfig.from_dict(value)
    freeze_protocol(config, output_dir / "protocol")
    summary_path = output_dir / "summary.json"
    query_results_path = output_dir / "query_results.jsonl"
    if summary_path.exists() and query_results_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print("reader evaluation artifacts restored; skipping generation")
    else:
        summary = Experiment(config).evaluate()

    official = {}
    gold_suffix = "jsonl" if args.dataset == "musique" else "json"
    gold_path = (
        dataset_root / f"reader_eval_{args.size}_gold.{gold_suffix}"
    )
    for policy in ("hybrid", "mrv_gated"):
        prediction_path = (
            output_dir
            / f"official_{args.dataset}_{policy}."
            f"{'jsonl' if args.dataset == 'musique' else 'json'}"
        )
        export_official_predictions(
            dataset=args.dataset,
            query_results_path=query_results_path,
            queries_path=config.eval_queries_path,
            output_path=prediction_path,
            policy=policy,
            reader="ollama:qwen3:8b",
        )
        official[policy] = score_official_predictions(
            dataset=args.dataset,
            prediction_path=prediction_path,
            gold_path=gold_path,
            alias_path=(
                Path("work/datasets/2wiki_official/id_aliases.json")
                if args.dataset == "2wiki"
                else None
            ),
            output_path=output_dir / f"official_scores_{policy}.json",
        )
    result = {
        "dataset": args.dataset,
        "size": args.size,
        "summary": summary,
        "official": official,
    }
    (output_dir / "reader_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
