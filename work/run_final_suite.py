from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, stdev

from graph_rescue.config import ExperimentConfig
from graph_rescue.experiment import Experiment
from graph_rescue.reproducibility import freeze_protocol


def safe_name(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def generated_config(
    base: ExperimentConfig,
    *,
    dataset: str,
    embedding_model: str,
    seed: int,
    root: Path,
) -> ExperimentConfig:
    value = asdict(base)
    embedding_name = safe_name(embedding_model)
    value["ollama"]["embedding_model"] = embedding_model
    value["learning"]["random_seed"] = seed
    value["graph"]["corruption_seed"] = seed
    value["output_dir"] = str(
        root / dataset / embedding_name / f"seed_{seed}"
    )
    value["model_dir"] = str(
        (
            Path("work/final_protocol")
            / dataset
            / "models"
            / embedding_name
            / f"seed_{seed}"
        ).resolve()
    )
    if embedding_model == base.ollama.embedding_model:
        value["cache_dir"] = base.cache_dir
    else:
        value["cache_dir"] = str(
            Path("work/final_protocol/cache") / embedding_name
        )
    return ExperimentConfig.from_dict(value)


def aggregate_seed_summaries(
    records: list[dict],
) -> dict:
    policies = sorted(
        {
            policy
            for record in records
            for policy in record["summary"]["aggregate"]
        }
    )
    result: dict[str, dict[str, dict[str, float]]] = {}
    for policy in policies:
        metric_names = sorted(
            set.intersection(
                *[
                    set(record["summary"]["aggregate"][policy])
                    for record in records
                    if policy in record["summary"]["aggregate"]
                ]
            )
        )
        result[policy] = {}
        for metric in metric_names:
            values = [
                float(record["summary"]["aggregate"][policy][metric])
                for record in records
                if policy in record["summary"]["aggregate"]
            ]
            result[policy][metric] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "runs": float(len(values)),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=["qwen3-embedding:0.6b", "bge-m3:latest"],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[101, 202, 303]
    )
    parser.add_argument(
        "--output-root", default="outputs/final_v1"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_path = Path(args.base_config).resolve()
    base = ExperimentConfig.load(base_path).resolve_paths(base_path.parent)
    output_root = Path(args.output_root).resolve()
    records = []
    generated_root = (
        Path("work/final_protocol/generated_configs") / args.dataset
    )
    generated_root.mkdir(parents=True, exist_ok=True)

    for embedding_model in args.embedding_models:
        for seed in args.seeds:
            config = generated_config(
                base,
                dataset=args.dataset,
                embedding_model=embedding_model,
                seed=seed,
                root=output_root,
            )
            config_path = (
                generated_root
                / f"{safe_name(embedding_model)}_seed_{seed}.json"
            )
            config.save(config_path)
            summary_path = Path(config.output_dir) / "summary.json"
            protocol_dir = Path(config.output_dir) / "protocol"
            freeze_protocol(config, protocol_dir)
            if summary_path.exists() and not args.force:
                summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
                status = "reused"
            else:
                _, summary = Experiment(config).run()
                status = "completed"
            records.append(
                {
                    "dataset": args.dataset,
                    "embedding_model": embedding_model,
                    "seed": seed,
                    "status": status,
                    "config": str(config_path),
                    "summary_path": str(summary_path),
                    "summary": summary,
                }
            )

    aggregate = {}
    for embedding_model in args.embedding_models:
        selected = [
            record
            for record in records
            if record["embedding_model"] == embedding_model
        ]
        aggregate[embedding_model] = aggregate_seed_summaries(selected)
    result = {
        "dataset": args.dataset,
        "embedding_models": args.embedding_models,
        "seeds": args.seeds,
        "runs": [
            {key: value for key, value in record.items() if key != "summary"}
            for record in records
        ],
        "aggregate_across_seeds": aggregate,
    }
    target = output_root / args.dataset / "suite_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
