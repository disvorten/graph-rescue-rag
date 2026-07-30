"""Evaluate gated graph rescue under deterministic graph corruptions.

Hybrid rankings are computed once per dataset and reused because graph
corruption does not alter flat retrieval. The stress test evaluates only the
paired hybrid and primary gated-MRV policies; broader policy comparisons are
reported by the frozen clean experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from graph_rescue.config import ExperimentConfig
from graph_rescue.experiment import Experiment
from graph_rescue.reproducibility import freeze_protocol


CONDITIONS = {
    "clean": {},
    "dropout_10": {"edge_dropout_rate": 0.10},
    "dropout_25": {"edge_dropout_rate": 0.25},
    "dropout_50": {"edge_dropout_rate": 0.50},
    "false_edges_10": {"false_edge_ratio": 0.10},
    "false_edges_25": {"false_edge_ratio": 0.25},
    "false_edges_50": {"false_edge_ratio": 0.50},
    "mixed_25_25": {
        "edge_dropout_rate": 0.25,
        "false_edge_ratio": 0.25,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--clean-config", required=True)
    parser.add_argument(
        "--conditions", nargs="+", default=list(CONDITIONS)
    )
    parser.add_argument(
        "--output-root", default="outputs/final_v1_robustness"
    )
    parser.add_argument(
        "--corruption-seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Graph-corruption seeds. With more than one seed, outputs are "
            "stored under condition/seed_<n> and the summary contains one "
            "row per condition, policy, and seed."
        ),
    )
    args = parser.parse_args()
    clean_path = Path(args.clean_config).resolve()
    clean = ExperimentConfig.load(clean_path).resolve_paths(
        clean_path.parent
    )
    retrieval_experiment = Experiment(clean)
    precomputed_rankings = {}
    for index, example in enumerate(
        retrieval_experiment.eval_queries, start=1
    ):
        started = time.perf_counter()
        ranking = retrieval_experiment.retriever.retrieve(
            example.question,
            clean.retrieval.rerank_k,
        )
        precomputed_rankings[example.id] = (
            ranking,
            (time.perf_counter() - started) * 1000.0,
        )
        if index % 25 == 0 or index == len(
            retrieval_experiment.eval_queries
        ):
            print(
                "ranking precompute progress: "
                f"{index}/{len(retrieval_experiment.eval_queries)} queries",
                flush=True,
            )
    rows = []
    corruption_seeds = args.corruption_seeds or [
        int(clean.graph.corruption_seed)
    ]
    multiple_seeds = len(corruption_seeds) > 1
    clean_seed = corruption_seeds[0]
    for corruption_seed in corruption_seeds:
        for condition in args.conditions:
            if condition not in CONDITIONS:
                raise ValueError(f"Unknown condition: {condition}")
            if condition == "clean" and corruption_seed != clean_seed:
                continue
            value = asdict(clean)
            value["graph"].update(CONDITIONS[condition])
            value["graph"]["corruption_seed"] = corruption_seed
            value["ollama"]["generation_model"] = None
            value["ollama"]["reader_policies"] = []
            value["evaluation"]["include_classic_baselines"] = False
            value["evaluation"]["include_oracle_upper_bound"] = True
            output = Path(args.output_root) / args.dataset / condition
            if multiple_seeds:
                output = output / f"seed_{corruption_seed}"
            output = output.resolve()
            value["output_dir"] = str(output)
            config = ExperimentConfig.from_dict(value)
            freeze_protocol(config, output / "protocol")
            summary = Experiment(config).evaluate(
                precomputed_rankings=precomputed_rankings,
                policy_names_filter={"hybrid", "mrv_gated"},
                compute_slices=False,
            )
            for policy in ("hybrid", "mrv_gated"):
                aggregate = summary["aggregate"][policy]
                rows.append(
                    {
                        "dataset": args.dataset,
                        "condition": condition,
                        "corruption_seed": corruption_seed,
                        "policy": policy,
                        "graph_edges": summary["graph"]["edges"],
                        "dropped_edges": summary["graph"]["dropped_edges"],
                        "false_edges": summary["graph"]["false_edges"],
                        **{
                            metric: aggregate[metric]
                            for metric in (
                                "full_evidence",
                                "support_recall",
                                "graph_actions",
                                "harmful_expansions",
                                "policy_latency_ms",
                            )
                        },
                    }
                )
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "corruption_seed": corruption_seed,
                        "mrv_gated_full_evidence": summary["aggregate"][
                            "mrv_gated"
                        ]["full_evidence"],
                    }
                ),
                flush=True,
            )
    target = (
        Path(args.output_root) / args.dataset / "robustness_summary.json"
    )
    target.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
