from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .adapters import convert_dataset
from .config import ExperimentConfig
from .counterfactual import generate_counterfactual_labels
from .experiment import Experiment
from .ollama import OllamaClient
from .official_export import export_official_predictions
from .official_runner import score_official_predictions
from .reproducibility import freeze_protocol


def load_config(path: str) -> ExperimentConfig:
    config_path = Path(path).resolve()
    return ExperimentConfig.load(config_path).resolve_paths(config_path.parent)


def print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def evaluation_model_artifacts(config: ExperimentConfig) -> dict[str, str | None]:
    """Resolve the frozen files needed by ``evaluate`` without building indexes."""
    model_dir = Path(config.model_dir)
    preflight = model_dir / "preflight_gate_model.json"
    if not preflight.exists():
        preflight = model_dir / "gate_model.json"
    continuation = model_dir / "continue_gate_model.json"
    return {
        "mrv": str(model_dir / "mrv_model.json"),
        "preflight_gate": str(preflight),
        "continue_gate": str(continuation) if continuation.exists() else None,
    }


def validate_evaluation_model_artifacts(config: ExperimentConfig) -> dict[str, str | None]:
    artifacts = evaluation_model_artifacts(config)
    missing = [
        path
        for name, path in artifacts.items()
        if name != "continue_gate" and path is not None and not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Frozen evaluation model artifacts are missing: "
            + ", ".join(missing)
        )
    return artifacts


def command_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    client = OllamaClient(
        base_url=config.ollama.base_url,
        timeout_seconds=config.ollama.timeout_seconds,
    )
    models = client.models()
    report = {
        "ollama_url": config.ollama.base_url,
        "ollama_models": models,
        "embedding_model_ready": config.ollama.embedding_model in models,
        "generation_model_ready": (
            config.ollama.generation_model in models
            if config.ollama.generation_model
            else None
        ),
        "paths": {
            "corpus": config.corpus_path,
            "train_queries": config.train_queries_path,
            "eval_queries": config.eval_queries_path,
            "model_dir": config.model_dir,
        },
        "evaluation_model_artifacts": evaluation_model_artifacts(config),
    }
    print_json(report)
    return 0 if report["embedding_model_ready"] else 2


def make_experiment(args: argparse.Namespace) -> Experiment:
    return Experiment(
        load_config(args.config),
        allow_hashing_fallback=getattr(args, "hashing_fallback", False),
    )


def command_train(args: argparse.Namespace) -> int:
    experiment = make_experiment(args)
    _, _, _, summary = experiment.train()
    print_json({"doctor": experiment.doctor(), "training": asdict(summary)})
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    validate_evaluation_model_artifacts(config)
    experiment = Experiment(
        config,
        allow_hashing_fallback=getattr(args, "hashing_fallback", False),
    )
    print_json(experiment.evaluate())
    return 0


def command_run(args: argparse.Namespace) -> int:
    experiment = make_experiment(args)
    training, evaluation = experiment.run()
    print_json(
        {
            "doctor": experiment.doctor(),
            "training": asdict(training),
            "evaluation": evaluation,
        }
    )
    return 0


def command_convert(args: argparse.Namespace) -> int:
    print_json(
        convert_dataset(
            args.format,
            args.input,
            args.corpus_output,
            args.queries_output,
        )
    )
    return 0


def command_counterfactual(args: argparse.Namespace) -> int:
    experiment = make_experiment(args)
    print_json(
        generate_counterfactual_labels(
            experiment,
            args.output,
            max_queries=args.max_queries,
            max_candidates=args.max_candidates,
        )
    )
    return 0


def command_freeze(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = freeze_protocol(config, args.output)
    print_json(
        {
            "protocol_id": result["protocol_id"],
            "output": str(Path(args.output).resolve()),
            "audit": result["audit"],
        }
    )
    return 0 if result["audit"]["passed"] else 3


def command_export_official(args: argparse.Namespace) -> int:
    print_json(
        export_official_predictions(
            dataset=args.dataset,
            query_results_path=args.results,
            queries_path=args.queries,
            output_path=args.output,
            policy=args.policy,
            reader=args.reader,
        )
    )
    return 0


def command_score_official(args: argparse.Namespace) -> int:
    print_json(
        score_official_predictions(
            dataset=args.dataset,
            prediction_path=args.predictions,
            gold_path=args.gold,
            output_path=args.output,
            alias_path=args.aliases,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graph-rescue",
        description="Cost-aware graph rescue after hybrid retrieval.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler in (
        ("doctor", command_doctor),
        ("train", command_train),
        ("evaluate", command_evaluate),
        ("run", command_run),
    ):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--config", default="examples/demo_config.json", help="Experiment JSON file"
        )
        if name != "doctor":
            command.add_argument(
                "--hashing-fallback",
                action="store_true",
                help="Use deterministic test embeddings if Ollama is unavailable. "
                "Never use this flag for reported experiments.",
            )
        command.set_defaults(handler=handler)
    convert = subparsers.add_parser("convert")
    convert.add_argument("--format", choices=["hotpot", "2wiki", "musique"], required=True)
    convert.add_argument("--input", required=True)
    convert.add_argument("--corpus-output", required=True)
    convert.add_argument("--queries-output", required=True)
    convert.set_defaults(handler=command_convert)
    counterfactual = subparsers.add_parser("label-counterfactual")
    counterfactual.add_argument("--config", required=True)
    counterfactual.add_argument("--output", required=True)
    counterfactual.add_argument("--max-queries", type=int, default=60)
    counterfactual.add_argument("--max-candidates", type=int, default=3)
    counterfactual.set_defaults(handler=command_counterfactual)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(handler=command_freeze)
    export = subparsers.add_parser("export-official")
    export.add_argument(
        "--dataset",
        choices=["hotpot", "2wiki", "musique"],
        required=True,
    )
    export.add_argument("--results", required=True)
    export.add_argument("--queries", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--policy", required=True)
    export.add_argument("--reader", required=True)
    export.set_defaults(handler=command_export_official)
    score = subparsers.add_parser("score-official")
    score.add_argument(
        "--dataset",
        choices=["hotpot", "2wiki", "musique"],
        required=True,
    )
    score.add_argument("--predictions", required=True)
    score.add_argument("--gold", required=True)
    score.add_argument("--aliases")
    score.add_argument("--output", required=True)
    score.set_defaults(handler=command_score_official)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
