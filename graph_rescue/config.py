from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    generation_model: str | None = None
    timeout_seconds: int = 120
    reader_policies: list[str] = field(
        default_factory=lambda: [
            "hybrid",
            "relevance_always",
            "mrv_always",
            "mrv_gated",
        ]
    )


@dataclass
class RetrievalConfig:
    bm25_k: int = 100
    dense_k: int = 100
    rrf_k: int = 60
    rerank_k: int = 50
    seed_k: int = 5
    final_k: int = 10
    evidence_token_budget: int = 2048


@dataclass
class GraphConfig:
    max_hops: int = 2
    frontier_cap: int = 200
    max_actions: int = 5
    min_entity_df: int = 2
    max_entity_df_ratio: float = 0.20
    entity_mode: str = "provided"
    edge_dropout_rate: float = 0.0
    false_edge_ratio: float = 0.0
    corruption_seed: int = 42


@dataclass
class LearningConfig:
    epochs: int = 350
    learning_rate: float = 0.05
    l2: float = 0.01
    calibration_fraction: float = 0.10
    gate_target_recall: float = 0.95
    gate_calibration_method: str = "auto"
    gate_calibration_folds: int = 5
    gate_threshold_bootstrap_samples: int = 400
    gate_threshold_quantile: float = 0.10
    counterfactual_labels_path: str | None = None
    reader_gain_weight: float = 0.50
    token_cost_weight: float = 0.08
    hop_cost_weight: float = 0.05
    noise_cost_weight: float = 0.10
    random_seed: int = 42


@dataclass
class EvaluationConfig:
    include_classic_baselines: bool = False
    include_oracle_upper_bound: bool = False
    bootstrap_samples: int = 2000
    latency_percentiles: list[float] = field(
        default_factory=lambda: [0.50, 0.95]
    )
    resume_checkpoint: bool = True
    progress_every: int = 25


@dataclass
class ExperimentConfig:
    corpus_path: str
    train_queries_path: str
    eval_queries_path: str
    output_dir: str = "outputs/experiment"
    cache_dir: str = "work/cache"
    model_dir: str = "work/models"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            corpus_path=value["corpus_path"],
            train_queries_path=value["train_queries_path"],
            eval_queries_path=value["eval_queries_path"],
            output_dir=value.get("output_dir", "outputs/experiment"),
            cache_dir=value.get("cache_dir", "work/cache"),
            model_dir=value.get("model_dir", "work/models"),
            ollama=OllamaConfig(**value.get("ollama", {})),
            retrieval=RetrievalConfig(**value.get("retrieval", {})),
            graph=GraphConfig(**value.get("graph", {})),
            learning=LearningConfig(**value.get("learning", {})),
            evaluation=EvaluationConfig(**value.get("evaluation", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def resolve_paths(self, root: str | Path) -> "ExperimentConfig":
        root_path = Path(root).resolve()

        def resolved(value: str) -> str:
            path = Path(value)
            return str(path if path.is_absolute() else root_path / path)

        data = asdict(self)
        for name in (
            "corpus_path",
            "train_queries_path",
            "eval_queries_path",
            "output_dir",
            "cache_dir",
            "model_dir",
        ):
            data[name] = resolved(data[name])
        labels_path = data["learning"].get("counterfactual_labels_path")
        if labels_path:
            data["learning"]["counterfactual_labels_path"] = resolved(labels_path)
        return ExperimentConfig.from_dict(data)
