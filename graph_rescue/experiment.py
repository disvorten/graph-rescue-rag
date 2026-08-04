from __future__ import annotations

import json
from dataclasses import asdict
import hashlib
from pathlib import Path
import time
from typing import Any

from .config import ExperimentConfig
from .features import CandidateFeatureExtractor
from .graph import KnowledgeGraph
from .hybrid import HybridRetriever, make_embedder
from .io import load_passages, load_queries, write_jsonl
from .learning import GateModel, MRVModel
from .metrics import (
    aggregate_rows,
    binary_metrics,
    calibration_bins,
    factorial_interaction,
    holm_bonferroni,
    paired_bootstrap_difference,
    retrieval_metrics,
)
from .models import (
    Passage,
    QueryExample,
    ReaderPrediction,
    RetrievedPassage,
    RetrievalTrace,
)
from .ollama import OllamaClient
from .official_metrics import (
    best_answer_scores,
    evidence_scores,
    joint_scores,
    support_fact_scores,
)
from .policy import PolicyConfig, RescuePolicy
from .profiling import process_rss_bytes
from .reader import AnswerPresenceReader, OllamaReader
from .report import write_experiment_report
from .text import estimate_tokens
from .training import TrainingSummary, train_models


class Experiment:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        allow_hashing_fallback: bool = False,
    ):
        self.config = config
        initialization_started = time.perf_counter()
        stage_started = initialization_started
        self.passages_list = load_passages(config.corpus_path)
        corpus_loaded = time.perf_counter()
        self.passages: dict[str, Passage] = {
            item.id: item for item in self.passages_list
        }
        self.train_queries = load_queries(config.train_queries_path)
        self.eval_queries = load_queries(config.eval_queries_path)
        self._validate_support_ids(self.train_queries + self.eval_queries)
        queries_loaded = time.perf_counter()
        embedder = make_embedder(
            base_url=config.ollama.base_url,
            model=config.ollama.embedding_model,
            timeout_seconds=config.ollama.timeout_seconds,
            cache_dir=config.cache_dir,
            allow_hashing_fallback=allow_hashing_fallback,
        )
        embedder_ready = time.perf_counter()
        self.retriever = HybridRetriever(
            self.passages_list,
            embedder,
            bm25_k=config.retrieval.bm25_k,
            dense_k=config.retrieval.dense_k,
            rrf_k=config.retrieval.rrf_k,
            rerank_k=config.retrieval.rerank_k,
        )
        retriever_ready = time.perf_counter()
        embedding_cache_stats = (
            embedder.stats() if hasattr(embedder, "stats") else None
        )
        self.graph = KnowledgeGraph.build(
            self.passages_list,
            min_entity_df=config.graph.min_entity_df,
            max_entity_df_ratio=config.graph.max_entity_df_ratio,
            entity_mode=config.graph.entity_mode,
        )
        if (
            config.graph.edge_dropout_rate > 0.0
            or config.graph.false_edge_ratio > 0.0
        ):
            self.graph.corrupt(
                edge_dropout_rate=config.graph.edge_dropout_rate,
                false_edge_ratio=config.graph.false_edge_ratio,
                seed=config.graph.corruption_seed,
            )
        graph_ready = time.perf_counter()
        self.extractor = CandidateFeatureExtractor(
            passages=self.passages,
            retriever=self.retriever,
            token_budget=config.retrieval.evidence_token_budget,
        )
        extractor_ready = time.perf_counter()
        self.initialization_profile = {
            "corpus_load_ms": (corpus_loaded - stage_started) * 1000.0,
            "query_load_validate_ms": (queries_loaded - corpus_loaded) * 1000.0,
            "embedder_connect_ms": (embedder_ready - queries_loaded) * 1000.0,
            "retriever_index_ms": (retriever_ready - embedder_ready) * 1000.0,
            "embedding_cache": embedding_cache_stats,
            "graph_build_ms": (graph_ready - retriever_ready) * 1000.0,
            "feature_extractor_ms": (extractor_ready - graph_ready) * 1000.0,
            "total_ms": (extractor_ready - initialization_started) * 1000.0,
            "process_rss_bytes_after_init": process_rss_bytes(),
        }

    def _validate_support_ids(self, queries: list[QueryExample]) -> None:
        known = set(self.passages)
        unknown = sorted(
            {
                item
                for query in queries
                for item in query.supporting_passage_ids
                if item not in known
            }
        )
        if unknown:
            raise ValueError(f"Unknown supporting passage IDs: {unknown[:10]}")

    def doctor(self) -> dict[str, Any]:
        client = OllamaClient(
            self.config.ollama.base_url, self.config.ollama.timeout_seconds
        )
        return {
            "ollama_models": client.models(),
            "embedding_model": self.config.ollama.embedding_model,
            "generation_model": self.config.ollama.generation_model,
            "passages": len(self.passages),
            "train_queries": len(self.train_queries),
            "eval_queries": len(self.eval_queries),
            "graph": asdict(self.graph.stats),
            "initialization_profile": self.initialization_profile,
        }

    def train(self) -> tuple[MRVModel, GateModel, GateModel, TrainingSummary]:
        return train_models(
            self.train_queries,
            self.retriever,
            self.graph,
            self.passages,
            self.extractor,
            self.config,
        )

    def _policy_config(self) -> PolicyConfig:
        return PolicyConfig(
            seed_k=self.config.retrieval.seed_k,
            final_k=self.config.retrieval.final_k,
            token_budget=self.config.retrieval.evidence_token_budget,
            max_hops=self.config.graph.max_hops,
            frontier_cap=self.config.graph.frontier_cap,
            max_actions=self.config.graph.max_actions,
            token_cost_weight=self.config.learning.token_cost_weight,
            hop_cost_weight=self.config.learning.hop_cost_weight,
            noise_cost_weight=self.config.learning.noise_cost_weight,
            reader_gain_weight=self.config.learning.reader_gain_weight,
        )

    def policies(
        self,
        mrv_model: MRVModel,
        preflight_gate_model: GateModel,
        continue_gate_model: GateModel,
    ) -> list[RescuePolicy]:
        common = dict(
            passages=self.passages,
            graph=self.graph,
            feature_extractor=self.extractor,
            config=self._policy_config(),
        )
        result = [
            RescuePolicy(name="hybrid", selector="none", **common),
            RescuePolicy(
                name="relevance_always",
                selector="relevance",
                mrv_model=mrv_model,
                **common,
            ),
            RescuePolicy(
                name="mrv_always", selector="mrv", mrv_model=mrv_model, **common
            ),
            RescuePolicy(
                name="relevance_gated",
                selector="relevance",
                mrv_model=mrv_model,
                preflight_gate_model=preflight_gate_model,
                continue_gate_model=continue_gate_model,
                **common,
            ),
            RescuePolicy(
                name="mrv_gated",
                selector="mrv",
                mrv_model=mrv_model,
                preflight_gate_model=preflight_gate_model,
                continue_gate_model=continue_gate_model,
                **common,
            ),
        ]
        if self.config.evaluation.include_oracle_upper_bound:
            result.append(
                RescuePolicy(
                    name="oracle_upper_bound",
                    selector="oracle",
                    **common,
                )
            )
        return result

    def _slice(
        self,
        example: QueryExample,
        seed_ids: list[str],
        *,
        top_ten: set[str] | None = None,
    ) -> str:
        support = set(example.supporting_passage_ids)
        if top_ten is None:
            top_ten = {
                item.passage_id
                for item in self.retriever.retrieve(
                    example.question, self.config.retrieval.final_k
                )
            }
        if support and support.issubset(top_ten):
            return "already_solved"
        if not support & set(seed_ids):
            return "anchor_failure"
        candidates, _ = self.graph.candidate_paths(
            seed_ids,
            excluded_passage_ids=seed_ids,
            max_hops=self.config.graph.max_hops,
            cap=self.config.graph.frontier_cap,
        )
        reachable = {item.target_passage_id for item in candidates}
        return "rescuable" if (support - set(seed_ids)) & reachable else "unreachable"

    def _oracle_initial_rescuable(
        self, example: QueryExample, seed_ids: list[str]
    ) -> bool:
        return self._oracle_rescuable(example, seed_ids)

    def _oracle_rescuable(
        self, example: QueryExample, evidence_ids: list[str]
    ) -> bool:
        missing_support = set(example.supporting_passage_ids) - set(evidence_ids)
        if not missing_support:
            return False
        candidates, _ = self.graph.candidate_paths(
            evidence_ids,
            excluded_passage_ids=evidence_ids,
            max_hops=self.config.graph.max_hops,
            cap=self.config.graph.frontier_cap,
        )
        reachable = {item.target_passage_id for item in candidates}
        return bool(missing_support & reachable)

    def _readers(self) -> list[Any]:
        readers: list[Any] = [AnswerPresenceReader()]
        model = self.config.ollama.generation_model
        if model:
            client = OllamaClient(
                self.config.ollama.base_url, self.config.ollama.timeout_seconds
            )
            if model not in client.models():
                raise RuntimeError(
                    f"Generation model {model!r} is not installed in Ollama"
                )
            readers.append(
                OllamaReader(
                    client,
                    model,
                    cache_dir=Path(self.config.cache_dir) / "reader",
                )
            )
        return readers

    def _classic_trace(
        self,
        example: QueryExample,
        ranking: list[Any],
        name: str,
    ) -> RetrievalTrace:
        start = time.perf_counter()
        final_ids: list[str] = []
        tokens = 0
        for item in ranking:
            cost = estimate_tokens(self.passages[item.passage_id].full_text)
            if final_ids and (
                tokens + cost
                > self.config.retrieval.evidence_token_budget
            ):
                continue
            final_ids.append(item.passage_id)
            tokens += cost
            if len(final_ids) >= self.config.retrieval.final_k:
                break
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RetrievalTrace(
            query_id=example.id,
            policy=name,
            seed_passage_ids=[
                item.passage_id
                for item in ranking[: self.config.retrieval.seed_k]
            ],
            final_passage_ids=final_ids,
            actions=[],
            latency_ms=latency_ms,
            graph_reads=0,
            candidate_paths_scored=0,
            evidence_tokens=tokens,
            policy_latency_ms=latency_ms,
        )

    def _evaluation_fingerprint(self, policy_names: list[str]) -> str:
        """Identify the exact evaluation whose query bundles may be resumed."""

        package_dir = Path(__file__).resolve().parent
        source_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package_dir.glob("*.py"))
        }
        input_paths = {
            "corpus": self.config.corpus_path,
            "train_queries": self.config.train_queries_path,
            "eval_queries": self.config.eval_queries_path,
            "model_dir": self.config.model_dir,
        }
        if self.config.learning.counterfactual_labels_path:
            input_paths["counterfactual_labels"] = (
                self.config.learning.counterfactual_labels_path
            )
        input_hashes = {
            name: self._content_hash(Path(value))
            for name, value in input_paths.items()
        }
        payload = {
            "schema_version": 2,
            "config": asdict(self.config),
            "eval_query_ids": [item.id for item in self.eval_queries],
            "policies": policy_names,
            "source_hashes": source_hashes,
            "input_hashes": input_hashes,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _content_hash(path: Path) -> str:
        """Hash a required file or a directory tree deterministically."""

        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_dir():
            digest = hashlib.sha256()
            files = sorted(item for item in path.rglob("*") if item.is_file())
            for item in files:
                relative = item.relative_to(path).as_posix().encode("utf-8")
                digest.update(relative)
                digest.update(b"\0")
                digest.update(hashlib.sha256(item.read_bytes()).digest())
            return digest.hexdigest()
        raise FileNotFoundError(f"Fingerprint input does not exist: {path}")

    @staticmethod
    def _load_evaluation_checkpoint(
        path: Path,
        *,
        expected_policies: set[str],
    ) -> dict[str, dict[str, Any]]:
        """Load complete query bundles and ignore only a truncated final line."""

        if not path.exists():
            return {}
        lines = path.read_text(encoding="utf-8").splitlines()
        bundles: dict[str, dict[str, Any]] = {}
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                bundle = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break
                raise ValueError(
                    f"Corrupt evaluation checkpoint at {path}:{index + 1}"
                )
            query_id = str(bundle.get("query_id", ""))
            rows = bundle.get("rows", [])
            traces = bundle.get("traces", [])
            row_policies = {str(item.get("policy")) for item in rows}
            trace_policies = {str(item.get("policy")) for item in traces}
            if (
                query_id
                and len(rows) == len(expected_policies)
                and len(traces) == len(expected_policies)
                and row_policies == expected_policies
                and trace_policies == expected_policies
            ):
                bundles[query_id] = bundle
        return bundles

    @staticmethod
    def _append_evaluation_bundle(path: Path, bundle: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(bundle, ensure_ascii=False) + "\n")
            handle.flush()

    def evaluate(
        self,
        mrv_model: MRVModel | None = None,
        preflight_gate_model: GateModel | None = None,
        continue_gate_model: GateModel | None = None,
        *,
        precomputed_rankings: dict[
            str, tuple[list[RetrievedPassage], float]
        ]
        | None = None,
        policy_names_filter: set[str] | None = None,
        compute_slices: bool = True,
    ) -> dict[str, Any]:
        if mrv_model is None:
            mrv_model = MRVModel.load(Path(self.config.model_dir) / "mrv_model.json")
        model_dir = Path(self.config.model_dir)
        if preflight_gate_model is None:
            preflight_path = model_dir / "preflight_gate_model.json"
            if not preflight_path.exists():
                preflight_path = model_dir / "gate_model.json"
            preflight_gate_model = GateModel.load(preflight_path)
        if continue_gate_model is None:
            continue_path = model_dir / "continue_gate_model.json"
            continue_gate_model = (
                GateModel.load(continue_path)
                if continue_path.exists()
                else preflight_gate_model
            )
        policies = self.policies(
            mrv_model, preflight_gate_model, continue_gate_model
        )
        if policy_names_filter is not None:
            policies = [
                policy
                for policy in policies
                if policy.name in policy_names_filter
            ]
            missing_policies = policy_names_filter - {
                policy.name for policy in policies
            }
            if missing_policies:
                raise ValueError(
                    "Unknown policy names requested: "
                    f"{sorted(missing_policies)}"
                )
            if not policies:
                raise ValueError("At least one evaluation policy is required")
        readers = self._readers()
        rows: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "evaluation_checkpoint.jsonl"
        checkpoint_meta_path = output_dir / "evaluation_checkpoint_meta.json"
        policy_names = [policy.name for policy in policies]
        if self.config.evaluation.include_classic_baselines:
            policy_names.extend(("bm25", "dense", "rrf_fusion"))
        fingerprint = self._evaluation_fingerprint(policy_names)
        if checkpoint_meta_path.exists():
            checkpoint_meta = json.loads(
                checkpoint_meta_path.read_text(encoding="utf-8")
            )
            if checkpoint_meta.get("fingerprint") != fingerprint:
                raise ValueError(
                    "Evaluation checkpoint belongs to a different config. "
                    f"Remove {checkpoint_path} and {checkpoint_meta_path}, "
                    "or use a new output_dir."
                )
        else:
            checkpoint_meta_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "fingerprint": fingerprint,
                        "expected_policies": policy_names,
                        "eval_queries": len(self.eval_queries),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        completed = (
            self._load_evaluation_checkpoint(
                checkpoint_path,
                expected_policies=set(policy_names),
            )
            if self.config.evaluation.resume_checkpoint
            else {}
        )
        eval_ids = {item.id for item in self.eval_queries}
        completed = {
            query_id: bundle
            for query_id, bundle in completed.items()
            if query_id in eval_ids
        }
        for example in self.eval_queries:
            if example.id in completed:
                rows.extend(completed[example.id]["rows"])
                traces.extend(completed[example.id]["traces"])
        if completed:
            print(
                f"evaluation resume: {len(completed)}/{len(self.eval_queries)} "
                "queries restored",
                flush=True,
            )

        for example_index, example in enumerate(self.eval_queries, start=1):
            if example.id in completed:
                continue
            query_rows: list[dict[str, Any]] = []
            query_traces: list[dict[str, Any]] = []
            if (
                precomputed_rankings is not None
                and example.id in precomputed_rankings
            ):
                cached_ranking, hybrid_retrieval_ms = precomputed_rankings[
                    example.id
                ]
                ranking = list(cached_ranking)
            else:
                retrieval_started = time.perf_counter()
                ranking = self.retriever.retrieve(
                    example.question, self.config.retrieval.rerank_k
                )
                hybrid_retrieval_ms = (
                    time.perf_counter() - retrieval_started
                ) * 1000.0
            trace_runs: list[tuple[str, RetrievalTrace, float]] = []
            for policy in policies:
                trace = policy.run(example, ranking)
                trace_runs.append(
                    (policy.name, trace, hybrid_retrieval_ms)
                )
            if compute_slices:
                top_ten = {
                    item.passage_id
                    for item in ranking[: self.config.retrieval.final_k]
                }
                slice_label = self._slice(
                    example,
                    trace_runs[0][1].seed_passage_ids,
                    top_ten=top_ten,
                )
            else:
                slice_label = "not_computed"
            if self.config.evaluation.include_classic_baselines:
                for mode in ("bm25", "dense", "rrf_fusion"):
                    retrieval_started = time.perf_counter()
                    classic_ranking = self.retriever.retrieve_mode(
                        example.question,
                        k=self.config.retrieval.rerank_k,
                        mode=mode,
                    )
                    retrieval_ms = (
                        time.perf_counter() - retrieval_started
                    ) * 1000.0
                    trace_runs.append(
                        (
                            mode,
                            self._classic_trace(
                                example, classic_ranking, mode
                            ),
                            retrieval_ms,
                        )
                    )
            for policy_name, trace, retrieval_ms in trace_runs:
                reader_started = time.perf_counter()
                predictions_by_reader: dict[str, ReaderPrediction] = {
                    reader.name: reader.predict(
                        example, trace.final_passage_ids, self.passages
                    )
                    for reader in readers
                    if not isinstance(reader, OllamaReader)
                    or policy_name in self.config.ollama.reader_policies
                }
                reader_latency_ms = (
                    time.perf_counter() - reader_started
                ) * 1000.0
                trace.retrieval_latency_ms = retrieval_ms
                trace.policy_latency_ms = trace.latency_ms
                trace.reader_latency_ms = reader_latency_ms
                trace.total_latency_ms = (
                    retrieval_ms + trace.latency_ms + reader_latency_ms
                )
                predicted_by_reader = {
                    name: prediction.answer
                    for name, prediction in predictions_by_reader.items()
                }
                proxy_prediction = predicted_by_reader["answer_presence"]
                metrics = retrieval_metrics(example, trace, proxy_prediction)
                metrics.update(
                    {
                        "retrieval_latency_ms": trace.retrieval_latency_ms,
                        "policy_latency_ms": trace.policy_latency_ms,
                        "reader_latency_ms": trace.reader_latency_ms,
                        "total_latency_ms": trace.total_latency_ms,
                    }
                )
                reader_evidence: dict[str, dict[str, Any]] = {}
                for name, prediction in predictions_by_reader.items():
                    if name == "answer_presence":
                        continue
                    answer = best_answer_scores(
                        prediction.answer, example.answers
                    )
                    for metric_name, value in answer.items():
                        metrics[f"{name}:answer_{metric_name}"] = value
                    if example.supporting_facts:
                        support = support_fact_scores(
                            prediction.supporting_facts,
                            example.supporting_facts,
                        )
                        for metric_name, value in support.items():
                            metrics[f"{name}:support_{metric_name}"] = value
                        joint = joint_scores(answer, support)
                        for metric_name, value in joint.items():
                            metrics[f"{name}:joint_{metric_name}"] = value
                    if example.supporting_paragraph_indices:
                        paragraph_index_by_passage = dict(
                            example.passage_indices
                        )
                        predicted_indices = [
                            paragraph_index_by_passage[passage_id]
                            for passage_id in prediction.supporting_passage_ids
                            if passage_id in paragraph_index_by_passage
                        ]
                        paragraph_support = support_fact_scores(
                            [("", index) for index in predicted_indices],
                            [
                                ("", index)
                                for index in example.supporting_paragraph_indices
                            ],
                        )
                        for metric_name, value in paragraph_support.items():
                            metrics[
                                f"{name}:paragraph_support_{metric_name}"
                            ] = value
                    if example.evidence_triples:
                        evidence = evidence_scores(
                            prediction.evidence_triples,
                            example.evidence_triples,
                        )
                        for metric_name, value in evidence.items():
                            metrics[f"{name}:evidence_{metric_name}"] = value
                    reader_evidence[name] = {
                        "supporting_facts": [
                            list(item) for item in prediction.supporting_facts
                        ],
                        "supporting_passage_ids": list(
                            prediction.supporting_passage_ids
                        ),
                        "evidence_triples": [
                            list(item) for item in prediction.evidence_triples
                        ],
                    }
                query_rows.append(
                    {
                        "query_id": example.id,
                        "question": example.question,
                        "question_type": example.question_type,
                        "dataset": example.dataset,
                        "support_count": len(
                            example.supporting_passage_ids
                        ),
                        "policy": policy_name,
                        "slice": slice_label,
                        "supporting_ids": list(example.supporting_passage_ids),
                        "retrieved_ids": trace.final_passage_ids,
                        "predictions": predicted_by_reader,
                        "reader_evidence": reader_evidence,
                        "metrics": metrics,
                    }
                )
                query_traces.append(trace.to_dict())
            bundle = {
                "query_id": example.id,
                "rows": query_rows,
                "traces": query_traces,
            }
            self._append_evaluation_bundle(checkpoint_path, bundle)
            rows.extend(query_rows)
            traces.extend(query_traces)
            progress_every = max(1, self.config.evaluation.progress_every)
            if (
                example_index % progress_every == 0
                or example_index == len(self.eval_queries)
            ):
                print(
                    f"evaluation progress: {example_index}/"
                    f"{len(self.eval_queries)} queries",
                    flush=True,
                )

        hybrid_by_query = {
            row["query_id"]: row for row in rows if row["policy"] == "hybrid"
        }
        for row in rows:
            baseline = hybrid_by_query[row["query_id"]]
            support = set(row["supporting_ids"])
            baseline_retrieved = set(baseline["retrieved_ids"])
            retrieved = set(row["retrieved_ids"])
            missing_from_baseline = support - baseline_retrieved
            newly_recovered = missing_from_baseline & retrieved
            row["metrics"]["graph_support_gain"] = (
                row["metrics"]["support_recall"]
                - baseline["metrics"]["support_recall"]
            )
            row["metrics"]["graph_query_rescued"] = float(
                row["metrics"]["full_evidence"] > baseline["metrics"]["full_evidence"]
            )
            row["metrics"]["graph_query_harmed"] = float(
                row["metrics"]["support_recall"]
                < baseline["metrics"]["support_recall"]
            )
            row["metrics"]["rescue_recall"] = (
                len(newly_recovered) / len(missing_from_baseline)
                if missing_from_baseline
                else 0.0
            )

        aggregate = aggregate_rows(rows)
        by_policy = {
            policy: sorted(
                [row for row in rows if row["policy"] == policy],
                key=lambda item: item["query_id"],
            )
            for policy in aggregate
        }

        def values(policy: str, metric: str) -> list[float]:
            return [row["metrics"][metric] for row in by_policy[policy]]

        comparisons = {}
        if {"hybrid", "mrv_gated"}.issubset(by_policy):
            comparisons.update(
                {
                    "mrv_gated_vs_hybrid_full_evidence": (
                        paired_bootstrap_difference(
                            values("mrv_gated", "full_evidence"),
                            values("hybrid", "full_evidence"),
                            samples=self.config.evaluation.bootstrap_samples,
                            seed=self.config.learning.random_seed,
                        )
                    ),
                    "mrv_gated_vs_hybrid_support_recall": (
                        paired_bootstrap_difference(
                            values("mrv_gated", "support_recall"),
                            values("hybrid", "support_recall"),
                            samples=self.config.evaluation.bootstrap_samples,
                            seed=self.config.learning.random_seed,
                        )
                    ),
                }
            )
        if {"relevance_always", "mrv_always"}.issubset(by_policy):
            comparisons["mrv_vs_relevance_full_evidence"] = (
                paired_bootstrap_difference(
                    values("mrv_always", "full_evidence"),
                    values("relevance_always", "full_evidence"),
                    samples=self.config.evaluation.bootstrap_samples,
                    seed=self.config.learning.random_seed,
                )
            )
        if {"mrv_always", "mrv_gated"}.issubset(by_policy):
            comparisons["gated_vs_always_graph_actions"] = (
                paired_bootstrap_difference(
                    values("mrv_gated", "graph_actions"),
                    values("mrv_always", "graph_actions"),
                    samples=self.config.evaluation.bootstrap_samples,
                    seed=self.config.learning.random_seed,
                )
            )
        if self.config.evaluation.include_classic_baselines:
            for baseline in ("bm25", "dense", "rrf_fusion"):
                comparisons[
                    f"hybrid_vs_{baseline}_full_evidence"
                ] = paired_bootstrap_difference(
                    values("hybrid", "full_evidence"),
                    values(baseline, "full_evidence"),
                    samples=self.config.evaluation.bootstrap_samples,
                    seed=self.config.learning.random_seed,
                )
        reader_metric_names = sorted(
            {
                metric
                for row in by_policy["hybrid"]
                for metric in row["metrics"]
                if (
                    ":answer_" in metric
                    or ":support_" in metric
                    or ":joint_" in metric
                    or ":paragraph_support_" in metric
                    or ":evidence_" in metric
                )
            }
            & {
                metric
                for row in by_policy["mrv_gated"]
                for metric in row["metrics"]
            }
        )
        for metric in reader_metric_names:
            if all(
                metric in row["metrics"]
                for policy in ("hybrid", "mrv_gated")
                for row in by_policy[policy]
            ):
                comparisons[
                    f"mrv_gated_vs_hybrid_{metric}"
                ] = paired_bootstrap_difference(
                    values("mrv_gated", metric),
                    values("hybrid", metric),
                    samples=self.config.evaluation.bootstrap_samples,
                    seed=self.config.learning.random_seed,
                )
        adjusted_p_values = holm_bonferroni(
            {
                name: value["p_value_two_sided"]
                for name, value in comparisons.items()
            }
        )
        for name, adjusted in adjusted_p_values.items():
            comparisons[name]["p_value_holm"] = adjusted
        interaction_policies = {
            "relevance_always",
            "mrv_always",
            "relevance_gated",
            "mrv_gated",
        }
        interactions = (
            {
                metric: factorial_interaction(
                    values("relevance_always", metric),
                    values("mrv_always", metric),
                    values("relevance_gated", metric),
                    values("mrv_gated", metric),
                    samples=self.config.evaluation.bootstrap_samples,
                    seed=self.config.learning.random_seed,
                )
                for metric in (
                    "full_evidence",
                    "support_recall",
                    "graph_actions",
                    "latency_ms",
                )
            }
            if interaction_policies.issubset(by_policy)
            else {}
        )

        gate_probabilities: dict[str, list[float]] = {
            "preflight": [],
            "continue": [],
        }
        gate_labels: dict[str, list[int]] = {"preflight": [], "continue": []}
        examples_by_id = {item.id: item for item in self.eval_queries}
        gated_rows = by_policy["mrv_gated"]
        for row, trace in zip(
            gated_rows,
            sorted(
                [item for item in traces if item["policy"] == "mrv_gated"],
                key=lambda item: item["query_id"],
            ),
        ):
            for action in trace["actions"]:
                stage = action.get("gate_stage")
                probability = action.get("gate_probability")
                if stage not in gate_probabilities or probability is None:
                    continue
                gate_probabilities[stage].append(float(probability))
                evidence_ids = list(
                    action.get("evidence_ids_before")
                    or trace["seed_passage_ids"]
                )
                gate_labels[stage].append(
                    int(
                        self._oracle_rescuable(
                            examples_by_id[row["query_id"]], evidence_ids
                        )
                    )
                )
        gate_metrics_by_stage = {
            "preflight": binary_metrics(
                gate_probabilities["preflight"],
                gate_labels["preflight"],
                threshold=preflight_gate_model.threshold,
            ),
            "continue": binary_metrics(
                gate_probabilities["continue"],
                gate_labels["continue"],
                threshold=continue_gate_model.threshold,
            ),
        }
        gate_calibration_bins = {
            stage: calibration_bins(
                gate_probabilities[stage], gate_labels[stage]
            )
            for stage in ("preflight", "continue")
        }

        summary = {
            "config": asdict(self.config),
            "graph": asdict(self.graph.stats),
            "initialization_profile": self.initialization_profile,
            "aggregate": aggregate,
            "comparisons": comparisons,
            "factorial_interactions": interactions,
            "gate_metrics": gate_metrics_by_stage["preflight"],
            "gate_metrics_by_stage": gate_metrics_by_stage,
            "gate_calibration_bins": gate_calibration_bins,
            "gate_model": {
                "calibration_method": (
                    preflight_gate_model.selected_calibration_method
                ),
                "calibration_cv_log_loss": (
                    preflight_gate_model.calibration_scores
                ),
                "threshold": preflight_gate_model.threshold,
            },
            "continue_gate_model": {
                "calibration_method": (
                    continue_gate_model.selected_calibration_method
                ),
                "calibration_cv_log_loss": (
                    continue_gate_model.calibration_scores
                ),
                "threshold": continue_gate_model.threshold,
            },
            "reader_stats": {
                reader.name: reader.stats()
                for reader in readers
                if isinstance(reader, OllamaReader)
            },
            "evaluation_checkpoint": {
                "fingerprint": fingerprint,
                "completed_queries": len(completed)
                + sum(item.id not in completed for item in self.eval_queries),
                "path": str(checkpoint_path),
            },
            "slice_counts": {
                name: sum(
                    row["slice"] == name
                    for row in rows
                    if row["policy"] == "hybrid"
                )
                for name in (
                    "already_solved",
                    "anchor_failure",
                    "rescuable",
                    "unreachable",
                )
            },
        }
        write_jsonl(output_dir / "query_results.jsonl", rows)
        write_jsonl(output_dir / "retrieval_traces.jsonl", traces)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_experiment_report(summary, output_dir)
        return summary

    def run(self) -> tuple[TrainingSummary, dict[str, Any]]:
        mrv, preflight_gate, continue_gate, training_summary = self.train()
        return training_summary, self.evaluate(
            mrv, preflight_gate, continue_gate
        )
