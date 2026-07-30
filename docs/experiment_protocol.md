# Experiment protocol
1. Convert each dataset to the common JSONL schema with `graph-rescue convert`.
2. Build or import one frozen, source-grounded graph per corpus.
3. Freeze corpus, graph, query splits, retriever configuration and random
   seeds before training MRV/GRG.
4. Generate and cache initial rankings and candidate frontiers.
5. Train MRV on train queries only.
6. Train gate classifier on train states and calibrate on the held-out train
   fraction. Never use official dev for threshold selection.
7. Evaluate all five policies using identical cached inputs and budgets.
8. Report full data and the four diagnostic slices separately.
9. Use paired bootstrap confidence intervals for method differences and the
   factorial interaction.
10. Repeat training with three random seeds, test a second embedding backbone,
    and repeat graph corruption with five deterministic corruption seeds.
11. Evaluate the same fixed local reader on all 1,000 evaluation questions per
    dataset; stronger readers are desirable but not required for a
    resource-bounded within-model comparison.
12. Compare with an exact published adaptive graph baseline when official code
    and compatible licensing are available; otherwise document the
    availability search and label adaptations explicitly.

## Pre-report checklist

- No supporting-fact annotations entered graph construction or inference.
- No official dev examples entered training or calibration.
- All methods used the same passages, graph and initial ranking.
- Token/action/frontier budgets are identical.
- Embedding and generation model tags plus Ollama version are recorded.
- Hashing fallback is disabled.
- Every graph edge shown in qualitative analysis has source provenance.
- Mean and p95 latency exclude one-time indexing but include query embedding,
  frontier traversal, model scoring and gating.
