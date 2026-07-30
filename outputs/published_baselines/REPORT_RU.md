# Опубликованный baseline: KG²RAG-style equal-budget adaptation

Статус: независимая адаптация переносимых идей KG²RAG, а не точное воспроизведение исходной системы. Semantic seeds расширяются через наш passage/entity graph, кандидаты получают query relevance, propagated seed score и multi-seed support, после чего контекст организуется в seed-центричные группы.

Важное ограничение: исходные triplet KG, relation extraction и FlagReranker из KG²RAG здесь не воспроизводятся. Поэтому корректная формулировка — KG²RAG-style baseline на нашем общем протоколе.

| Dataset | Hybrid FE | KG²-style FE | Gated FE | KG²−Hybrid | Gated−KG² | 95% CI KG²−Hybrid | 95% CI KG²−Gated |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.640 | 0.743 | 0.810 | +0.103 | +0.067 | [0.080, 0.127] | [-0.089, -0.045] |
| 2WikiMultiHopQA | 0.367 | 0.494 | 0.572 | +0.127 | +0.078 | [0.105, 0.150] | [-0.102, -0.053] |
| MuSiQue | 0.156 | 0.202 | 0.252 | +0.046 | +0.050 | [0.027, 0.065] | [-0.067, -0.034] |

## Интерпретация

На всех трех датасетах KG²RAG-style вариант статистически улучшает full-evidence rate относительно hybrid. Одновременно gated MRV стабильно превосходит этот baseline. Следовательно, выигрыш Graph Rescue нельзя объяснить только общей схемой «взять semantic seeds и добавить графовых соседей»: полезны условный gate и обучаемый marginal-value selector.

Все сравнения имеют одинаковые seed_k, final_k, token budget, action budget и исходную hybrid ranking. Нарушений бюджета и расхождений seed-выдачи не обнаружено.

Primary sources:

- https://aclanthology.org/2025.naacl-long.449/
- https://github.com/nju-websoft/KG2RAG
