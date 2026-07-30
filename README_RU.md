# Graph Rescue RAG

[English README](README.md)

![Graph Rescue RAG: гибридный поиск и калиброванное локальное дополнение по графу](docs/assets/graph-rescue-social-preview-1280x640.png)

Исследовательский код и зафиксированные экспериментальные артефакты для
**селективного графового дополнения после гибридного поиска**.

Сначала BM25, dense retrieval, reciprocal-rank fusion и feature reranker
формируют компактный исходный контекст. Затем система рассматривает только
локальный passage/entity graph вокруг найденных элементов:

- **Marginal Rescue Value (MRV)** оценивает кандидата по ожидаемому вкладу с
  учётом уже найденного evidence, а не только по сходству с запросом;
- **Graph Rescuability Gate (GRG)** решает, следует ли начинать или продолжать
  graph expansion.

Итоговый passage/token budget совпадает с flat hybrid baseline.

## Как устроен метод

![Подробная схема Graph Rescue RAG: гибридный поиск, калиброванный gate, локальное расширение графа, MRV-selector и evidence pack фиксированного размера](docs/assets/graph-rescue-architecture-detailed-v2.png)

Граф используется как условный ремонт уже выполненного поиска:

1. BM25 и dense retrieval строят взаимодополняющие выдачи; RRF и feature
   reranker формируют top-\(k\) исходных passages.
2. Калиброванный gate оценивает, не пропущена ли часть supporting chain. Если
   gate закрыт, система возвращает исходный hybrid context без graph traversal.
3. При открытом gate обходится только ограниченная окрестность исходных
   passages глубиной 1–3 перехода. Для путей сохраняются provenance и
   edge-confidence.
4. MRV ранжирует кандидатов условно относительно уже найденных evidence:
   учитываются relevance, novelty, path confidence, redundancy и стоимость.
5. Выбранные graph neighbors заменяют менее полезные passages при неизменном
   passage/token budget, после чего evidence pack передаётся reader-модели.

Подробный баннер создан генеративной моделью как поясняющий материал для
репозитория. Он не используется как научный рисунок в рукописи и не является
экспериментальным свидетельством.

## На чём выполнены прогоны

Основные эксперименты выполнялись локально через Ollama на Windows-ноутбуке с
NVIDIA RTX 4070 Laptop GPU (8 GB VRAM). Ниже сравниваются метрики качества.
Зафиксированные latency пригодны для диагностики внутри одного запуска, но не
выдаются за hardware-normalized сравнение с внешними реализациями GraphRAG.

| Компонент | Зафиксированная настройка |
|---|---|
| Датасеты | HotpotQA, 2WikiMultiHopQA, MuSiQue |
| Размер split | 1 000 train + 1 000 непересекающихся eval-вопросов на датасет |
| Основные embeddings | `qwen3-embedding:0.6b` через Ollama |
| Проверка представления | `bge-m3:latest`, один seed |
| Reader | локальный `qwen3:8b`, одинаковые prompt и decoding для обоих контекстов |
| Seed обучения | 101, 202, 303 |
| Seed повреждения графа | 101, 202, 303, 404, 505 |
| Hybrid retrieval | BM25 + dense + RRF + feature reranker |
| Контекст | 2 seeds; 5 passages для Hotpot/2Wiki, 7 для MuSiQue |
| Evidence budget | 1 800 токенов для Hotpot/2Wiki; 2 400 для MuSiQue |
| Graph budget | до 2 actions и 2 hops; для MuSiQue — 3 actions и 3 hops |
| Статистика | 5 000 paired bootstrap samples; поправка Холма для reader-метрик |

## Сравнение при одинаковом retrieval budget

Замороженный протокол использует 1 000 train и 1 000 непересекающихся eval
вопросов на датасет. Full-evidence rate для основного
`qwen3-embedding:0.6b`, seed 101:

| Датасет | Hybrid | KG²RAG-style | Gated MRV | Δ к Hybrid | 95% paired CI |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 0.640 | 0.743 | **0.810** | +0.170 | [0.143, 0.197] |
| 2WikiMultiHopQA | 0.367 | 0.494 | **0.572** | +0.205 | [0.178, 0.231] |
| MuSiQue | 0.156 | 0.202 | **0.252** | +0.096 | [0.074, 0.118] |

KG²RAG-style — независимая равнобюджетная адаптация опубликованного паттерна,
а не точный запуск официальной реализации KG²RAG.

Во всех сравниваемых вариантах совпадают initial hybrid ranking, `seed_k`,
`final_k`, token budget и graph-action budget:

| Система | Доступ к графу | Выбор кандидатов | Роль |
|---|---|---|---|
| Hybrid | отсутствует | плоская reranked-выдача | сильный baseline без графа |
| KG²RAG-style | безусловное локальное расширение | query relevance + propagated seed score + multi-seed support | равнобюджетный published-pattern control |
| Graph Rescue / gated MRV | условное локальное расширение | calibrated gate + marginal value относительно текущего evidence | предлагаемый метод |
| Oracle upper bound | только gold-aware диагностика | лучшее достижимое evidence при том же budget | оценка резерва, не deployable baseline |

### Результат с reader-моделью

Один и тот же локальный Qwen3-8B был применён к обоим контекстам на всех 1 000
eval-вопросах каждого датасета.

| Датасет | Hybrid Answer F1 | Graph Rescue Answer F1 | Δ | 95% paired CI | Holm p |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 0.423 | **0.465** | +0.042 | [0.020; 0.064] | 0.0024 |
| 2WikiMultiHopQA | 0.232 | **0.307** | +0.075 | [0.054; 0.097] | <0.0002 |
| MuSiQue | 0.133 | 0.151 | +0.018 | [0.001; 0.036] | 0.126 |

Эффект сохраняет направление при трёх seed обучения и замене embeddings на
BGE-M3. Controlled corruption показывает, что ложные graph edges вредят
сильнее отсутствующих. На полном eval по 1 000 запросов фиксированный локальный
Qwen3-8B reader улучшает Answer F1 на +0.042 для HotpotQA и +0.075 для
2WikiMultiHopQA. Для MuSiQue прирост +0.018 положителен, но не сохраняет
значимость после поправки Холма. Эти downstream-результаты являются
resource-bounded диагностикой и анализируются отдельно от основного retrieval
claim.

## Установка и smoke test

Требуется Python 3.10+ и локальный Ollama.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
ollama pull qwen3-embedding:0.6b
python -m graph_rescue doctor --config examples/demo_config.json
python -m graph_rescue run --config examples/demo_config.json
python -m unittest discover -s tests -v
```

Hashing fallback предназначен только для unit/smoke tests и не используется
для результатов статьи.

## Воспроизведение

- [DATA.md](DATA.md) — источники, лицензии, checksums и исключённые данные;
- [REPRODUCING.md](REPRODUCING.md) — полный порядок запусков;
- [outputs/REPRODUCIBILITY_MANIFEST.md](outputs/REPRODUCIBILITY_MANIFEST.md) —
  зафиксированные идентификаторы протокола;
- [outputs/FINAL_EXPERIMENT_REPORT_RU.md](outputs/FINAL_EXPERIMENT_REPORT_RU.md)
  — подробный отчёт;
- [outputs/PUBLICATION_PLAN_RU.md](outputs/PUBLICATION_PLAN_RU.md) —
  публикационный план;
- [outputs/PUBLICATION_AND_GITHUB_STRATEGY_RU_v2.md](outputs/PUBLICATION_AND_GITHUB_STRATEGY_RU_v2.md)
  — актуальный выбор между русским и английским маршрутом и план продвижения
  GitHub;
- [docs/russian_manuscript_plan.md](docs/russian_manuscript_plan.md) — каркас
  русскоязычной журнальной статьи.

Репозиторий намеренно не содержит benchmark records, derived passage text,
model/embedding caches, checkpoints и per-query reader generations.
Полные Word/LaTeX-черновики также остаются локальными до письменного
подтверждения политики препринтов выбранного журнала.

## Ограничения

- Это pooled-corpus controlled study, а не официальный open-domain leaderboard.
- Абсолютные значения нельзя напрямую сравнивать с leaderboard-таблицами других
  работ.
- Gate уменьшает graph actions относительно always-expand proxy, но это ещё не
  доказывает превосходство над полным GraphRAG по end-to-end latency или cost.
- Человеческий trace audit и внешний reproduction остаются обязательными перед
  журнальной подачей.

## Использование ИИ

Генеративные AI-инструменты интенсивно использовались при прототипировании,
оркестрации экспериментов, анализе и подготовке исходного черновика. Это не
заменяет человеческое научное авторство. Перед подачей человек должен проверить
код, данные, статистику, ссылки и traces, повторить ключевой эксперимент,
заново сформулировать смысловые разделы и принять ответственность за все claims.
AI-система не указывается автором.

Результаты ещё требуют независимой проверки автором перед архивным релизом или
журнальной подачей.

## Лицензия

Код проекта распространяется по Apache License 2.0. Лицензии upstream
benchmark и моделей остаются независимыми; см. [DATA.md](DATA.md).

Метаданные релиза в [CITATION.cff](CITATION.cff) указывают Максима Одинцова как
человека-автора. Перед архивным релизом и журнальной подачей необходимо
повторить тесты и основной эксперимент, а также проверить научные выводы и
лицензии.
