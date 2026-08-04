# План доведения Graph Rescue RAG до подачи в Q2-журнал

Дата актуализации: 4 августа 2026 г.

## Цель

Основная площадка — **Journal of Intelligent Information Systems (JIIS)**.
Это не обещание принятия и не утверждение постоянного квартиля: перед подачей
квартиль проверяется в нужной базе и категории за конкретный год. Выбор JIIS
основан прежде всего на scope: intelligent information retrieval, интеграция
AI и database technologies, knowledge representation и reasoning under
uncertainty.

На дату проверки институциональный экспорт JCR 2024 указывает Q2 для
`Computer Science, Information Systems`, тогда как карточка Scopus 2024
Wageningen University указывает Q1. Поэтому формулировка «Q2-журнал» допустима
только с указанием JCR, года и категории; в рукопись квартиль не включается.

Рабочее название:

> Selective Local Graph Rescue after Hybrid Retrieval: Calibrated Evidence
> Completion for Multi-Hop Question Answering

## Центральный claim

После сильного BM25+dense+fusion retrieval локальное графовое дополнение может
восстановить отсутствующую часть evidence chain. Оно запускается только при
прогнозируемой неполноте и выбирает кандидата по условной marginal rescue
value. Claim ограничен фиксированным passage/token budget и проверенными
экспериментальными протоколами.

Не заявляется:

- SOTA;
- превосходство над Microsoft GraphRAG, HippoRAG, HHS-RAG или KG²RAG в целом;
- full-Wikipedia evaluation;
- универсальное ускорение GraphRAG;
- независимость эффекта от reader, графа и датасета.

## Что уже доказано в основном controlled protocol

В disjoint pooled-corpus evaluation по 1 000 запросов на датасет gated MRV
увеличил full-evidence rate относительно общего hybrid retriever:

- HotpotQA: 0.640 → 0.810, +0.170;
- 2WikiMultiHopQA: 0.367 → 0.572, +0.205;
- MuSiQue: 0.156 → 0.252, +0.096.

Эффект проверен на нескольких training seeds и двух embedding backbones.
Equal-budget KG²RAG-style control показывает, что generic graph expansion
объясняет часть выигрыша, но не весь observed gap. Qwen3-8B reader даёт
значимый Answer-F1 gain на HotpotQA и 2Wiki; прирост на MuSiQue после
коррекции незначим. False-edge corruption вреднее сопоставимого edge dropout.

Эти результаты дают внутреннее подтверждение гипотезы. Ниже перечислены уже
выполненные проверки, которые добавляют внешнюю и вычислительную валидность.

## Выполненные дополнительные проверки

| Блок | Фактический объём | Итог |
|---|---|---|
| Global transfer | 17 375 unseen official-dev queries, 3 датасета | FE gain +0.073…+0.116; все paired CI исключают ноль |
| Official baseline | 1 000 unique IDs, 11 656 released MuSiQue passages | Gated MRV − HippoRAG FE@7 = +0.097 [0.071; 0.125] |
| Equal-budget comparison | Общий released corpus и `k=7` | Graph Rescue FE@7 0.366; HippoRAG 0.269; StandardRAG 0.227 |
| Latency | 200 queries × 3 повтора на датасет | Gated − always-on MRV: -6.5…-12.0 мс paired mean |
| Failure funnel | Anchor/reachability/gate/selector strata | Counts и effect сохранены в generated table |
| Robustness | Missing и false edges, 5 seeds | False edges устойчиво вреднее dropout |
| Reproducibility | Frozen configs, compact outputs, fingerprint v2 | 58 tests проходят; release audit выполняется перед push |

Ключевой отрицательный результат: retrieval gain переносится, но селективность
gate переносится хуже. В official-code сравнении gate открывается на 99.8%
запросов; target recalibration в global protocol также часто поднимает open
rate выше 0.89. Поэтому статья не обещает универсальную compute-экономию.

## Решения по результатам global transfer

1. **Положительный перенос на трёх датасетах.** Вынести transfer table в
   основной текст; pooled experiment оставить как controlled mechanism study.
2. **Положительный перенос только на части датасетов.** Сузить claim и объяснить
   различия через graph reachability, support count и gate open rate.
3. **Нулевой перенос.** Не скрывать результат; позиционировать работу как
   исследование переобучения локального rescue и условий применимости.
4. **Отрицательный перенос.** Не подавать текущую версию как improvement paper;
   сначала реализовать edge confidence/domain calibration и повторить protocol.

## Структура рукописи JIIS

1. Introduction: структурный failure mode hybrid retrieval и точные вклады.
2. Related Work: hybrid IR, multi-hop retrieval, GraphRAG/KG-guided RAG,
   adaptive/cost-aware retrieval.
3. Method: graph construction, candidate paths, MRV objective, preflight и
   continuation gates, equal-budget constraints.
4. Experimental Design: datasets, splits, corpora, baselines, metrics,
   statistical tests, hardware and models.
5. Results: controlled retrieval, global transfer, official-code baseline,
   reader metrics и latency.
6. Error and Robustness Analysis: failure funnel, graph corruption и
   qualitative cases.
7. Limitations: not full-wiki, lightweight graph, local 8B reader, baseline
   alignment and lifecycle boundary.
8. Reproducibility and declarations.
9. Conclusion: подтверждённые и неподтверждённые выводы.

## Таблицы основного текста

Чтобы не превысить 25 страниц и сохранить читаемость:

1. datasets/corpus/protocol summary;
2. primary pooled retrieval results;
3. global unseen-dev transfer;
4. aligned official-code baseline;
5. reader results;
6. compact latency/resource comparison;
7. failure funnel или corruption robustness.

Seed-level, embedding-backbone, calibration-bin и полный corruption grid можно
вынести в supplementary material или GitHub aggregates.

## Пакет подачи

Локальная папка подачи должна содержать плоский набор:

- `main.tex`;
- `references.bib`;
- `svjour3.cls`, `svglov3.clo`, bibliography style;
- детерминированно построенные figure files;
- сгенерированные `.tex`-таблицы;
- `main.pdf`;
- cover letter как отдельный файл для submission form.

Нужно подтвердить до отправки:

- точное англоязычное название подразделения МГУ;
- почтовый адрес, если он должен быть опубликован;
- отсутствие или наличие специального финансирования;
- отсутствие или наличие competing interests;
- единоличный author contribution либо список соавторов с реальным вкладом.

## Резервная стратегия

Резервную площадку выбирают после получения редакционного решения, а не заранее
по одному названию. Если JIIS отклоняет работу по scope, нужен более широкий
IR/information-systems журнал. Если по новизне или экспериментам — сначала
исправляются указанные научные недостатки. При каждой переподаче заново
проверяются scope, квартиль, индексация, APC и требования к формату.

Полная русская статья до решения JIIS не публикуется отдельно: русский текст
используется как рабочая теория и подробное объяснение экспериментов, чтобы не
создавать риск duplicate publication.
