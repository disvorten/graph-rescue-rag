# Graph Rescue RAG: итоговый экспериментальный отчёт

Версия: 3.0

Дата фиксации: 30 июля 2026 г.
Статус: завершённый локальный экспериментальный snapshot, ещё не submission-ready статья
## Краткий вывод

Главная гипотеза получила устойчивую поддержку в контролируемой постановке: локальное графовое расширение после обычного hybrid retrieval действительно находит часть supporting evidence, которую плоский поиск пропускает. На 1 000 eval-вопросах каждого из трёх датасетов gated MRV повысил full-evidence rate относительно hybrid:

| Датасет | Hybrid | Gated MRV | Разность | 95% paired bootstrap CI | Oracle |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 0.640 | 0.810 | +0.170 | [0.143; 0.197] | 0.932 |
| 2WikiMultiHopQA | 0.367 | 0.572 | +0.205 | [0.178; 0.231] | 0.753 |
| MuSiQue | 0.156 | 0.252 | +0.096 | [0.074; 0.118] | 0.438 |

Эффект воспроизводится при смене embedding-backbone на BGE-M3 и при трёх seed обучения. На полном прогоне по 1 000 eval-вопросов на датасет улучшение evidence сопровождается статистически значимым повышением Answer F1 у локального Qwen3-8B reader на HotpotQA и 2Wiki; на MuSiQue положительный сдвиг не сохраняет значимость после поправки Холма.

Это уже достаточная основа для методологической статьи о selective graph rescue. Однако текущие числа нельзя выдавать за официальный leaderboard или SOTA: корпуса и splits образуют наш замороженный pooled-corpus протокол, а reader-оценка является controlled resource-bounded comparison с одной локальной моделью.

## 1. Что проверялось

Исходный pipeline:

1. BM25 и dense retrieval формируют два списка кандидатов.
2. Reciprocal Rank Fusion объединяет списки.
3. Лёгкий reranker формирует seeds и базовый контекст.
4. Поиск в source-grounded passage graph раскрывает только локальный frontier вокруг найденных seeds.
5. Marginal Rescue Value, или MRV, оценивает условную полезность кандидата относительно уже найденного evidence.
6. Graph Rescuability Gate решает, следует ли запускать или продолжать графовый обход.
7. Итоговый контекст остаётся в том же evidence budget, что и baseline.

Ключевой объект исследования — не «любой GraphRAG», а rescue-событие: baseline не собрал полную supporting chain, а ограниченное графовое действие добавило недостающее доказательство.

## 2. Реализованная система

### 2.1. Flat retrieval

- BM25: top-150 для HotpotQA и 2Wiki, top-180 для MuSiQue.
- Dense retriever: Qwen3 Embedding 0.6B как primary backbone.
- BGE-M3: независимая проверка чувствительности representation.
- RRF и feature reranking: top-80 или top-100 в зависимости от датасета.
- Seeds: 2 passage.
- Финальный контекст: 5 passage для HotpotQA/2Wiki и 7 для MuSiQue.
- Evidence budget: 1 800 или 2 400 токенов.

### 2.2. Граф

Узлы — passages и нормализованные сущности/заголовки. Рёбра получены из предоставленных entity annotations, заголовков и их упоминаний. Gold supporting facts не использовались при построении графа. Для каждого пути сохраняются provenance, длина, confidence и hubness-признаки.

Размеры замороженных графов:

| Датасет | Passages | Рёбра | Max hops | Max actions |
|---|---:|---:|---:|---:|
| HotpotQA | 19 189 | 144 586 | 2 | 2 |
| 2WikiMultiHopQA | 11 347 | 74 676 | 2 | 2 |
| MuSiQue | 23 630 | 180 645 | 3 | 3 |

### 2.3. MRV

MRV является условным selector: кандидат оценивается не только по сходству с вопросом, но и по тому, что уже присутствует в контексте. Модель использует признаки покрытия evidence gap, redundancy, пути, hubness, стоимости токенов, вероятности reader gain и риска harmful expansion.

Практическая utility имеет вид:

`utility = predicted evidence gain + reader gain - token cost - hop cost - noise risk`.

Counterfactual reader-supervision было получено с Qwen3-8B на 60 train-вопросах каждого датасета: один и тот же вопрос решался с кандидатом и без кандидата. Это дополнительный, а не единственный supervision signal.

### 2.4. Gate

Gate калибруется на отдельной части train и оптимизируется с ограничением на recall потенциально rescuable запросов. На primary run:

| Датасет | Gate AUROC | ECE | Доля положительных решений | Среднее число actions |
|---|---:|---:|---:|---:|
| HotpotQA | 0.736 | 0.049 | 0.914 | 1.57 |
| 2WikiMultiHopQA | 0.813 | 0.055 | 0.832 | 1.52 |
| MuSiQue | 0.651 | 0.068 | 0.991 | 2.69 |

Gate экономит примерно 10-23% graph actions относительно MRV-always, почти не теряя full evidence. На MuSiQue gate пока слаб: он пропускает почти все запросы и экономит мало. Это честный отрицательный результат и естественная цель следующего улучшения.

## 3. Протокол данных

Для каждого датасета использованы 1 000 train и 1 000 непересекающихся eval-вопросов. Из passages выбранных вопросов построен единый pooled corpus. Для каждого протокола сохранены:

- input hashes;
- frozen configuration;
- code-tree hash;
- версии Ollama-моделей;
- leakage audit;
- protocol ID;
- per-query traces и checkpoints.

Protocol IDs:

- HotpotQA: `02fea6d4337bd7105f38aba29ae0b13c8830d1a0031cb0c23410b3b90d70c12f`
- 2WikiMultiHopQA: `914a116c9fbf473b4d806a3525e46b8805f9cf56cb5c7f7f9174dca6ea345a32`
- MuSiQue: `06f341a90ccf54c80cb8c039cee720d380dafbcf1386a2f243f7f549b6bf68e7`

Все три leakage audit завершились без medium/high findings.

## 4. Основные retrieval-результаты

### 4.1. Устойчивость к seed

Среднее и стандартное отклонение gated full evidence по seed 101/202/303:

| Датасет | Hybrid | Gated MRV | Actions |
|---|---:|---:|---:|
| HotpotQA | 0.640 | 0.806 ± 0.003 | 1.652 ± 0.086 |
| 2WikiMultiHopQA | 0.367 | 0.573 ± 0.001 | 1.565 ± 0.079 |
| MuSiQue | 0.156 | 0.255 ± 0.003 | 2.616 ± 0.087 |

Разброс мал. Это означает, что основной retrieval-выигрыш не является случайностью одного seed обучения.

### 4.2. Чувствительность к embeddings

На seed 101 BGE-M3 дал:

| Датасет | BGE hybrid | BGE gated | Разность |
|---|---:|---:|---:|
| HotpotQA | 0.669 | 0.822 | +0.153 |
| 2WikiMultiHopQA | 0.393 | 0.601 | +0.208 |
| MuSiQue | 0.164 | 0.265 | +0.101 |

Направление эффекта не зависит от primary embedding-модели. BGE-M3 запускался только с одним seed, поэтому это representation sensitivity check, а не полноценный второй многосидовый эксперимент.

### 4.3. Paired outcomes

По 3 000 eval-вопросам gated MRV относительно hybrid:

| Датасет | Wins | Losses | Ties |
|---|---:|---:|---:|
| HotpotQA | 199 | 33 | 768 |
| 2WikiMultiHopQA | 328 | 19 | 653 |
| MuSiQue | 164 | 81 | 755 |
| Всего | 691 | 133 | 2 176 |

Метод не просто сдвигает среднее за счёт нескольких выбросов: выигрышей существенно больше, чем потерь. Одновременно 133 losses показывают, что расширение не безвредно и noise control остаётся центральной частью метода.

## 5. Полная reader-оценка

Один и тот же Qwen3-8B reader, prompt, decoding и контекстный бюджет применены к hybrid и gated MRV на всех 1 000 eval-вопросах каждого датасета. Агрегаты получены официальными scorer implementations; интервалы и p-values пересчитаны по официальным per-query определениям, с 5 000 paired bootstrap samples и поправкой Холма внутри каждого датасета.

| Датасет | Hybrid Answer F1 | Gated Answer F1 | Δ Answer F1 [95% CI] | Hybrid Support F1 | Gated Support F1 | Δ Support F1 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.423 | 0.465 | +0.042 [0.020; 0.064] | 0.425 | 0.457 | +0.032 [0.016; 0.048] |
| 2WikiMultiHopQA | 0.232 | 0.307 | +0.075 [0.054; 0.097] | 0.384 | 0.418 | +0.034 [0.018; 0.050] |
| MuSiQue | 0.133 | 0.151 | +0.018 [0.001; 0.036] | 0.304 | 0.305 | +0.002 [-0.013; 0.016] |

После поправки Холма Answer F1 значим на HotpotQA (`p=0.0024`) и 2Wiki (`p<0.0002` при разрешении bootstrap), но не на MuSiQue (`p=0.126`). Support F1 значим на HotpotQA и 2Wiki, но не на MuSiQue. Для HotpotQA Joint F1 вырос с 0.229 до 0.261: Δ=+0.032 [0.017; 0.047], Holm-adjusted `p=0.0016`.

Правильная интерпретация: retrieval gain переносится в downstream answer quality на двух из трёх наборов при фиксированной локальной 8B-модели. MuSiQue задаёт важную границу применимости: дополнительное evidence само по себе почти не меняет support F1 и не даёт скорректированно значимого answer gain. Это не leaderboard-сравнение и не доказательство reader-инвариантности.

## 6. Устойчивость к повреждению графа

Модели обучались на clean graph и без переобучения оценивались при удалении и добавлении рёбер.

Single-seed таблица ниже получена в исходном последовательном frozen run.
Дополнительный five-seed stress-test агрегирует только evidence-quality и
graph-action показатели. Его условия выполнялись параллельно, поэтому
зафиксированное там wall-clock/policy latency не используется для сравнительных
выводов.

| Датасет | Clean | Drop 10% | Drop 25% | Drop 50% | False 10% | False 25% | False 50% | Mixed 25/25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.810 | 0.770 | 0.757 | 0.686 | 0.668 | 0.580 | 0.537 | 0.574 |
| 2WikiMultiHopQA | 0.572 | 0.544 | 0.522 | 0.461 | 0.450 | 0.348 | 0.316 | 0.342 |
| MuSiQue | 0.252 | 0.228 | 0.251 | 0.189 | 0.198 | 0.148 | 0.129 | 0.145 |

Главный результат robustness-анализа: ложные рёбра намного опаснее пропущенных. При false-edge injection качество монотонно падает на всех датасетах. Следовательно, следующий методический приоритет — precision-aware edge validation/denoising, а не простое увеличение полноты графа.

MuSiQue показывает немонотонность при dropout 10/25% на одном фиксированном corruption seed. Это не следует интерпретировать как пользу удаления 25% рёбер: необходимы несколько corruption seeds и доверительные интервалы.

### 6.1. Five-seed robustness

Повтор по заранее зафиксированным corruption seeds 101/202/303/404/505 устраняет
single-seed немонотонность: среднее full-evidence монотонно уменьшается с ростом
дозы и для dropout, и для false-edge injection на всех трёх датасетах.

| Датасет | Clean | Drop 10% | Drop 25% | Drop 50% | False 10% | False 25% | False 50% | Mixed 25/25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.810 | 0.768±0.002 | 0.743±0.009 | 0.685±0.009 | 0.660±0.008 | 0.572±0.009 | 0.536±0.002 | 0.564±0.011 |
| 2WikiMultiHopQA | 0.572 | 0.546±0.001 | 0.523±0.003 | 0.463±0.003 | 0.440±0.010 | 0.345±0.006 | 0.317±0.001 | 0.339±0.006 |
| MuSiQue | 0.252 | 0.236±0.008 | 0.230±0.013 | 0.195±0.007 | 0.196±0.004 | 0.148±0.002 | 0.130±0.002 | 0.147±0.005 |

Значения после clean — mean±sample SD по пяти seed. На 25% повреждения
ложные рёбра хуже dropout на 0.172 для HotpotQA, 0.178 для 2Wiki и 0.082 для
MuSiQue. Это устойчивый, а не single-seed вывод. Параллельный запуск условий
не позволяет интерпретировать их latency; quality-метрики от совместной нагрузки
не зависят.

## 6.1. Опубликованный паттерн: KG²RAG-style baseline

Реализована независимая equal-budget адаптация KG²RAG: те же semantic seeds расширяются через passage/entity graph, кандидаты ранжируются по query relevance, propagated seed score и multi-seed support, а финальный контекст организуется в seed-центричные группы. Это не точное воспроизведение оригинальной системы: исходные triplet KG, relation extraction и FlagReranker не используются.

| Датасет | Hybrid FE | KG²-style FE | Gated FE | KG²−Hybrid | Gated−KG² |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 0.640 | 0.743 | 0.810 | +0.103 | +0.067 |
| 2WikiMultiHopQA | 0.367 | 0.494 | 0.572 | +0.127 | +0.078 |
| MuSiQue | 0.156 | 0.202 | 0.252 | +0.046 | +0.050 |

Для KG²-style минус hybrid 95% CI равны `[0.080; 0.127]`, `[0.105; 0.150]` и `[0.027; 0.065]`. Для KG²-style минус gated MRV интервалы равны `[-0.089; -0.045]`, `[-0.102; -0.053]` и `[-0.067; -0.034]`. Все интервалы исключают ноль.

Интерпретация: общий паттерн semantic seed → graph expansion действительно полезен, но gated MRV стабильно лучше при одинаковых `seed_k`, `final_k`, token budget и action budget. В 3 000 запросах не обнаружено ни одного нарушения бюджета и ни одного расхождения initial seeds с замороженным run. Поэтому основной результат нельзя объяснить только безусловным добавлением графовых соседей.

## 7. Адекватны ли данные и метрики

### Что уже адекватно

Да, текущий набор адекватен для controlled retrieval study:

- используются три признанных multi-hop QA benchmark: [HotpotQA](https://arxiv.org/abs/1809.09600), [2WikiMultiHopQA](https://aclanthology.org/2020.coling-main.580/) и [MuSiQue](https://aclanthology.org/2022.tacl-1.31/);
- методы сравниваются на одних и тех же запросах, initial rankings и бюджетах;
- eval не используется для обучения;
- 1 000 eval-вопросов на датасет дают информативные paired bootstrap intervals;
- есть три seed, второй embedding-backbone, oracle upper bound и controlled corruption;
- full-evidence rate и support recall непосредственно измеряют заявленный retrieval bottleneck.

### Чего эти данные не дают

Текущий pooled-corpus протокол отличается от официальных leaderboard settings. Он не является HotpotQA distractor leaderboard и не является HotpotQA fullwiki. Поэтому:

- нельзя напрямую сравнивать наши проценты с таблицами других статей;
- нельзя заявлять SOTA;
- нельзя утверждать, что метод быстрее полного GraphRAG: KG²RAG-style control не воспроизводит полный GraphRAG lifecycle, а offline construction costs не измерены;
- нельзя делать общий вывод о reader-agnostic answer quality по одной локальной 8B-модели, даже при полном прогоне 3 × 1 000;

### Нужно ли больше данных

Для самой retrieval-гипотезы текущие 3 000 eval-запросов уже содержательны. Для submission-ready статьи желательно добавить:

1. Полный официальный dev либо существенно более крупный фиксированный eval для каждого набора.
2. Официальный HotpotQA distractor или fullwiki run как отдельный внешний протокол.
3. Второй reader-backbone либо внешний API-run для проверки reader-инвариантности.
4. Для прямого claim против конкретной системы — точное воспроизведение её исходного pipeline либо официальный output на том же протоколе.
5. Независимый повтор robustness на другой машине.
6. Независимый повтор запуска другим человеком или на другой машине.

## 8. Нормально ли использовать Qwen3-8B

Да — если правильно ограничить claim.

8B reader подходит для локального resource-bounded исследования, поскольку все методы получают одну и ту же модель, prompt, decoding и контекстный бюджет. Это позволяет сравнивать изменения, вызванные retrieval, внутри нашего протокола. Более того, слабый reader полезен диагностически: он показывает, что полный evidence не автоматически превращается в правильный ответ.

Допустимо утверждать:

- gated graph rescue улучшил retrieval completeness;
- эффект устойчив к seed и embedding-backbone;
- на диагностической 8B-выборке улучшились official Answer/Support F1;
- false edges причиняют больший ущерб, чем edge dropout.

Недопустимо утверждать:

- наш Answer F1 лучше систем с GPT-4/70B из-за retrieval;
- метод является SOTA;
- результат одинаков для любых readers;
- система быстрее обычного GraphRAG без прямого equal-hardware сравнения.

## 9. Что осталось для сильной статьи

Приоритеты в порядке научной ценности:

1. Разложение ошибок на anchor failure, unreachable, selector failure, gate false negative и reader failure.
2. End-to-end latency и offline index cost на одинаковом hardware.
3. Человеческая проверка минимум 100 случайных traces и всех центральных claims.
4. Второй reader-backbone для проверки переноса answer-level эффекта.
5. Независимый повтор ключевого запуска на другой машине.
6. Опционально — exact reproduction конкретной опубликованной системы, если понадобится прямой claim против неё.
7. Только после этого — edge-confidence/denoising как следующий самостоятельный методический вклад.

## 10. Итоговая научная формулировка

Наиболее защищаемый результат звучит так:

> В pooled-corpus multi-hop retrieval selective local graph expansion, управляемый conditional marginal-value selector и calibrated gate, повышает полноту supporting evidence относительно сильного hybrid baseline на трёх датасетах. Независимый KG²RAG-style equal-budget control подтверждает пользу общего graph-expansion pattern, но gated MRV стабильно остаётся выше. Эффект устойчив к seed и embedding representation, но чувствителен прежде всего к ложным рёбрам. Полный прогон локального 8B reader подтверждает downstream-перенос на HotpotQA и 2Wiki, тогда как MuSiQue показывает слабый и скорректированно незначимый answer-level эффект.

Это уже хороший исследовательский результат. Published-pattern control, полный downstream reader evaluation и five-seed robustness закрыты; следующий шаг должен не раздувать статью новыми идеями, а завершить timing/offline-cost audit и ручную проверку traces.
