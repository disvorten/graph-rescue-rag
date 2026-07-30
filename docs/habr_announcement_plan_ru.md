# План технической публикации о Graph Rescue RAG

Цель материала — привести заинтересованных исследователей и инженеров к
воспроизводимому GitHub release, а не заменить научную статью.

## Рабочий заголовок

**Когда обычный RAG не видит второй шаг: локальное восстановление контекста по
графу**

Не использовать заголовки вида «мы сделали RAG лучше GraphRAG» или
«революционный GraphRAG на ноутбуке».

## Структура

### 1. Проблема на одном синтетическом примере

Показать запрос, первый релевантный документ и связанный с ним второй документ,
который не похож на исходный запрос. Не копировать benchmark text.

### 2. Почему одного fusion/reranker недостаточно

Коротко объяснить lexical/semantic distance и evidence chain.

### 3. Основная идея

Схема:

`BM25 + dense + reranker → calibrated gate → local graph candidates → MRV → fixed-budget context`

Подчеркнуть, что граф не заменяет retrieval и не обходится целиком.

### 4. Что именно сравнивалось

- hybrid;
- similarity expansion;
- MRV-always;
- gated MRV;
- equal-budget KG²RAG-style control;
- oracle;
- corruption stress test;
- одинаковый 8B reader.

### 5. Три результата

После финальной проверки вставить:

1. full-evidence gain на трёх датасетах;
2. разницу gated MRV и KG²RAG-style при равном бюджете;
3. reader delta на полном 3 × 1 000 прогоне.

Для каждого числа дать ссылку на CSV и tag релиза.

### 6. Где метод ломается

- anchor отсутствует;
- нужный passage недостижим;
- selector выбирает hub;
- gate даёт false negative;
- false edges вводят ложную цепочку;
- reader игнорирует корректное evidence.

Это самая ценная часть для доверия и обсуждения.

### 7. Воспроизведение

Дать только:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python -m graph_rescue run --config examples/demo_config.json
```

Полный протокол — по ссылке на `REPRODUCING.md`.

### 8. Призыв к проверке

Попросить читателей не просто поставить star, а:

- выполнить smoke test;
- открыть reproduction issue;
- предложить публичный comparable baseline;
- проверить один агрегат;
- указать на protocol mismatch.

## Материалы запуска

- GitHub release и tag;
- Zenodo DOI;
- social preview;
- одна архитектурная схема;
- одна таблица результатов;
- один failure-analysis рисунок;
- честный AI-use disclosure;
- ссылка на preprint после разрешения выбранного журнала.

## Короткий анонс

> Hybrid retrieval хорошо находит стартовые свидетельства, но может пропустить
> второй шаг цепочки, если он слабо похож на запрос. В Graph Rescue RAG граф
> используется как локальный repair: calibrated gate решает, нужен ли обход, а
> conditional MRV выбирает кандидата относительно уже найденного evidence.
> Репозиторий содержит fixed-budget comparisons, paired confidence intervals,
> corruption tests и локальный 8B reader. Нужны независимые воспроизведения и
> предложения comparable baselines: [release URL].

Перед публикацией заменить placeholder, проверить все числа и переписать анонс
в собственном стиле.
