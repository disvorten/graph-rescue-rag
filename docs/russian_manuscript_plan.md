# Русскоязычный план статьи Graph Rescue RAG

Целевая площадка первого выбора: **Journal of Intelligent Information Systems
(JIIS)**. Русская версия нужна как содержательный рабочий текст; подача в JIIS
выполняется на английском. При отказе резервная площадка — журнал, выбранный по
актуальному квартилю и тематике; решение не следует фиксировать до повторной
проверки индексации непосредственно перед переподачей.

План составлен под опубликованные требования JIIS на 4 августа 2026 года:

- общий объём — не более 25 страниц, включая ссылки, таблицы и рисунки;
- аннотация — 150–250 слов;
- 4–6 ключевых слов;
- LaTeX-класс Springer `svjour3`, опция `smallcondensed`;
- нумерованный стиль ссылок;
- исходники LaTeX и собранный PDF при подаче;
- обязательные declarations, data/code availability и author contributions.

Это не готовый текст для подачи. Это каркас, который автор должен заполнить
собственными формулировками после проверки экспериментов и traces.

## Рабочее название

**Селективное графовое дополнение результатов гибридного поиска в задачах
многошагового ответа на вопросы**

Альтернатива с более системным акцентом:

**Метод условного восстановления недостающих свидетельств в гибридной
RAG-системе**

## Центральный исследовательский вопрос

Может ли локальное расширение графа, выполняемое только после сильного
BM25+dense+reranker поиска и управляемое оценкой условной предельной ценности,
повысить полноту цепочки свидетельств при фиксированном бюджете лучше, чем:

1. один гибридный поиск;
2. безусловное graph expansion;
3. similarity-only expansion;
4. равнобюджетная KG²RAG-style стратегия?

## Допустимый основной вывод

В замороженном pooled-corpus протоколе на HotpotQA, 2WikiMultiHopQA и MuSiQue
селективное локальное graph rescue повышает full-evidence rate относительно
сильного hybrid baseline. Улучшение устойчиво к seed и двум embedding
backbones, переносится на локальный 8B reader, но чувствительно к ложным рёбрам.

Не использовать в названии, аннотации и выводах слова SOTA, «лучше GraphRAG»,
«универсальный» или «доказано для реальных систем».

## Структура и бюджет объёма

### 1. Введение — 450–550 слов

Автор своими словами отвечает на четыре вопроса:

- почему hybrid retrieval пропускает связанные, но лексически и семантически
  удалённые элементы;
- почему полный GraphRAG может быть избыточным;
- почему graph expansion следует рассматривать как локальный repair;
- какой проверяемый вклад даёт работа.

В конце — три вклада без рекламных формулировок.

### 2. Связанные работы — 500–650 слов

Четыре компактные группы:

1. sparse+dense fusion и reranking;
2. GraphRAG и KG-guided RAG;
3. adaptive/cost-aware graph retrieval;
4. multi-hop retrieval evaluation.

Обязательно провести границу с KG²RAG, A2RAG, CatRAG, HHS-RAG и PruneRAG.
Новизна не в самом факте адаптивного graph retrieval, а в комбинации:

- frozen hybrid retrieval before graph access;
- graph as local evidence repair;
- marginal rescue value conditioned on current evidence;
- separate calibrated gate;
- equal-budget paired evaluation;
- false-edge versus missing-edge stress test.

### 3. Метод — 850–950 слов

Подразделы:

1. постановка и обозначения;
2. BM25+dense fusion и reranker;
3. построение графа;
4. генерация локальных кандидатов;
5. conditional marginal rescue value;
6. calibrated gate;
7. ограничения бюджета и псевдокод.

Все формулы сверить с кодом. Для каждого feature указать, доступен ли он во
время inference и не использует ли gold evidence.

### 4. Экспериментальный протокол — 650–750 слов

Зафиксировать:

- три датасета;
- disjoint train/calibration/eval;
- 1 000 train + 1 000 eval вопросов на датасет;
- размеры pooled corpora и графов;
- Qwen и BGE embeddings;
- три training seeds;
- одинаковые initial rankings и budgets;
- retrieval, reader, robustness и latency metrics;
- paired bootstrap CI и correction for multiple comparisons;
- local Qwen3-8B reader и одинаковую decoding policy.

Отдельно объяснить, почему протокол подходит для controlled retrieval study, но
не является официальным leaderboard setting.

### 5. Результаты — 800–900 слов

Оставить не более четырёх основных таблиц:

1. hybrid против gated MRV на трёх датасетах;
2. сравнение политик и equal-budget KG²RAG-style control;
3. полный reader на 3 × 1 000 вопросах;
4. corruption mean±SD по пяти seeds.

В тексте интерпретировать эффекты, а не повторять все числа из таблиц.

### 6. Анализ ошибок и ограничений — 450–550 слов

Включить минимум:

- 3–5 вручную проверенных wins/losses/gate false negatives;
- различие anchor failure и selector failure;
- вред ложных рёбер;
- pooled-corpus limitation;
- отсутствие exact reproduction новых закрытых/неполных baseline;
- отсутствие полного равнобюджетного GraphRAG lifecycle benchmark;
- ограничение 8B reader.

Этот раздел должен опираться на размеченную стратифицированную выборку traces.

### 7. Заключение — 180–230 слов

Три элемента:

- что именно подтверждено;
- что не подтверждено;
- следующий эксперимент: precision-aware edge confidence и одинаковый
  end-to-end cost benchmark.

## Что вынести из основного текста

В репозиторий или приложение:

- все конфигурации;
- hashes и exact commands;
- таблицы по каждому seed;
- дополнительные slices;
- полный corruption grid;
- инструкция загрузки датасетов;
- reproducibility manifest; внутренние материалы авторской проверки остаются
  локальными.

## До подачи

- проверить, допускает ли журнал публичный preprint до подачи;
- проверить агрегаты завершённого five-seed corruption; полный reader 3 × 1 000
  также завершён;
- завершить размеченный анализ стратифицированной выборки retrieval traces;
- выполнить внешний smoke test репозитория;
- проверить все ссылки по первичным источникам;
- согласовать author list, affiliation, ORCID и conflicts;
- подготовить короткое письмо редактору с точным описанием setting.
