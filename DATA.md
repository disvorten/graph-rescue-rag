# Data sources and local preparation

Benchmark files are **not distributed in this repository**. Download them from
their original maintainers, review the applicable terms, and place them in the
paths below before running the preparation scripts.

## HotpotQA

- Official project: <https://hotpotqa.github.io/>
- Official code/data repository: <https://github.com/hotpotqa/hotpot>
- Dataset and processed Wikipedia text: CC BY-SA 4.0 according to the official
  project.
- Expected local files:
  - `work/datasets/hotpot_train_v1.1.json`
  - `work/datasets/hotpot_dev_distractor_v1.json`

Files used for the frozen local snapshot:

| File | Bytes | SHA-256 |
|---|---:|---|
| `hotpot_train_v1.1.json` | 566,426,227 | `26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316` |
| `hotpot_dev_distractor_v1.json` | 61,065,698 | `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8` |

## 2WikiMultiHopQA

- Official repository: <https://github.com/Alab-NII/2wikimultihop>
- The official repository declares Apache-2.0. Dataset users remain responsible
  for checking whether incorporated source text has additional attribution
  obligations.
- Expected local files:
  - `work/datasets/2wiki_official/train.json`
  - `work/datasets/2wiki_official/dev.json`

Files used for the frozen local snapshot:

| File | Bytes | SHA-256 |
|---|---:|---|
| `train.json` | 707,810,660 | `b318dbafbfed51a8029718fa59be8b616600cbff675a3b587694b28c5eedfc13` |
| `dev.json` | 57,614,142 | `79f77ae104088ea8e25b1a65dbece768d45771194663bc5660ec9a98070dadf5` |

## MuSiQue

- Official repository: <https://github.com/StonyBrookNLP/musique>
- Paper: <https://aclanthology.org/2022.tacl-1.31/>
- Dataset license: CC BY 4.0, as stated in the official repository:
  <https://github.com/StonyBrookNLP/musique/blob/main/LICENSE>
- Expected local files:
  - `work/datasets/musique_official/data/musique_ans_v1.0_train.jsonl`
  - `work/datasets/musique_official/data/musique_ans_v1.0_dev.jsonl`

Files used for the frozen local snapshot:

| File | Bytes | SHA-256 |
|---|---:|---|
| `musique_ans_v1.0_train.jsonl` | 241,046,755 | `83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a` |
| `musique_ans_v1.0_dev.jsonl` | 30,439,728 | `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b` |

The official MuSiQue repository states that the dataset is distributed under
CC BY 4.0. The public artifact still omits MuSiQue records and derived passage
text: users should download from the authoritative source, preserve
attribution, and verify that any incorporated seed-dataset material is used
consistently with its own terms. The repository contains only preparation code,
hashes, aggregate metrics, and protocol metadata.

## Derived artifacts

Run:

```powershell
python work/prepare_final_protocol.py --dataset all
```

This creates dataset-specific corpora, train/evaluation JSONL files, graph
artifacts, manifests, and audits under `work/final_protocol/`. Those generated
files can contain benchmark text and are ignored by Git.

The project never transfers the project code license to benchmark data,
official evaluator code, model weights, or Ollama outputs. Keep upstream
notices with any local copies.

## Global development/distractor protocol

`work/prepare_global_corpus_protocol.py` derives a second, leakage-controlled
evaluation from the same authoritative files. It removes every official-dev
question ID used by earlier pilots or the primary 1,000-query evaluation, then
uses all remaining official-dev questions as an external test set. The
searchable corpus contains every unique passage in the complete official
development/distractor split plus the frozen 1,000-query training sample.

This setting is larger and more realistic than the per-question pooled
protocol, but it is not full-wiki retrieval. Graph construction reads passage
titles and text only; answers, supporting-fact labels, decompositions, and
evidence triples remain evaluation-only fields. Generated corpora, queries,
graphs, and embedding caches remain ignored because they contain benchmark
text.

## Released HippoRAG benchmark artifact

The exact-corpus comparison uses the corpus, questions, and precomputed OpenIE
artifact released by the official HippoRAG repository:
<https://github.com/OSU-NLP-Group/HippoRAG>. The local adapter records SHA-256
hashes and converts identifiers without modifying text or relevance labels.
The third-party repository, its generated artifacts, and model outputs are not
redistributed by this project. Its MIT license applies to upstream code; this
repository's Apache-2.0 license does not supersede that license or the dataset
terms.
