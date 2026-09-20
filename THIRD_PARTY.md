# Third-party datasets, code and prior work

## Prior work this repo builds on

**[`AHTOOOXA/jev-cyrillic-audit`](https://github.com/AHTOOOXA/jev-cyrillic-audit)** (MIT) — the
Russian/English audit of Jev. This repo deliberately reuses its method so the two are
comparable: the same three-arm structure, the same decision-rule thresholds (3 pp for accuracy,
0.02 for ECE), the same inherited sanity checks on `confidence` and on `choice` versus argmax.
That audit also left the "instructions in the state's language" cells frozen but never run;
answering that question for Spanish is what this repo adds.

**[`AbdelStark/jev-benchmarks`](https://github.com/AbdelStark/jev-benchmarks)** (Apache-2.0) —
benchmark conventions.

## Datasets

None of the source text is redistributed here. `data/items.parquet` contains only item ids, gold
labels and state hashes; the text is re-downloaded from Hugging Face and re-joined at run time.
Users of this repo are bound by each dataset's own licence.

| Dataset | Hugging Face id | Licence | Citation |
|---|---|---|---|
| XNLI | `facebook/xnli` | No licence tag on the HF dataset page; see the paper and the MultiNLI terms | Conneau et al., *XNLI: Evaluating Cross-lingual Sentence Representations*, EMNLP 2018 |
| PAWS-X | `google-research-datasets/paws-x` | "Free for use for any purpose" (Google, per the dataset card) | Yang et al., *PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification*, EMNLP 2019 |
| MASSIVE | `mteb/amazon_massive_intent` | CC BY 4.0 | FitzGerald et al., *MASSIVE: A 1M-Example Multilingual NLU Dataset*, 2022 |
| Belebele | `facebook/belebele` | CC BY-SA 4.0 | Bandarkar et al., *The Belebele Benchmark*, 2023 |

**XNLI's licence is genuinely unclear** — the Hugging Face dataset page carries no licence tag.
This repo commits no XNLI text, only hashes, which is why that ambiguity does not propagate to
anyone cloning it. Anyone redistributing XNLI text should resolve the licence themselves.

## Model and API

**Jev**, by **TypeSafe AI**, accessed through the **Vercel AI Gateway** passthrough. Use is
subject to TypeSafe's [Master Customer Agreement](https://typesafe.ai/legal/mca). No model
output in this repo is used to train or distil any model, which that agreement prohibits.

## Python dependencies

`httpx` (BSD-3), `datasets` (Apache-2.0), `pyarrow` (Apache-2.0), `numpy` (BSD-3),
`matplotlib` (PSF-based), and for development `pytest` (MIT), `netcal` (Apache-2.0) and
`ruff` (MIT).

`netcal` is a development dependency only. It is never imported at run time; the test suite uses
it to cross-check this repo's own ECE implementation against an independent one.
