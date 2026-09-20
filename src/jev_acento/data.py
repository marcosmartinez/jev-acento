"""Dataset loading, cross-language alignment and stratified sampling.

Four public, parallel, human-labelled datasets back the audit. Each one needs three things
before it can be used:

1. **A join key across languages.** Only three of the four ship one. XNLI has no id at all and
   must be joined positionally, which is exactly where the Russian audit found two crossed rows.
2. **An alignment check.** Two rows that claim to be the same item must carry the same gold
   label. Rows that fail are excluded *before* sampling and the count is reported, never
   silently dropped.
3. **A state builder.** The `state` sent to Jev is a JSON object. Per the brief, its *field
   names stay in English in every arm* -- only `instructions` and the `criteria` descriptions are
   translated in arm C. That is what isolates instruction language from everything else.

**No source text is ever committed.** :func:`freeze_items` writes only item ids, gold labels and
state hashes to ``data/items.parquet``; the text is re-derived from the upstream dataset at run
time and verified against those hashes.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .providers import canonical_json

# Keep the HF cache inside the repo but out of git (see .gitignore).
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"

State = dict[str, Any]


@dataclass(frozen=True)
class Item:
    """One logical item, materialised in every language we need."""

    item_id: str
    gold: str
    states: dict[str, State]  # lang -> state object

    def state_hash(self, lang: str) -> str:
        return sha256_state(self.states[lang])


def sha256_state(state: State) -> str:
    """Content hash of a state object, stable across key orderings."""
    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to turn an upstream dataset into aligned :class:`Item` objects."""

    name: str
    hf_id: str
    split: str
    configs: dict[str, str]
    """Maps our language code to the upstream config name (e.g. ``{"es": "spa_Latn"}``)."""
    primitive: str  # "choice" | "noul"
    key_fn: Callable[[dict[str, Any], int], str]
    """Stable item id from a row and its position. Positional for datasets with no id."""
    gold_fn: Callable[[dict[str, Any]], str]
    state_fn: Callable[[dict[str, Any]], State]
    options: tuple[str, ...]
    """The full label space, in English, identical across all arms."""
    n_target: int
    """Pre-registered sample size. Capped by the dataset when it is smaller."""


# --------------------------------------------------------------------------------------------
# Per-dataset adapters
# --------------------------------------------------------------------------------------------

XNLI_LABELS = ("entailment", "neutral", "contradiction")


def _xnli_gold(row: dict[str, Any]) -> str:
    return XNLI_LABELS[int(row["label"])]


def _pawsx_gold(row: dict[str, Any]) -> str:
    # PAWS-X ClassLabel names are ("0", "1"); 1 means the pair *is* a paraphrase.
    return "true" if int(row["label"]) == 1 else "false"


def _belebele_gold(row: dict[str, Any]) -> str:
    # `correct_answer_num` arrives as the string "1".."4".
    return f"option_{int(row['correct_answer_num'])}"


SPECS: dict[str, DatasetSpec] = {
    "xnli": DatasetSpec(
        name="xnli",
        hf_id="facebook/xnli",
        split="test",
        configs={"en": "en", "es": "es"},
        primitive="choice",
        # XNLI ships no id column. The language configs are row-aligned by construction, so the
        # position *is* the key -- and the alignment check below is what makes that safe.
        key_fn=lambda row, i: f"xnli-{i:05d}",
        gold_fn=_xnli_gold,
        state_fn=lambda row: {"premise": row["premise"], "hypothesis": row["hypothesis"]},
        options=XNLI_LABELS,
        n_target=1000,
    ),
    "pawsx": DatasetSpec(
        name="pawsx",
        hf_id="google-research-datasets/paws-x",
        split="test",
        configs={"en": "en", "es": "es"},
        primitive="noul",
        key_fn=lambda row, i: f"pawsx-{int(row['id']):05d}",
        gold_fn=_pawsx_gold,
        state_fn=lambda row: {"sentence_1": row["sentence1"], "sentence_2": row["sentence2"]},
        options=("true", "false"),
        n_target=1000,
    ),
    "massive": DatasetSpec(
        name="massive",
        hf_id="mteb/amazon_massive_intent",
        split="test",
        configs={"en": "en", "es": "es"},
        primitive="choice",
        key_fn=lambda row, i: f"massive-{str(row['id']).zfill(5)}",
        gold_fn=lambda row: str(row["label"]),
        state_fn=lambda row: {"utterance": row["text"]},
        options=(),  # filled at load time from the data (59 intents, verified 2026-09-20)
        n_target=600,
    ),
    "belebele": DatasetSpec(
        name="belebele",
        hf_id="facebook/belebele",
        split="test",
        configs={"en": "eng_Latn", "es": "spa_Latn"},
        primitive="choice",
        # The last path segment of `link` is NOT unique -- two different wiki pages collide on
        # it, which silently merged 8 of the 900 rows. The full URL is unique and identical
        # across languages, so it is hashed for a compact, stable id.
        key_fn=lambda row, i: (
            f"belebele-{hashlib.sha1(row['link'].encode()).hexdigest()[:10]}"
            f"-{row['question_number']}"
        ),
        gold_fn=_belebele_gold,
        # Passage, question and the four options all live inside the state; `criteria` stays
        # neutral (`option_1`..`option_4`) so no answer text leaks into the label space.
        state_fn=lambda row: {
            "passage": row["flores_passage"],
            "question": row["question"],
            "option_1": row["mc_answer1"],
            "option_2": row["mc_answer2"],
            "option_3": row["mc_answer3"],
            "option_4": row["mc_answer4"],
        },
        options=("option_1", "option_2", "option_3", "option_4"),
        # Belebele ships 900 rows per language, so 600 is already 67% of the set. n cannot be
        # raised here the way it was for XNLI and PAWS-X.
        n_target=600,
    ),
}


# --------------------------------------------------------------------------------------------
# Loading and alignment
# --------------------------------------------------------------------------------------------


def _load_split(spec: DatasetSpec, lang: str, cache_dir: Path) -> list[dict[str, Any]]:
    from datasets import load_dataset

    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))
    ds = load_dataset(spec.hf_id, spec.configs[lang], split=spec.split)
    return [dict(r) for r in ds]


@dataclass
class AlignmentReport:
    """Audit trail for the cross-language join. Goes into the run manifest verbatim."""

    dataset: str
    n_rows: dict[str, int]
    n_joined: int
    n_gold_mismatch: int
    n_missing: int
    mismatched_ids: list[str]

    def summary(self) -> str:
        return (
            f"{self.dataset}: joined {self.n_joined} items "
            f"({self.n_gold_mismatch} excluded for gold mismatch, "
            f"{self.n_missing} missing in one language)"
        )


def load_aligned(
    spec: DatasetSpec, cache_dir: Path = DEFAULT_CACHE
) -> tuple[list[Item], AlignmentReport]:
    """Load every language, join on the spec's key, and drop anything that does not line up.

    The gold label is the alignment probe: two rows that are genuinely the same item, translated,
    must carry the same human label. A disagreement means the join is wrong -- crossed rows, a
    reindexed split, an upstream fix applied to one language only -- and the item is excluded
    rather than guessed at.
    """
    langs = list(spec.configs)
    per_lang: dict[str, dict[str, dict[str, Any]]] = {}
    n_rows: dict[str, int] = {}

    for lang in langs:
        rows = _load_split(spec, lang, cache_dir)
        n_rows[lang] = len(rows)
        keyed: dict[str, dict[str, Any]] = {}
        for i, row in enumerate(rows):
            keyed[spec.key_fn(row, i)] = row
        per_lang[lang] = keyed

    common = set.intersection(*(set(k) for k in per_lang.values()))
    all_keys = set.union(*(set(k) for k in per_lang.values()))

    items: list[Item] = []
    mismatched: list[str] = []

    for key in sorted(common):
        golds = {lang: spec.gold_fn(per_lang[lang][key]) for lang in langs}
        if len(set(golds.values())) != 1:
            mismatched.append(key)
            continue
        items.append(
            Item(
                item_id=key,
                gold=next(iter(golds.values())),
                states={lang: spec.state_fn(per_lang[lang][key]) for lang in langs},
            )
        )

    report = AlignmentReport(
        dataset=spec.name,
        n_rows=n_rows,
        n_joined=len(items),
        n_gold_mismatch=len(mismatched),
        n_missing=len(all_keys - common),
        mismatched_ids=sorted(mismatched)[:50],
    )
    return items, report


def resolve_options(spec: DatasetSpec, items: list[Item]) -> tuple[str, ...]:
    """The label space for a dataset, read from the data when the spec leaves it open.

    MASSIVE's ~60 intents are data-defined, so they are collected from the gold column and
    sorted. Sorting matters: the option order is part of the frozen prompt.
    """
    if spec.options:
        return spec.options
    return tuple(sorted({item.gold for item in items}))


# --------------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------------


def stratified_sample(items: list[Item], n: int, seed: int) -> list[Item]:
    """Sample ``n`` items, preserving the gold-label distribution.

    Allocation uses largest-remainder so the strata sum to exactly ``n`` without any single
    stratum being rounded away -- which matters for MASSIVE, where the rarest intents would
    otherwise vanish and take macro-F1 with them. Every stratum with at least one member gets at
    least one slot.

    When ``n`` is smaller than the number of strata -- only possible in a capped smoke run,
    since MASSIVE's 59 intents are far below the pre-registered n of 600 -- the distribution
    cannot be preserved, and the function falls back to one item from each of the ``n`` largest
    strata.

    Deterministic given ``seed``: the same seed always yields the same item ids.
    """
    if n >= len(items):
        return sorted(items, key=lambda it: it.item_id)

    rng = np.random.default_rng(seed)
    by_gold: dict[str, list[Item]] = {}
    for item in sorted(items, key=lambda it: it.item_id):
        by_gold.setdefault(item.gold, []).append(item)

    strata = sorted(by_gold)
    sizes = np.array([len(by_gold[g]) for g in strata], dtype=float)
    exact = sizes / sizes.sum() * n

    if n < len(strata):
        # Fewer slots than strata, so the distribution cannot be preserved at all. This only
        # happens for capped smoke runs (MASSIVE has 59 intents); a real run always has
        # n >> strata. Take the n largest strata, one item each, and say so.
        alloc = np.zeros(len(strata), dtype=int)
        alloc[np.argsort(-sizes, kind="stable")[:n]] = 1
    else:
        # Floor, guaranteeing one slot per non-empty stratum, then hand out the remainder to
        # the largest fractional parts.
        alloc = np.maximum(np.floor(exact), 1).astype(int)
        alloc = np.minimum(alloc, sizes.astype(int))

        while alloc.sum() > n:  # too many after the per-stratum floor of 1
            shrinkable = np.where(alloc > 1)[0]
            victim = shrinkable[np.argmax(alloc[shrinkable])]
            alloc[victim] -= 1
        while alloc.sum() < n:
            headroom = sizes.astype(int) - alloc
            candidates = np.where(headroom > 0)[0]
            winner = candidates[np.argmax((exact - alloc)[candidates])]
            alloc[winner] += 1

    chosen: list[Item] = []
    for g, k in zip(strata, alloc, strict=True):
        pool = by_gold[g]
        idx = rng.permutation(len(pool))[:k]
        chosen.extend(pool[i] for i in sorted(idx))

    return sorted(chosen, key=lambda it: it.item_id)


# --------------------------------------------------------------------------------------------
# Freezing
# --------------------------------------------------------------------------------------------

ITEMS_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("item_id", pa.string()),
        ("gold", pa.string()),
        ("state_hash_en", pa.string()),
        ("state_hash_es", pa.string()),
    ]
)


def freeze_items(rows: list[dict[str, str]], path: Path) -> None:
    """Write the sampled items to parquet.

    Ids, gold labels and hashes only. This file is safe to commit and to publish; the dataset
    text is not ours to redistribute.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=ITEMS_SCHEMA)
    pq.write_table(table, path)


def items_to_rows(dataset: str, items: list[Item]) -> Iterator[dict[str, str]]:
    for item in items:
        yield {
            "dataset": dataset,
            "item_id": item.item_id,
            "gold": item.gold,
            "state_hash_en": item.state_hash("en"),
            "state_hash_es": item.state_hash("es"),
        }


def read_items(path: Path) -> list[dict[str, Any]]:
    """Read back the frozen item table."""
    return pq.read_table(path).to_pylist()
