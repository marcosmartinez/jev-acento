"""Alignment and sampling. Both are places where a silent error would poison every result."""

from __future__ import annotations

from collections import Counter

import pytest

from jev_acento.data import SPECS, Item, sha256_state, stratified_sample


def _items(n: int, golds: list[str]) -> list[Item]:
    return [
        Item(item_id=f"i-{i:04d}", gold=golds[i % len(golds)],
             states={"en": {"t": f"en {i}"}, "es": {"t": f"es {i}"}})
        for i in range(n)
    ]


def test_state_hash_is_stable_across_key_order():
    """The Gateway returns keys in varying order; hashes must not depend on it."""
    assert sha256_state({"a": 1, "b": 2}) == sha256_state({"b": 2, "a": 1})


def test_state_hash_changes_with_content():
    assert sha256_state({"t": "hola"}) != sha256_state({"t": "hello"})


def test_sample_is_deterministic_given_a_seed():
    items = _items(500, ["x", "y", "z"])
    a = [i.item_id for i in stratified_sample(items, 60, seed=7)]
    b = [i.item_id for i in stratified_sample(items, 60, seed=7)]
    assert a == b


def test_a_different_seed_gives_a_different_sample():
    items = _items(500, ["x", "y", "z"])
    a = [i.item_id for i in stratified_sample(items, 60, seed=7)]
    b = [i.item_id for i in stratified_sample(items, 60, seed=8)]
    assert a != b


def test_sample_hits_the_requested_size_exactly():
    for n in (1, 7, 60, 199, 500):
        assert len(stratified_sample(_items(500, ["x", "y", "z"]), n, seed=1)) == n


def test_sample_preserves_the_gold_distribution():
    # 400 x, 100 y  ->  a 100-item sample should hold roughly 80 x and 20 y.
    items = ([Item(f"x-{i}", "x", {"en": {}, "es": {}}) for i in range(400)]
             + [Item(f"y-{i}", "y", {"en": {}, "es": {}}) for i in range(100)])
    counts = Counter(i.gold for i in stratified_sample(items, 100, seed=3))
    assert counts["x"] == pytest.approx(80, abs=2)
    assert counts["y"] == pytest.approx(20, abs=2)


def test_every_stratum_keeps_at_least_one_item():
    """MASSIVE has intents with a single test example; they must not be rounded away."""
    items = ([Item(f"big-{i}", "big", {"en": {}, "es": {}}) for i in range(990)]
             + [Item("rare-0", "rare", {"en": {}, "es": {}})])
    counts = Counter(i.gold for i in stratified_sample(items, 100, seed=3))
    assert counts["rare"] == 1


def test_fewer_slots_than_strata_falls_back_cleanly():
    """Only reachable in a capped smoke run, but it must not crash."""
    items = _items(300, [f"label-{i}" for i in range(59)])
    sample = stratified_sample(items, 40, seed=1)
    assert len(sample) == 40
    assert len({i.gold for i in sample}) == 40, "one item from each of 40 distinct strata"


def test_requesting_more_than_available_returns_everything():
    items = _items(10, ["x", "y"])
    assert len(stratified_sample(items, 999, seed=1)) == 10


def test_sample_is_sorted_by_item_id():
    sample = stratified_sample(_items(200, ["x", "y"]), 50, seed=2)
    assert [i.item_id for i in sample] == sorted(i.item_id for i in sample)


def test_registered_sample_sizes_match_the_prereg():
    assert SPECS["xnli"].n_target == 1000
    assert SPECS["pawsx"].n_target == 1000
    assert SPECS["massive"].n_target == 600
    assert SPECS["belebele"].n_target == 600, "capped by the dataset's 900 rows per language"


def test_belebele_key_uses_the_full_link():
    """The last path segment of `link` collides across wiki pages and merged 8 of 900 rows."""
    spec = SPECS["belebele"]
    a = spec.key_fn({"link": "https://x.org/wiki/Foo/Bar", "question_number": 1}, 0)
    b = spec.key_fn({"link": "https://y.org/wiki/Baz/Bar", "question_number": 1}, 1)
    assert a != b
