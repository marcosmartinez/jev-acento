"""Shared fixtures: a miniature repo with its own prompts, items and runs directory.

Every test that touches the runner works against this, never against the real repo, so a test
can never write into `runs/` or disturb a freeze.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_acento.data import Item


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """A two-language prompt pair for a fake 'toy' dataset with a 2-option label space."""
    d = tmp_path / "prompts"
    d.mkdir()
    for lang, instr, crit in (
        ("en", "Decide whether `text` is about weather.",
         {"yes": "The text is about weather.", "no": "The text is not about weather."}),
        ("es", "Determinar si `text` trata sobre el clima.",
         {"yes": "El texto trata sobre el clima.", "no": "El texto no trata sobre el clima."}),
    ):
        (d / f"toy.{lang}.json").write_text(
            json.dumps({
                "dataset": "toy", "lang": lang, "question_id": "topic",
                "type": "choice", "instructions": instr, "criteria": crit,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return d


@pytest.fixture
def toy_items() -> list[Item]:
    """Twelve aligned items, balanced across the two gold labels."""
    items = []
    for i in range(12):
        gold = "yes" if i % 2 == 0 else "no"
        items.append(Item(
            item_id=f"toy-{i:03d}",
            gold=gold,
            states={
                "en": {"text": f"english text number {i} about {gold}"},
                "es": {"text": f"texto en español número {i} sobre {gold}"},
            },
        ))
    return items
