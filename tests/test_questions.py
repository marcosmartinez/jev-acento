"""The arm invariants. These are what make B - A and C - B mean what they claim to mean."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_acento.questions import (
    ARMS,
    PromptValidationError,
    load_arm_prompt,
    load_prompt,
    validate_prompt_set,
)

OPTIONS = ("yes", "no")


def _edit(prompts_dir: Path, lang: str, **changes) -> None:
    path = prompts_dir / f"toy.{lang}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc.update(changes)
    path.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")


def test_valid_prompt_set_passes(prompts_dir: Path):
    validate_prompt_set("toy", OPTIONS, prompts_dir)


def test_arms_a_and_b_share_the_english_wording(prompts_dir: Path):
    """B - A must isolate the state language, so both arms must use the identical prompt."""
    a = load_arm_prompt("toy", "A", prompts_dir)
    b = load_arm_prompt("toy", "B", prompts_dir)
    assert a == b
    assert ARMS["A"][0] == "en" and ARMS["B"][0] == "es"  # ...but different state languages


def test_arm_c_differs_only_in_prompt_language(prompts_dir: Path):
    b = load_arm_prompt("toy", "B", prompts_dir)
    c = load_arm_prompt("toy", "C", prompts_dir)
    assert ARMS["B"][0] == ARMS["C"][0] == "es"   # same state language
    assert b.instructions != c.instructions       # different instruction language
    assert tuple(b.criteria) == tuple(c.criteria)  # identical English keys


def test_translated_option_keys_are_rejected(prompts_dir: Path):
    """Translating the keys would change the label space as well as the language."""
    _edit(prompts_dir, "es", criteria={"sí": "trata sobre el clima", "no": "no trata"})
    with pytest.raises(PromptValidationError, match="criteria keys"):
        validate_prompt_set("toy", OPTIONS, prompts_dir)


def test_reordered_option_keys_are_rejected(prompts_dir: Path):
    doc = json.loads((prompts_dir / "toy.es.json").read_text(encoding="utf-8"))
    doc["criteria"] = {k: doc["criteria"][k] for k in reversed(list(doc["criteria"]))}
    (prompts_dir / "toy.es.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(PromptValidationError, match="order or content"):
        validate_prompt_set("toy", OPTIONS, prompts_dir)


def test_missing_option_is_rejected(prompts_dir: Path):
    _edit(prompts_dir, "es", criteria={"yes": "sí"})
    with pytest.raises(PromptValidationError, match="missing"):
        validate_prompt_set("toy", OPTIONS, prompts_dir)


def test_extra_option_is_rejected(prompts_dir: Path):
    _edit(prompts_dir, "en", criteria={"yes": "y", "no": "n", "maybe": "m"})
    with pytest.raises(PromptValidationError, match="extra"):
        validate_prompt_set("toy", OPTIONS, prompts_dir)


def test_untranslated_placeholder_is_rejected(prompts_dir: Path):
    """An 'es' file left as a copy of the English one would collapse arm C into arm B."""
    english = load_prompt("toy", "en", prompts_dir)
    _edit(prompts_dir, "es", instructions=english.instructions)
    with pytest.raises(PromptValidationError, match="indistinguishable from arm B"):
        validate_prompt_set("toy", OPTIONS, prompts_dir)


def test_mismatched_question_id_is_rejected(prompts_dir: Path):
    _edit(prompts_dir, "es", question_id="tema")
    with pytest.raises(PromptValidationError, match="question id"):
        validate_prompt_set("toy", OPTIONS, prompts_dir)


def test_empty_instructions_are_rejected(prompts_dir: Path):
    _edit(prompts_dir, "es", instructions="   ")
    with pytest.raises(PromptValidationError, match="empty instructions"):
        validate_prompt_set("toy", OPTIONS, prompts_dir)


def test_payload_shape_matches_the_api(prompts_dir: Path):
    payload = load_prompt("toy", "en", prompts_dir).payload()
    assert list(payload) == ["topic"]
    assert payload["topic"]["type"] == "choice"
    assert set(payload["topic"]["criteria"]) == {"yes", "no"}
