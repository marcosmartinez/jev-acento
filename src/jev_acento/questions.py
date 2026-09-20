"""Prompt loading, arm construction, and the invariants that make the arms comparable.

An *arm* is a (state language, prompt language) pair:

===  ==============  =====================  ===================================================
Arm  ``state``       ``instructions``       Measures
===  ==============  =====================  ===================================================
A    English         English                Baseline
B    Spanish         English                Cost of the language      (B - A)
C    Spanish         Spanish                Value of native instructions (C - B)
===  ==============  =====================  ===================================================

The comparison is only meaningful if *exactly one thing* changes between adjacent arms. Two
invariants enforce that, and :func:`validate_prompt_set` refuses to proceed when either breaks:

1. **Option keys stay in English and identical across every arm.** Arm C translates the
   `instructions` string and the `criteria` *descriptions* -- never the keys. Translating keys
   would change the label space as well as the instruction language, confounding C - B.
2. **The criteria key set matches the dataset's label space exactly.** A missing intent in
   MASSIVE would quietly make an item unanswerable; a stray one would invent a class.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# arm -> (language of the state, language of the instructions/criteria)
ARMS: dict[str, tuple[str, str]] = {
    "A": ("en", "en"),
    "B": ("es", "en"),
    "C": ("es", "es"),
}


@dataclass(frozen=True)
class QuestionSpec:
    """One frozen question wording, in one language, for one dataset."""

    dataset: str
    lang: str
    question_id: str
    type: str
    instructions: str
    criteria: dict[str, str]

    def payload(self) -> dict[str, dict[str, Any]]:
        """The `questions` object to send to Jev.

        One question per call, per the pre-registration -- batching questions would let one
        question's wording influence another's answer.
        """
        q: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria:
            q["criteria"] = dict(self.criteria)
        return {self.question_id: q}


def prompt_path(dataset: str, lang: str, prompts_dir: Path) -> Path:
    return prompts_dir / f"{dataset}.{lang}.json"


def load_prompt(dataset: str, lang: str, prompts_dir: Path) -> QuestionSpec:
    path = prompt_path(dataset, lang, prompts_dir)
    if not path.exists():
        raise FileNotFoundError(f"missing prompt file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return QuestionSpec(
        dataset=raw["dataset"],
        lang=raw["lang"],
        question_id=raw["question_id"],
        type=raw["type"],
        instructions=raw["instructions"],
        criteria=dict(raw.get("criteria") or {}),
    )


def load_arm_prompt(dataset: str, arm: str, prompts_dir: Path) -> QuestionSpec:
    """The prompt an arm uses. Arms A and B deliberately share the English wording."""
    _, prompt_lang = ARMS[arm]
    return load_prompt(dataset, prompt_lang, prompts_dir)


class PromptValidationError(ValueError):
    """The prompt set violates an invariant the experimental design depends on."""


def validate_prompt_set(
    dataset: str, options: tuple[str, ...], prompts_dir: Path, langs: tuple[str, ...] = ("en", "es")
) -> None:
    """Check every language's prompt for one dataset. Raises on the first violation.

    Called by ``make check-prereg`` and again by the runner before it writes anything, so a
    broken prompt set cannot reach a real run.
    """
    specs = {lang: load_prompt(dataset, lang, prompts_dir) for lang in langs}

    reference = specs[langs[0]]
    expected_keys = set(options)

    for lang, spec in specs.items():
        if spec.dataset != dataset:
            raise PromptValidationError(
                f"{dataset}.{lang}.json declares dataset {spec.dataset!r}"
            )
        if spec.lang != lang:
            raise PromptValidationError(f"{dataset}.{lang}.json declares lang {spec.lang!r}")
        if spec.question_id != reference.question_id:
            raise PromptValidationError(
                f"{dataset}.{lang}.json uses question id {spec.question_id!r}, "
                f"but {langs[0]} uses {reference.question_id!r}; the id must be identical "
                f"so answers line up across arms"
            )
        if spec.type != reference.type:
            raise PromptValidationError(
                f"{dataset}.{lang}.json is type {spec.type!r}, {langs[0]} is {reference.type!r}"
            )

        got = set(spec.criteria)
        if got != expected_keys:
            missing = sorted(expected_keys - got)
            extra = sorted(got - expected_keys)
            raise PromptValidationError(
                f"{dataset}.{lang}.json criteria keys do not match the label space. "
                f"missing={missing[:10]} extra={extra[:10]}"
            )

        if not spec.instructions.strip():
            raise PromptValidationError(f"{dataset}.{lang}.json has empty instructions")
        for key, desc in spec.criteria.items():
            if not str(desc).strip():
                raise PromptValidationError(f"{dataset}.{lang}.json criteria[{key!r}] is empty")

    # Invariant 1: identical English keys everywhere. Compared as ordered tuples because the
    # option order is part of the frozen wording, not an implementation detail.
    key_orders = {lang: tuple(spec.criteria) for lang, spec in specs.items()}
    if len(set(key_orders.values())) != 1:
        raise PromptValidationError(
            f"{dataset}: criteria keys differ in order or content across languages. "
            f"Keys must be identical English strings in every arm. Got: "
            + "; ".join(f"{lang}={ko[:4]}..." for lang, ko in key_orders.items())
        )

    # A translated prompt that is byte-identical to the English one is almost certainly an
    # untranslated placeholder that would silently collapse arm C into arm B.
    for lang, spec in specs.items():
        if lang == langs[0]:
            continue
        if spec.instructions.strip() == reference.instructions.strip():
            raise PromptValidationError(
                f"{dataset}.{lang}.json instructions are identical to {langs[0]}; "
                f"arm C would be indistinguishable from arm B"
            )


def all_prompt_files(prompts_dir: Path) -> list[Path]:
    """Every file covered by the prompt freeze, in a stable order."""
    return sorted(prompts_dir.glob("*.json"))
