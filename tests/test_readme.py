"""Every number in the READMEs must come from `results.json`.

The claim "nothing here is hand-typed" is easy to make and easy to break: someone edits a
sentence, re-runs the audit, and the prose quietly keeps the old figure. These tests parse the
published tables and check them against the artefact they claim to come from.

They skip cleanly when `results.json` is absent, so a fresh clone with no run still passes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results.json"

pytestmark = pytest.mark.skipif(
    not RESULTS.exists(), reason="no results.json — nothing has been run yet"
)

DATASET_NAMES = {"xnli": "xnli", "paws-x": "pawsx", "massive": "massive", "belebele": "belebele"}


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def arms(results: dict) -> dict:
    return {(a["dataset"], a["arm"]): a for a in results["arms"]}


@pytest.fixture(scope="module")
def comparisons(results: dict) -> dict:
    return {(c["dataset"], c["label"]): c for c in results["comparisons"]}


def _number(text: str) -> float:
    """Parse a figure from either README: '0,786' and '−6,4' are the Spanish spellings."""
    return float(text.replace(",", ".").replace("−", "-").replace("–", "-"))


@pytest.mark.parametrize("readme", ["README.md", "README.es.md"])
def test_b_minus_a_table_matches_results(readme: str, arms: dict, comparisons: dict):
    """The headline table: per-arm accuracies and the paired delta, in percentage points."""
    text = (REPO / readme).read_text(encoding="utf-8")
    rows = re.findall(
        r"^\|\s*(XNLI|PAWS-X|MASSIVE|Belebele)\s*\|\s*([\d.,]+)\s*\|\s*([\d.,]+)\s*\|"
        r"\s*([−\-\d.,]+)\s*pp",
        text,
        re.MULTILINE,
    )
    assert len(rows) == 4, f"{readme}: expected 4 B-A rows, found {len(rows)}"

    for label, acc_a, acc_b, delta_pp in rows:
        ds = DATASET_NAMES[label.lower()]
        assert _number(acc_a) == pytest.approx(arms[(ds, "A")]["accuracy"]["point"], abs=0.0005)
        assert _number(acc_b) == pytest.approx(arms[(ds, "B")]["accuracy"]["point"], abs=0.0005)
        assert _number(delta_pp) / 100 == pytest.approx(
            comparisons[(ds, "B - A")]["delta_accuracy"]["point"], abs=0.0005
        ), f"{readme}: {label} B-A delta is stale"


@pytest.mark.parametrize("readme", ["README.md", "README.es.md"])
def test_c_minus_b_deltas_match_results(readme: str, comparisons: dict):
    text = (REPO / readme).read_text(encoding="utf-8")
    section = text.split("C − B")[1] if "C − B" in text else ""
    rows = re.findall(
        r"^\|\s*(XNLI|PAWS-X|MASSIVE|Belebele)\s*\|\s*([+−\-][\d.,]+)\s*pp", section, re.MULTILINE
    )
    assert len(rows) == 4, f"{readme}: expected 4 C-B rows, found {len(rows)}"
    for label, delta_pp in rows:
        ds = DATASET_NAMES[label.lower()]
        assert _number(delta_pp.lstrip("+")) / 100 == pytest.approx(
            comparisons[(ds, "C - B")]["delta_accuracy"]["point"], abs=0.0005
        ), f"{readme}: {label} C-B delta is stale"


@pytest.mark.parametrize("readme", ["README.md", "README.es.md"])
def test_every_verdict_word_matches_the_decision_rule(readme: str, comparisons: dict):
    """The prose verdict must be the one the rule actually produced, not a softened paraphrase."""
    text = (REPO / readme).read_text(encoding="utf-8").lower()
    for (ds, label), comp in comparisons.items():
        verdict = comp["accuracy_verdict"]
        if verdict == "MEASURABLY WORSE" and label == "B - A":
            assert "measurably worse" in text or "mediblemente peor" in text
        if verdict == "AMBIGUOUS" and (ds, label) == ("pawsx", "C - B"):
            assert "ambiguous" in text or "ambiguo" in text, (
                f"{readme}: PAWS-X C-B is AMBIGUOUS and the README must not round that to a null"
            )


def test_token_ratio_range_in_the_readme_brackets_the_real_values(comparisons: dict):
    """README claims Spanish costs 17-38% more tokens; check it spans the real B-A ratios."""
    ratios = [c["token_ratio_state_only"] for k, c in comparisons.items() if k[1] == "B - A"]
    assert min(ratios) == pytest.approx(1.17, abs=0.01)
    assert max(ratios) == pytest.approx(1.38, abs=0.01)


def test_the_run_is_version_pinned(results: dict):
    """The headline claims a pinned model. That is a fact about the rows, so check it."""
    prov = results["checks"]["model_provenance"]
    assert prov["status"] == "pinned"
    assert prov["model"] == "jev-1.13.0"
    assert "jev-1.13.0" in (REPO / "README.md").read_text(encoding="utf-8")


def test_no_warnings_survived_into_a_published_run(results: dict):
    """A published run must not carry a mixed-model or unpinned warning."""
    assert results["warnings"] == [], f"published run has warnings: {results['warnings']}"


def test_stability_gate_passed_for_every_published_comparison(results: dict, comparisons: dict):
    """Nothing above may be interpreted if the model's own jitter swamped the effect."""
    for (ds, label), comp in comparisons.items():
        assert comp["accuracy_verdict"] != "UNSTABLE", f"{ds} {label} failed the stability gate"
        assert comp["stability_jitter"] < abs(comp["delta_accuracy"]["point"]) or (
            comp["accuracy_verdict"] == "NO DETECTABLE DIFFERENCE"
        )
