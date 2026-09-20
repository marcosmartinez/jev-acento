"""`acento compare` — the part of this repo that other people will actually use."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_acento.cli import build_parser, main


@pytest.fixture
def user_data(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(24):
            gold = "yes" if i % 2 == 0 else "no"
            fh.write(json.dumps({
                "id": f"row-{i:03d}",
                "state": {"text": f"item number {i}"},
                "gold": {"topic": gold},
            }) + "\n")
    return path


def _question_file(path: Path, name: str, instructions: str) -> Path:
    path.write_text(json.dumps({"name": name, "questions": {
        "topic": {"type": "choice", "instructions": instructions,
                  "criteria": {"yes": "it is", "no": "it is not"}}}}), encoding="utf-8")
    return path


@pytest.fixture
def two_versions(tmp_path: Path) -> list[str]:
    return [
        str(_question_file(tmp_path / "q.en.json", "en", "Is `text` about weather?")),
        str(_question_file(tmp_path / "q.es.json", "es", "¿`text` trata sobre el clima?")),
    ]


def test_compare_produces_a_report(tmp_path: Path, user_data: Path, two_versions):
    out = tmp_path / "reporte"
    assert main(["compare", "--data", str(user_data), "--questions", *two_versions,
                 "--out", str(out), "--dry-run"]) == 0

    report = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert report["baseline"] == "en"
    assert report["versions"] == ["en", "es"]

    entry = report["questions"][0]
    assert entry["question_id"] == "topic"
    assert entry["n"] == 24, "both versions must be scored on the same items"
    assert {a["version"] for a in entry["arms"]} == {"en", "es"}
    assert entry["deltas"][0]["vs"] == "en"

    markdown = (out / "results.md").read_text(encoding="utf-8")
    assert "Δ accuracy" in markdown
    assert "ECE ratio below 1.5" in markdown, "the report must warn about unmeasurable ECE"
    assert "finite-sample noise" in markdown


def test_compare_rejects_mismatched_question_ids(tmp_path: Path, user_data: Path):
    a = _question_file(tmp_path / "a.json", "a", "Is `text` about weather?")
    b = tmp_path / "b.json"
    b.write_text(json.dumps({"name": "b", "questions": {
        "tema": {"type": "choice", "instructions": "otra", "criteria": {"yes": "s", "no": "n"}}}}),
        encoding="utf-8")
    with pytest.raises(SystemExit, match="same question ids"):
        main(["compare", "--data", str(user_data), "--questions", str(a), str(b),
              "--out", str(tmp_path / "o"), "--dry-run"])


def test_compare_needs_at_least_two_versions(tmp_path: Path, user_data: Path):
    a = _question_file(tmp_path / "a.json", "a", "Is `text` about weather?")
    with pytest.raises(SystemExit, match="at least two versions"):
        main(["compare", "--data", str(user_data), "--questions", str(a),
              "--out", str(tmp_path / "o"), "--dry-run"])


def test_compare_reports_a_missing_field_with_its_line_number(tmp_path: Path, two_versions):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "state": {}, "gold": {"topic": "yes"}}\n{"id": "b"}\n',
                   encoding="utf-8")
    with pytest.raises(SystemExit, match=r"bad\.jsonl:2: missing required field 'state'"):
        main(["compare", "--data", str(bad), "--questions", *two_versions,
              "--out", str(tmp_path / "o"), "--dry-run"])


def test_compare_accepts_a_bare_question_mapping(tmp_path: Path, user_data: Path):
    """A file may be just {qid: question}, without the `questions` wrapper."""
    for name in ("v1", "v2"):
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "topic": {"type": "choice", "instructions": f"wording {name}",
                      "criteria": {"yes": "y", "no": "n"}}}), encoding="utf-8")
    out = tmp_path / "o"
    assert main(["compare", "--data", str(user_data), "--out", str(out), "--dry-run",
                 "--questions", str(tmp_path / "v1.json"), str(tmp_path / "v2.json")]) == 0
    assert json.loads((out / "results.json").read_text())["versions"] == ["v1", "v2"]


def test_scalar_gold_works_for_a_single_question(tmp_path: Path, two_versions):
    path = tmp_path / "scalar.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(10):
            fh.write(json.dumps({"id": f"r{i}", "state": {"text": f"t{i}"},
                                 "gold": "yes" if i % 2 else "no"}) + "\n")
    out = tmp_path / "o"
    assert main(["compare", "--data", str(path), "--questions", *two_versions,
                 "--out", str(out), "--dry-run"]) == 0


def test_the_examples_directory_actually_works(tmp_path: Path):
    """The README promises this runs on a fresh clone with no key. Hold it to that."""
    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "demo"
    assert main(["compare",
                 "--data", str(repo / "examples" / "tickets.jsonl"),
                 "--questions", str(repo / "examples" / "q.en.json"),
                 str(repo / "examples" / "q.es.json"),
                 "--out", str(out), "--dry-run"]) == 0
    report = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert {q["question_id"] for q in report["questions"]} == {"urgency", "is_complaint"}


def test_parser_exposes_every_documented_subcommand():
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    names = set(actions[0].choices)
    assert names == {"sample", "freeze", "check-prereg", "run", "analyse", "figures",
                     "reproduce", "compare"}
