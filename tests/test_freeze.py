"""The freeze must actually stop a run. These tests try to sneak changes past it."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from jev_acento import freeze
from jev_acento.run import RunConfig, execute


@pytest.fixture
def frozen_repo(tmp_path: Path, prompts_dir: Path) -> Path:
    """A repo root with PREREG.md and prompts/, freshly frozen."""
    root = tmp_path
    (root / "PREREG.md").write_text("# Pre-registration\n\nseed 1\n", encoding="utf-8")
    freeze.freeze_all(root)
    return root


def test_a_fresh_freeze_verifies(frozen_repo: Path):
    assert freeze.check_freeze(frozen_repo) == []


def test_editing_the_prereg_breaks_the_freeze(frozen_repo: Path):
    (frozen_repo / "PREREG.md").write_text("# Pre-registration\n\nseed 2\n", encoding="utf-8")
    issues = freeze.check_freeze(frozen_repo)
    assert any("PREREG.md" in i and "hash changed" in i for i in issues)


def test_editing_a_prompt_breaks_the_freeze(frozen_repo: Path):
    path = frozen_repo / "prompts" / "toy.es.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["instructions"] = "una redacción distinta"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    issues = freeze.check_freeze(frozen_repo)
    assert any("toy.es.json" in i and "hash changed" in i for i in issues)


def test_adding_a_prompt_after_the_freeze_is_caught(frozen_repo: Path):
    """A new file changes no existing hash, so it needs its own check."""
    (frozen_repo / "prompts" / "toy.pt.json").write_text("{}", encoding="utf-8")
    issues = freeze.check_freeze(frozen_repo)
    assert any("toy.pt.json" in i and "not in the freeze" in i for i in issues)


def test_deleting_a_prompt_is_caught(frozen_repo: Path):
    (frozen_repo / "prompts" / "toy.en.json").unlink()
    issues = freeze.check_freeze(frozen_repo)
    assert any("toy.en.json" in i and "missing on disk" in i for i in issues)


def test_a_missing_manifest_means_nothing_is_frozen(tmp_path: Path):
    issues = freeze.check_freeze(tmp_path)
    assert len(issues) == 2
    assert all("nothing is frozen" in i for i in issues)


def test_assert_frozen_raises_with_every_reason(frozen_repo: Path):
    (frozen_repo / "PREREG.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(freeze.NotFrozen, match="refusing to run"):
        freeze.assert_frozen(frozen_repo)


def test_runner_refuses_a_real_run_when_the_freeze_is_broken(tmp_path: Path, prompts_dir: Path):
    """The gate that matters: no freeze, no rows in runs/."""
    root = tmp_path
    config = RunConfig(run_id="t", dry_run=False, datasets=("xnli",))
    with pytest.raises(freeze.NotFrozen):
        asyncio.run(execute(config, root=root, progress=False))
    assert not (root / "runs").exists(), "nothing may be written before the freeze verifies"


def test_dry_run_is_exempt_because_it_writes_nowhere_real(tmp_path: Path, prompts_dir: Path):
    """A dry run must work in an unfrozen repo, or nobody could develop against it."""
    config = RunConfig(run_id="t", dry_run=True, datasets=())
    summary = asyncio.run(
        execute(config, root=tmp_path, prompts_dir=prompts_dir,
                runs_dir=tmp_path / "runs", progress=False)
    )
    assert summary.calls == 0  # no datasets requested, but crucially: no NotFrozen raised


def test_sha256_matches_the_reference_implementation(tmp_path: Path):
    path = tmp_path / "f.txt"
    path.write_text("hello\n", encoding="utf-8")
    # sha256 of "hello\n"
    assert freeze.sha256_file(path) == (
        "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    )
