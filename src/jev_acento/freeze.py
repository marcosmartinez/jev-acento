"""The freeze: making the pre-registration mechanically binding.

A pre-registration that can be quietly edited after seeing the data is worth nothing. This
module turns "we promised not to change the prompts" into something the runner can *check*, so
the promise does not depend on anybody's memory or good intentions.

Two gates, both enforced by :func:`assert_frozen` before the runner writes a single row:

1. **Hashes match.** ``PREREG.sha256`` and ``prompts.sha256`` pin the exact bytes of the
   pre-registration and of every prompt file. Any edit changes a hash and stops the run.
2. **Git is clean for frozen files.** Hashes can be regenerated, so they only prove
   self-consistency. Requiring the frozen files to be committed and unmodified in git means an
   edit has to leave a trace in history that a reader can find.

The escape hatch is deliberate and loud: you may re-freeze, but doing so is a commit, with a
diff, that anyone auditing the repo will see.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(paths: list[Path], manifest: Path, root: Path) -> None:
    """Write a ``sha256sum``-compatible manifest with repo-relative paths."""
    lines = [f"{sha256_file(p)}  {p.relative_to(root).as_posix()}" for p in sorted(paths)]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class FreezeProblem:
    path: str
    problem: str

    def __str__(self) -> str:
        return f"{self.path}: {self.problem}"


def verify_manifest(manifest: Path, root: Path) -> list[FreezeProblem]:
    """Check every file listed in a manifest, and catch files that were added since.

    An *extra* prompt file is a failure, not a curiosity: it means the frozen set is no longer
    the set that will run.
    """
    problems: list[FreezeProblem] = []
    if not manifest.exists():
        return [FreezeProblem(manifest.name, "manifest does not exist -- nothing is frozen")]

    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, _, rel = line.partition("  ")
        listed.add(rel)
        target = root / rel
        if not target.exists():
            problems.append(FreezeProblem(rel, "listed in the manifest but missing on disk"))
            continue
        actual = sha256_file(target)
        if actual != expected:
            problems.append(
                FreezeProblem(rel, f"hash changed since freeze (expected {expected[:12]}..., "
                                   f"got {actual[:12]}...)")
            )

    # Only meaningful for the prompts manifest, which covers a whole directory.
    if manifest.name == "prompts.sha256":
        on_disk = {p.relative_to(root).as_posix() for p in (root / "prompts").glob("*.json")}
        for extra in sorted(on_disk - listed):
            problems.append(FreezeProblem(extra, "exists on disk but is not in the freeze"))

    return problems


def git_dirty_files(root: Path, paths: list[str]) -> list[str]:
    """Frozen paths with uncommitted modifications, staged or not.

    Returns an empty list when git is unavailable rather than failing the run -- the hash check
    is the primary gate, and this one is defence in depth.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []

    dirty = []
    for line in result.stdout.splitlines():
        if len(line) > 3:
            dirty.append(line[3:].strip())
    return dirty


FROZEN_PATHS = ["PREREG.md", "PREREG.sha256", "prompts", "prompts.sha256"]


class NotFrozen(RuntimeError):
    """The freeze does not verify. The runner must not write to ``runs/``."""


def check_freeze(root: Path) -> list[str]:
    """Every reason this repo is not in a runnable frozen state. Empty list means good."""
    issues: list[str] = []
    for manifest_name in ("PREREG.sha256", "prompts.sha256"):
        issues.extend(str(p) for p in verify_manifest(root / manifest_name, root))
    for dirty in git_dirty_files(root, FROZEN_PATHS):
        issues.append(f"{dirty}: uncommitted change to a frozen file")
    return issues


def assert_frozen(root: Path) -> None:
    """Raise unless the repo is in a state where a real run is legitimate."""
    issues = check_freeze(root)
    if issues:
        raise NotFrozen(
            "refusing to run: the pre-registration freeze does not verify.\n  - "
            + "\n  - ".join(issues)
            + "\n\nRun `make freeze` only if you intend to re-register, and commit the result."
        )


def freeze_all(root: Path) -> None:
    """(Re-)create both manifests from what is currently on disk."""
    prereg = root / "PREREG.md"
    if not prereg.exists():
        raise FileNotFoundError("PREREG.md does not exist; write it before freezing")
    write_manifest([prereg], root / "PREREG.sha256", root)

    prompt_files = sorted((root / "prompts").glob("*.json"))
    if not prompt_files:
        raise FileNotFoundError("prompts/ contains no .json files")
    write_manifest(prompt_files, root / "prompts.sha256", root)
