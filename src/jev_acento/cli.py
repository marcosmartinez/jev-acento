"""``acento`` — the command line interface.

Two distinct jobs live behind one entry point:

**The audit** (``sample``, ``freeze``, ``check-prereg``, ``run``, ``analyse``, ``figures``,
``reproduce``) reproduces the published Spanish study. It is opinionated: fixed datasets, fixed
prompts, a pre-registration it refuses to run without.

**The tool** (``compare``) is the part that is useful to anybody else. It runs the *same* harness
and the *same* metrics over your own labelled data and your own question wordings, so you can
answer "should I write my instructions in English or in Spanish?" for your own workload instead
of trusting that a result on XNLI transfers to your tickets.

Nothing leaves the machine except the calls to Jev.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import freeze as freeze_mod
from . import metrics as M
from .analyse import analyse, load_cells, write_results
from .data import (
    DEFAULT_CACHE,
    SPECS,
    freeze_items,
    items_to_rows,
    load_aligned,
    resolve_options,
    stratified_sample,
)
from .figures import make_all
from .providers import (
    PROVIDERS,
    JevClient,
    fetch_model_release_date,
    is_versioned_model_id,
)
from .questions import validate_prompt_set
from .run import REPO_ROOT, FakeJev, RunConfig, execute, iter_rows

DEFAULT_SEED = 20260920


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------------------------
# Audit subcommands
# --------------------------------------------------------------------------------------------


def cmd_sample(args: argparse.Namespace) -> int:
    """Load, align, sample, and write the frozen item table."""
    root = Path(args.root)
    rows: list[dict[str, str]] = []
    for name in args.datasets:
        spec = SPECS[name]
        items, report = load_aligned(spec, Path(args.cache))
        print(f"  {report.summary()}")
        options = resolve_options(spec, items)
        validate_prompt_set(name, options, root / "prompts")
        n = min(spec.n_target, args.max_items) if args.max_items else spec.n_target
        sample = stratified_sample(items, n, args.seed)
        print(f"  {name}: sampled {len(sample)} of {len(items)} (seed {args.seed})")
        rows.extend(items_to_rows(name, sample))

    out = root / "data" / "items.parquet"
    freeze_items(rows, out)
    print(f"\nwrote {len(rows)} items to {out}")
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    root = Path(args.root)
    freeze_mod.freeze_all(root)
    print("froze PREREG.md and prompts/*.json")
    print("  -> PREREG.sha256")
    print("  -> prompts.sha256")
    print("\nCommit both manifests before running. The runner checks git, not just the hashes.")
    return 0


def cmd_check_prereg(args: argparse.Namespace) -> int:
    issues = freeze_mod.check_freeze(Path(args.root))
    if not issues:
        print("freeze verifies: PREREG.md and prompts/ are unchanged since registration")
        return 0
    print("FREEZE DOES NOT VERIFY:", file=sys.stderr)
    for issue in issues:
        print(f"  - {issue}", file=sys.stderr)
    return 1


def cmd_run(args: argparse.Namespace) -> int:
    config = RunConfig(
        run_id=args.run_id or _default_run_id(),
        provider=args.provider,
        datasets=tuple(args.datasets),
        passes=args.passes,
        seed=args.seed,
        rpm=args.rpm,
        concurrency=args.concurrency,
        max_usd=args.max_usd,
        dry_run=args.dry_run,
        n_override=args.max_items,
    )
    print(f"run_id = {config.run_id}"
          f"{'  (DRY RUN — fabricated numbers, nothing spent)' if config.dry_run else ''}")
    summary = asyncio.run(
        execute(config, root=Path(args.root), cache_dir=Path(args.cache))
    )
    if summary.errors:
        print(f"\n{len(summary.errors)} errors; first few:", file=sys.stderr)
        for err in summary.errors[:10]:
            print(f"  - {err}", file=sys.stderr)
    print(f"\nrun_id {config.run_id} — pass this to `acento analyse --run-id`")
    return 0


def _latest_run_id(runs_dir: Path) -> str | None:
    manifest = runs_dir / "manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return max(data, default=None)


def _load_run(root: Path, run_id: str | None) -> tuple[str, list[dict], dict]:
    runs_dir = root / "runs"
    run_id = run_id or _latest_run_id(runs_dir)
    if not run_id:
        raise SystemExit("no run found in runs/manifest.json; run `acento run` first")
    rows = list(iter_rows(runs_dir, run_id))
    if not rows:
        raise SystemExit(f"run {run_id!r} has no rows in {runs_dir}")
    manifest_path = runs_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get(run_id, {})
    return run_id, rows, manifest


def cmd_analyse(args: argparse.Namespace) -> int:
    root = Path(args.root)
    run_id, rows, manifest = _load_run(root, args.run_id)
    results = analyse(rows, run_id, manifest=manifest, seed=args.seed)
    json_path, md_path = write_results(results, root)
    print(f"analysed {len(rows)} rows from run {run_id}")
    print(f"  -> {json_path}")
    print(f"  -> {md_path}")
    for warning in results.warnings:
        print(f"  WARNING: {warning}", file=sys.stderr)
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    root = Path(args.root)
    _, rows, _ = _load_run(root, args.run_id)
    paths = make_all(load_cells(rows), root / "figures")
    for p in paths:
        print(f"  -> {p}")
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Regenerate every published artefact from ``runs/``, deterministically."""
    rc = cmd_analyse(args)
    return rc or cmd_figures(args)


def cmd_smoke(args: argparse.Namespace) -> int:
    """Check what a provider actually returns, before trusting it with a real run.

    Makes three cheap calls -- one Choice, one Noul, and a model listing -- and prints the facts
    the audit depends on: whether the response pins a model version, whether Noul carries a
    confidence field, whether per-call cost is reported, and what the round trip costs. This is
    the Phase 0 smoke test, kept runnable rather than written down once.
    """
    provider = PROVIDERS[args.provider]
    print(f"provider: {provider.name}  ({provider.endpoint})")
    print(f"model sent: {provider.model}\n")

    async def go() -> int:
        client = JevClient(provider, max_usd=args.max_usd)
        try:
            try:
                listing = await fetch_model_release_date(provider)
                print(f"  models endpoint : {listing}")
            except Exception as exc:
                print(f"  models endpoint : unavailable ({exc})")

            choice = await client.ask(
                {"premise": "A man inspects a uniform.", "hypothesis": "The man is asleep."},
                {"nli": {"type": "choice",
                         "instructions": "Decide what the `premise` establishes about the "
                                         "`hypothesis`.",
                         "criteria": {
                             "entailment": "The `hypothesis` must be true.",
                             "neutral": "The `hypothesis` may be true or false.",
                             "contradiction": "The `hypothesis` cannot be true."}}},
            )
            noul = await client.ask(
                {"sentence_1": "Two students founded it in 1998.",
                 "sentence_2": "It was founded in 1998 by two students."},
                {"paraphrase": {"type": "noul",
                                "instructions": "Do `sentence_1` and `sentence_2` state the "
                                                "same facts?"}},
            )
        except Exception as exc:
            print(f"\nFAILED: {exc}", file=sys.stderr)
            await client.aclose()
            return 1

        c, n = choice.answers["nli"], noul.answers["paraphrase"]
        pinned = is_versioned_model_id(choice.model)
        print(f"\n  choice          : {c.pred} p_max={c.p_max} confidence={c.confidence}")
        print(f"  noul            : {n.pred} p_max={n.p_max} confidence={n.confidence}")
        print(f"  model reported  : {choice.model!r}")
        print(f"  VERSION PINNED  : {'YES' if pinned else 'NO — this is an alias'}")
        print(f"  generation_id   : {choice.generation_id}")
        print(f"  input_tokens    : {choice.input_tokens} (choice), "
              f"{noul.input_tokens} (noul)")
        print(f"  cost reported   : {choice.cost_usd:.9f} USD")
        print(f"  latency         : {choice.latency_ms:.0f} ms, {noul.latency_ms:.0f} ms")
        print(f"\n  total spent     : {client.spent_usd:.6f} USD over {client.calls} calls")

        if provider.reports_version and not pinned:
            print("\n  WARNING: this provider is expected to report a versioned model id but "
                  "did not.", file=sys.stderr)
        await client.aclose()
        return 0

    return asyncio.run(go())


# --------------------------------------------------------------------------------------------
# compare: the same harness, on your data
# --------------------------------------------------------------------------------------------


def _load_question_set(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    """Read one question-set file.

    Accepts either a bare mapping of question id to question object, or that mapping under a
    ``"questions"`` key alongside an optional ``"name"``. The version name defaults to the file
    stem, which is what appears in the report.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, dict) and "questions" in doc:
        return str(doc.get("name", path.stem)), doc["questions"]
    return path.stem, doc


def _load_user_items(path: Path) -> list[dict[str, Any]]:
    """Read the user's JSONL: ``id``, ``state``, and ``gold`` per question."""
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            for required in ("id", "state", "gold"):
                if required not in row:
                    raise SystemExit(f"{path}:{lineno}: missing required field {required!r}")
            items.append(row)
    if not items:
        raise SystemExit(f"{path} contains no items")
    return items


def _gold_for(row: dict[str, Any], qid: str, question_ids: list[str]) -> str | None:
    """The gold label for one question of one item.

    ``gold`` may be a mapping from question id to label, or -- when there is exactly one
    question -- a bare scalar.
    """
    gold = row["gold"]
    if isinstance(gold, dict):
        value = gold.get(qid)
    elif len(question_ids) == 1:
        value = gold
    else:
        raise SystemExit(
            f"item {row['id']!r}: `gold` must be an object keyed by question id when there is "
            f"more than one question (questions: {question_ids})"
        )
    return None if value is None else str(value)


async def _run_compare(
    versions: list[tuple[str, dict[str, dict[str, Any]]]],
    items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Run every question-set version over every item. Returns rows per (version, question)."""
    provider = PROVIDERS[args.provider]
    client: Any = (
        FakeJev() if args.dry_run
        else JevClient(provider, rpm=args.rpm, concurrency=args.concurrency,
                       max_usd=args.max_usd)
    )
    collected: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    try:
        for version_name, questions in versions:
            question_ids = list(questions)
            print(f"  version {version_name!r}: {len(items)} items × "
                  f"{len(question_ids)} question(s)")

            async def one(
                row: dict[str, Any],
                qid: str,
                qobj: dict[str, Any],
                # Bound as defaults, not captured: `one` is defined inside the version loop,
                # and a late-binding closure here would silently attribute every answer to the
                # last version if the gather were ever made to outlive the iteration.
                version_name: str = version_name,
                question_ids: list[str] = question_ids,
            ) -> None:
                gold = _gold_for(row, qid, question_ids)
                if gold is None:
                    return
                payload = {qid: qobj}
                try:
                    if isinstance(client, FakeJev):
                        resp = await client.ask(row["state"], payload, gold=gold)
                    else:
                        resp = await client.ask(row["state"], payload)
                except Exception as exc:
                    print(f"    ! {row['id']}/{qid}: {exc}", file=sys.stderr)
                    return
                answer = resp.answers.get(qid)
                if answer is None:
                    return
                collected[(version_name, qid)].append({
                    "id": str(row["id"]),
                    "gold": gold,
                    "pred": answer.pred,
                    "p_max": answer.p_max,
                    "input_tokens": resp.input_tokens,
                    "latency_ms": resp.latency_ms,
                })

            await asyncio.gather(
                *(one(row, qid, qobj) for row in items for qid, qobj in questions.items())
            )
    finally:
        await client.aclose()

    return collected


def _compare_report(
    collected: dict[tuple[str, str], list[dict[str, Any]]],
    baseline: str,
    seed: int,
) -> dict[str, Any]:
    """Per-version metrics plus paired deltas against the baseline version."""
    versions = sorted({v for v, _ in collected})
    question_ids = sorted({q for _, q in collected})
    report: dict[str, Any] = {"baseline": baseline, "versions": versions, "questions": []}

    for qid in question_ids:
        by_version = {v: {r["id"]: r for r in collected.get((v, qid), [])} for v in versions}
        shared = sorted(set.intersection(*(set(d) for d in by_version.values()))) if by_version else []
        if not shared:
            continue

        entry: dict[str, Any] = {"question_id": qid, "n": len(shared), "arms": [], "deltas": []}

        arrays = {}
        for v in versions:
            rows = [by_version[v][i] for i in shared]
            gold = np.array([r["gold"] for r in rows])
            pred = np.array([r["pred"] for r in rows])
            p_max = np.array([r["p_max"] for r in rows])
            correct = gold == pred
            arrays[v] = (gold, pred, p_max, correct, rows)

            observed = M.ece(p_max, correct)
            floor = M.ece_noise_floor(p_max, n_sims=300, seed=seed)
            entry["arms"].append({
                "version": v,
                "accuracy": M.bootstrap_ci(
                    lambda idx, c=correct: M.accuracy(c[idx]), len(shared), seed=seed
                ).as_dict(),
                "macro_f1": M.macro_f1(gold, pred),
                "ece": observed,
                "ece_floor": floor,
                "ece_ratio": observed / floor if floor else float("nan"),
                "mean_input_tokens": float(np.mean([r["input_tokens"] for r in rows])),
                "latency_p50": float(np.percentile([r["latency_ms"] for r in rows], 50)),
                "coverage": [
                    M.coverage_at_threshold(p_max, correct, t) for t in (0.5, 0.9)
                ],
            })

        _, _, p_base, correct_base, rows_base = arrays[baseline]
        for v in versions:
            if v == baseline:
                continue
            _, _, p_v, correct_v, rows_v = arrays[v]
            # Same reason as above: bind the per-version arrays rather than closing over them.
            d_acc = M.paired_delta_ci(
                lambda idx, c=correct_base: M.accuracy(c[idx]),
                lambda idx, c=correct_v: M.accuracy(c[idx]),
                len(shared), seed=seed,
            )
            d_ece = M.paired_delta_ci(
                lambda idx, p=p_base, c=correct_base: M.ece(p[idx], c[idx]),
                lambda idx, p=p_v, c=correct_v: M.ece(p[idx], c[idx]),
                len(shared), seed=seed + 1,
            )
            entry["deltas"].append({
                "version": v,
                "vs": baseline,
                "delta_accuracy": d_acc.as_dict(),
                "delta_ece": d_ece.as_dict(),
                "token_ratio": (
                    float(np.mean([r["input_tokens"] for r in rows_v]))
                    / float(np.mean([r["input_tokens"] for r in rows_base]))
                ),
            })
        report["questions"].append(entry)

    return report


def _render_compare_md(report: dict[str, Any]) -> str:
    out = [
        "# `acento compare` report",
        "",
        f"Baseline version: **{report['baseline']}**. "
        f"Versions compared: {', '.join(report['versions'])}.",
        "",
        "Deltas are paired: every version saw the same items, and the bootstrap resamples item "
        "indices once per replicate and recomputes both versions on that same resample.",
        "",
    ]
    for q in report["questions"]:
        out += [
            f"## `{q['question_id']}` (n = {q['n']})",
            "",
            "| Version | Accuracy | Macro-F1 | ECE | Floor | Ratio | Tokens | p50 ms |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for a in q["arms"]:
            acc = a["accuracy"]
            out.append(
                f"| {a['version']} | {acc['point']:.4f} [{acc['lo']:.4f}, {acc['hi']:.4f}] | "
                f"{a['macro_f1']:.4f} | {a['ece']:.4f} | {a['ece_floor']:.4f} | "
                f"{a['ece_ratio']:.2f} | {a['mean_input_tokens']:.0f} | {a['latency_p50']:.0f} |"
            )
        if q["deltas"]:
            out += [
                "",
                f"| Δ vs {report['baseline']} | Δ accuracy | Δ ECE | Token ratio |",
                "|---|---|---|---|",
            ]
            for d in q["deltas"]:
                da, de = d["delta_accuracy"], d["delta_ece"]
                out.append(
                    f"| {d['version']} | {da['point']:+.4f} [{da['lo']:+.4f}, {da['hi']:+.4f}] | "
                    f"{de['point']:+.4f} [{de['lo']:+.4f}, {de['hi']:+.4f}] | "
                    f"{d['token_ratio']:.3f}× |"
                )
        out += [
            "",
            "An ECE ratio below 1.5 means the calibration error is indistinguishable from "
            "finite-sample noise at this n — do not read anything into the ECE column for that "
            "version.",
            "",
        ]
    return "\n".join(out)


def cmd_compare(args: argparse.Namespace) -> int:
    versions = [_load_question_set(Path(p)) for p in args.questions]
    if len(versions) < 2:
        raise SystemExit("--questions needs at least two versions to compare")

    key_sets = {name: tuple(sorted(qs)) for name, qs in versions}
    if len(set(key_sets.values())) != 1:
        raise SystemExit(
            "every question set must use the same question ids, so the versions are "
            f"comparable. Got: { {k: list(v) for k, v in key_sets.items()} }"
        )

    items = _load_user_items(Path(args.data))
    print(f"{len(items)} items, {len(versions)} versions"
          f"{'  (DRY RUN — fabricated numbers)' if args.dry_run else ''}")

    collected = asyncio.run(_run_compare(versions, items, args))
    if not collected:
        raise SystemExit("no answers collected; nothing to report")

    report = _compare_report(collected, baseline=versions[0][0], seed=args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    (out_dir / "results.md").write_text(_render_compare_md(report), encoding="utf-8")
    print(f"  -> {out_dir / 'results.md'}")
    print(f"  -> {out_dir / 'results.json'}")
    return 0


# --------------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acento",
        description="Audit Jev on Spanish, and compare question wordings on your own data.",
    )
    parser.add_argument("--root", default=str(REPO_ROOT), help="repository root")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        p.add_argument("--cache", default=str(DEFAULT_CACHE))

    def add_api(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", choices=sorted(PROVIDERS), default="gateway")
        p.add_argument("--rpm", type=int, default=None,
                       help="pacer target in requests per minute "
                            "(default: the provider's own limit — gateway 600, typesafe 1200)")
        p.add_argument("--concurrency", type=int, default=8)
        p.add_argument("--max-usd", type=float, default=5.0, help="hard spend cap")
        p.add_argument("--dry-run", action="store_true",
                       help="use the local fake API: no network, no spend, fabricated numbers")

    p_sample = sub.add_parser("sample", help="build data/items.parquet")
    p_sample.add_argument("--datasets", nargs="+", default=sorted(SPECS), choices=sorted(SPECS))
    p_sample.add_argument("--max-items", type=int, default=None)
    add_common(p_sample)
    p_sample.set_defaults(func=cmd_sample)

    p_freeze = sub.add_parser("freeze", help="hash PREREG.md and prompts/ (re-registration)")
    p_freeze.set_defaults(func=cmd_freeze)

    p_check = sub.add_parser("check-prereg", help="verify the freeze still holds")
    p_check.set_defaults(func=cmd_check_prereg)

    p_run = sub.add_parser("run", help="run the audit")
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument("--datasets", nargs="+", default=sorted(SPECS), choices=sorted(SPECS))
    p_run.add_argument("--passes", type=int, default=2)
    p_run.add_argument("--max-items", type=int, default=None,
                       help="cap n per dataset (smoke runs only, never a real run)")
    add_common(p_run)
    add_api(p_run)
    p_run.set_defaults(func=cmd_run)

    for name, fn, help_text in (
        ("analyse", cmd_analyse, "compute results.json and results.md"),
        ("figures", cmd_figures, "regenerate figures/"),
        ("reproduce", cmd_reproduce, "analyse + figures, deterministically"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--run-id", default=None, help="defaults to the most recent run")
        add_common(p)
        p.set_defaults(func=fn)

    p_smoke = sub.add_parser(
        "smoke",
        help="check what a provider actually returns (3 cheap calls)",
        description="Verify a provider before trusting it with a real run: response shape, "
                    "whether the model version is pinned, token counts and cost.",
    )
    p_smoke.add_argument("--provider", choices=sorted(PROVIDERS), default="gateway")
    p_smoke.add_argument("--max-usd", type=float, default=0.01)
    p_smoke.set_defaults(func=cmd_smoke)

    p_cmp = sub.add_parser(
        "compare",
        help="compare question wordings on your own labelled data",
        description="Run two or more versions of the same question set over the same items and "
                    "report paired deltas. Nothing leaves your machine except calls to Jev.",
    )
    p_cmp.add_argument("--data", required=True, help="JSONL with id, state, gold")
    p_cmp.add_argument("--questions", nargs="+", required=True,
                       help="two or more question-set files with identical question ids")
    p_cmp.add_argument("--out", default="reporte", help="output directory")
    add_common(p_cmp)
    add_api(p_cmp)
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
