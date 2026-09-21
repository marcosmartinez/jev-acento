"""The runner against the fake API: parsing, resumption, the spend cap, and row integrity."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from jev_acento.providers import (
    PROVIDERS,
    CostCapExceeded,
    JevClient,
    is_versioned_model_id,
    parse_answer,
    parse_response,
)
from jev_acento.questions import load_prompt
from jev_acento.run import (
    CellPlan,
    FakeJev,
    RunSummary,
    _run_cell,
    cell_path,
    completed_item_ids,
    probe_question_overhead,
)

# ---------------------------------------------------------------- response parsing


def test_choice_answer_uses_argmax_not_the_stated_choice():
    """Rounding to 2 decimals makes ties routine; argmax is the single source of truth."""
    answer = parse_answer("q", {
        "type": "choice",
        "choice": "neutral",                      # the provider's pick...
        "probabilities": {"entailment": 0.55, "neutral": 0.45, "contradiction": 0.0},
        "confidence": 0.3,
    })
    assert answer.pred == "entailment"            # ...disagrees with argmax
    assert answer.choice_disagrees_with_argmax is True
    assert answer.p_max == pytest.approx(0.55)


def test_choice_tie_is_broken_deterministically_by_key():
    a = parse_answer("q", {"type": "choice", "probabilities": {"b": 0.5, "a": 0.5}})
    b = parse_answer("q", {"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}})
    assert a.pred == b.pred == "a", "key order from the API must not change the answer"


def test_noul_is_expanded_into_an_explicit_distribution():
    answer = parse_answer("q", {"type": "noul", "noul": 0.95})
    assert answer.pred == "true"
    assert answer.probs == {"true": 0.95, "false": 0.05}
    assert answer.p_max == pytest.approx(0.95)
    assert answer.confidence is None, "Noul answers carry no confidence field"


def test_noul_below_half_predicts_false_and_p_max_flips():
    answer = parse_answer("q", {"type": "noul", "noul": 0.2})
    assert answer.pred == "false"
    assert answer.p_max == pytest.approx(0.8)


def test_noul_exact_tie_resolves_to_false():
    """Pre-registered: 0.50 is `false`, and such cases are counted rather than dropped."""
    assert parse_answer("q", {"type": "noul", "noul": 0.5}).pred == "false"


def test_cost_prefers_market_cost_over_the_gateway_zero():
    """The Gateway reports cost "0" under system credentials; that must not defeat the cap."""
    body = {
        "model": "typesafe-ai/jev",
        "answers": {"q": {"type": "noul", "noul": 0.9}},
        "usage": {"input_tokens": 403, "output_tokens": 45},
        "provider_metadata": {"gateway": {
            "cost": "0", "marketCost": "0.000016926", "generationId": "gen_abc",
        }},
    }
    resp = parse_response(body, latency_ms=12.0)
    assert resp.cost_usd == pytest.approx(0.000016926)
    assert resp.generation_id == "gen_abc"
    assert resp.input_tokens == 403


def test_cost_falls_back_to_the_published_price():
    body = {
        "answers": {"q": {"type": "noul", "noul": 0.9}},
        "usage": {"input_tokens": 1_000_000},
        "provider_metadata": {"gateway": {"cost": "0"}},
    }
    assert parse_response(body, 1.0).cost_usd == pytest.approx(0.042)


def test_unknown_answer_type_is_rejected_loudly():
    with pytest.raises(ValueError, match="unknown answer type"):
        parse_answer("q", {"type": "ordinal", "value": 3})


# ---------------------------------------------------------------- the fake API


def test_fake_api_is_deterministic():
    fake = FakeJev()
    q = {"topic": {"type": "choice", "criteria": {"yes": "y", "no": "n"}}}
    a = asyncio.run(fake.ask({"text": "hello"}, q))
    b = asyncio.run(fake.ask({"text": "hello"}, q))
    assert a.answers["topic"].probs == b.answers["topic"].probs


def test_fake_api_respects_its_accuracy_target():
    fake = FakeJev(target_accuracy=0.8)
    q = {"topic": {"type": "choice", "criteria": {"yes": "y", "no": "n"}}}
    hits = sum(
        asyncio.run(fake.ask({"text": f"item {i}"}, q, gold="yes")).answers["topic"].pred == "yes"
        for i in range(400)
    )
    assert 0.7 < hits / 400 < 0.95


def test_overhead_probe_sends_empty_fields_not_an_empty_object(prompts_dir: Path):
    spec = load_prompt("toy", "en", prompts_dir)
    fake = FakeJev()
    with_fields = asyncio.run(probe_question_overhead(fake, spec, ["text"]))
    assert with_fields > 0


# ---------------------------------------------------------------- cells and resumption


def _plan(prompts_dir: Path, toy_items, arm="A", pass_k=0) -> CellPlan:
    return CellPlan(
        dataset="toy", arm=arm, pass_k=pass_k, items=toy_items,
        spec=load_prompt("toy", "en", prompts_dir), state_lang="en",
    )


def test_cell_writes_one_row_per_item(tmp_path: Path, prompts_dir: Path, toy_items):
    runs = tmp_path / "runs"
    runs.mkdir()
    summary = RunSummary(run_id="t")
    asyncio.run(_run_cell(FakeJev(), _plan(prompts_dir, toy_items), runs, summary))

    path = cell_path(runs, "t", "toy", "A", 0)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(toy_items)
    assert summary.calls == len(toy_items)
    assert {r["item_id"] for r in rows} == {i.item_id for i in toy_items}

    row = rows[0]
    for field in ("dataset", "item_id", "arm", "pass", "gold", "pred", "p_max", "probs",
                  "state_hash", "input_tokens", "latency_ms", "model", "generation_id", "ts"):
        assert field in row, f"row is missing {field!r}"


def test_run_is_resumable_and_does_not_duplicate_work(tmp_path: Path, prompts_dir: Path, toy_items):
    runs = tmp_path / "runs"
    runs.mkdir()

    first = RunSummary(run_id="t")
    asyncio.run(_run_cell(FakeJev(), _plan(prompts_dir, toy_items[:5]), runs, first))
    assert first.calls == 5

    # Resume with the full item list: only the seven new items should be called.
    second = RunSummary(run_id="t")
    asyncio.run(_run_cell(FakeJev(), _plan(prompts_dir, toy_items), runs, second))
    assert second.calls == 7
    assert second.skipped == 5

    path = cell_path(runs, "t", "toy", "A", 0)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len({r["item_id"] for r in rows}) == len(toy_items)


def test_a_truncated_final_line_is_tolerated(tmp_path: Path):
    path = tmp_path / "cell.jsonl"
    path.write_text('{"item_id": "a"}\n{"item_id": "b"}\n{"item_id": "c", "par\n',
                    encoding="utf-8")
    assert completed_item_ids(path) == {"a", "b"}


def test_completed_ids_of_a_missing_file_is_empty(tmp_path: Path):
    assert completed_item_ids(tmp_path / "nope.jsonl") == set()


# ---------------------------------------------------------------- the spend cap


def test_spend_cap_refuses_before_spending(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key-not-real")
    client = JevClient(PROVIDERS["gateway"], max_usd=0.01)
    client.spent_usd = 0.02  # already over
    with pytest.raises(CostCapExceeded):
        asyncio.run(client.ask({"a": "b"}, {"q": {"type": "noul", "instructions": "x"}}))
    assert client.calls == 0, "the cap must stop the call, not merely record it afterwards"
    asyncio.run(client.aclose())


def test_missing_api_key_is_reported_clearly(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AI_GATEWAY_API_KEY"):
        JevClient(PROVIDERS["gateway"])


def test_direct_provider_uses_a_versioned_model_id():
    """The Gateway cannot pin a version; the direct provider must, so it does."""
    assert PROVIDERS["typesafe"].model == "jev-1.13.0"
    assert PROVIDERS["gateway"].model == "typesafe-ai/jev"


def test_only_the_direct_provider_claims_to_report_a_version():
    assert PROVIDERS["typesafe"].reports_version is True
    assert PROVIDERS["gateway"].reports_version is False


def test_provider_default_rates_match_the_documented_limits():
    assert PROVIDERS["typesafe"].default_rpm == 1200   # documented by TypeSafe
    assert PROVIDERS["gateway"].default_rpm == 600     # undocumented; a conservative guess


def test_endpoints_are_built_correctly_for_both_paths():
    assert PROVIDERS["typesafe"].endpoint == "https://api.typesafe.ai/v1/systemone"
    assert PROVIDERS["typesafe"].models_endpoint == "https://api.typesafe.ai/v1/models"
    assert PROVIDERS["gateway"].endpoint == (
        "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
    )


@pytest.mark.parametrize(
    ("model", "pinned"),
    [
        ("jev-1.13.0", True),
        ("typesafe-ai/jev-1.13.0", True),
        ("typesafe-ai/jev", False),
        ("jev-latest", False),
        ("jev-preview", False),
        ("", False),
    ],
)
def test_version_pin_detection(model: str, pinned: bool):
    assert is_versioned_model_id(model) is pinned


def test_overloaded_is_retryable():
    """TypeSafe documents 529 alongside 429 as a back-off-and-retry status."""
    assert 529 in JevClient.RETRY_STATUSES
    assert 429 in JevClient.RETRY_STATUSES
    assert 400 not in JevClient.RETRY_STATUSES, "a malformed request stays malformed"


def test_pacer_defaults_to_the_provider_rate(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
    client = JevClient(PROVIDERS["typesafe"])
    assert client.pacer.min_interval == pytest.approx(60.0 / 1200)
    asyncio.run(client.aclose())


def test_explicit_rpm_overrides_the_provider_default(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
    client = JevClient(PROVIDERS["typesafe"], rpm=120)
    assert client.pacer.min_interval == pytest.approx(0.5)
    asyncio.run(client.aclose())


def test_direct_path_has_no_gateway_metadata_and_still_accounts_correctly():
    """The direct API returns no provider_metadata, so cost falls back to the list price."""
    body = {
        "model": "jev-1.13.0",
        "answers": {"q": {"type": "noul", "noul": 0.9}},
        "usage": {"input_tokens": 500, "output_tokens": 20},
    }
    resp = parse_response(body, latency_ms=30.0)
    assert resp.model == "jev-1.13.0"
    assert resp.generation_id is None, "no generationId outside the Gateway"
    assert resp.cost_usd == pytest.approx(500 * 0.042 / 1_000_000)
