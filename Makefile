# jev-acento — see README.md for what each stage is for.
#
# Typical first run:
#     make install && make test && make dry-run
#
# Publishing a real run:
#     make sample && make freeze && git commit && make run && make reproduce

PYTHON  := .venv/bin/python
ACENTO  := .venv/bin/acento
PYTEST  := .venv/bin/pytest

# The HF cache lives inside the repo but is gitignored, so a clean clone re-downloads rather
# than silently reusing a stale copy of a dataset.
export HF_DATASETS_CACHE := $(CURDIR)/data/cache
export HF_HUB_DISABLE_PROGRESS_BARS := 1

.PHONY: help install test lint dry-run sample freeze check-prereg run reproduce figures verify scan-secrets clean distclean

help:
	@echo "install       create .venv and install the package with dev extras"
	@echo "test          run the test suite (metrics, runner, freeze, decision rule)"
	@echo "lint          ruff"
	@echo "dry-run       full pipeline against the local fake API: no network, no spend"
	@echo "sample        build data/items.parquet (ids, gold and hashes only)"
	@echo "freeze        hash PREREG.md and prompts/ -- this is the registration"
	@echo "check-prereg  verify the freeze still holds; the runner does this too"
	@echo "run           the real audit (~19k calls, ~USD 0.50, ~35 min)"
	@echo "reproduce     regenerate results.* and figures/ from runs/, deterministically"
	@echo "verify        the pre-publication checklist, in the order it has to run"
	@echo "scan-secrets  check the whole git history for keys and source text"
	@echo "clean         remove dry-run artefacts and caches"

# Idempotent: `verify` depends on this, so it has to be safe to run against an existing venv.
install:
	uv venv --python 3.12 --allow-existing
	uv pip install -e ".[dev]"

test:
	$(PYTEST)

lint:
	.venv/bin/ruff check src tests

# A dry run is deliberately exempt from the freeze check: it writes only fabricated numbers, and
# requiring a freeze would make the harness undevelopable.
dry-run:
	$(ACENTO) run --dry-run --max-items 40 --run-id dryrun
	$(ACENTO) reproduce --run-id dryrun
	@echo
	@echo "Dry run complete. The numbers above are FABRICATED -- they come from the fake API."

sample:
	$(ACENTO) sample

freeze:
	$(ACENTO) freeze
	@echo
	@echo "Now commit PREREG.md, PREREG.sha256, prompts/ and prompts.sha256."
	@echo "The runner checks git as well as the hashes, so an uncommitted freeze will not run."

check-prereg:
	$(ACENTO) check-prereg

# Requires a key for whichever provider you choose: AI_GATEWAY_API_KEY for the Gateway
# (default) or TYPESAFE_API_KEY for the direct path. Put it in .env; see .env.example.
# Override the provider with: make run PROVIDER=typesafe
PROVIDER ?= gateway

run: check-prereg
	@set -a; [ -f .env ] && . ./.env; set +a; $(ACENTO) run --provider $(PROVIDER)

reproduce:
	$(ACENTO) reproduce

figures:
	$(ACENTO) figures

# The Phase 2b gate. Ordering is load-bearing: check-prereg needs the venv to exist, and the
# reproducibility check only means anything when it runs against already-committed artefacts.
verify: install check-prereg test scan-secrets
	@echo
	@echo "--- reproducibility: regenerate and compare against what is committed ---"
	@cp results.json /tmp/jev-acento-results-before.json
	@$(ACENTO) reproduce >/dev/null
	@diff -q /tmp/jev-acento-results-before.json results.json >/dev/null \
		&& echo "results.json regenerates byte-identical" \
		|| (echo "FAIL: results.json changed on regeneration" && exit 1)
	@git diff --quiet -- figures/ \
		&& echo "figures regenerate byte-identical" \
		|| (echo "FAIL: figures changed on regeneration" && exit 1)
	@echo
	@echo "VERIFIED. Safe to tag and publish."

# Greps the whole history, not just the working tree: a key deleted in a later commit is still
# a leaked key.
scan-secrets:
	@echo "--- scanning git history ---"
	@if [ -f .env ]; then \
		while IFS='=' read -r k v; do \
			[ -z "$$v" ] && continue; \
			if git log -p --all | grep -qF "$$v"; then \
				echo "FAIL: the value of $$k appears in git history"; exit 1; fi; \
		done < .env; \
		echo "no .env value appears in the history"; \
	else echo "no .env present; skipping value scan"; fi
	@test -z "$$(git log --all --name-only --format='' | sort -u | grep '^\.env$$')" \
		&& echo ".env was never committed" \
		|| (echo "FAIL: .env is in the history" && exit 1)
	@test "$$(git log -p --all -- 'runs/*.jsonl' | grep -cE 'premise|hypothesis|flores_passage|sentence1')" = "0" \
		&& echo "no dataset source text in runs/" \
		|| (echo "FAIL: dataset source text found in runs/" && exit 1)

clean:
	rm -rf runs/dryrun-*.jsonl .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf .venv data/cache
