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

.PHONY: help install test lint dry-run sample freeze check-prereg run reproduce figures clean distclean

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
	@echo "clean         remove dry-run artefacts and caches"

install:
	uv venv --python 3.12
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

# Requires AI_GATEWAY_API_KEY. Put it in .env (gitignored) and it is picked up here.
run: check-prereg
	@set -a; [ -f .env ] && . ./.env; set +a; $(ACENTO) run

reproduce:
	$(ACENTO) reproduce

figures:
	$(ACENTO) figures

clean:
	rm -rf runs/dryrun-*.jsonl .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf .venv data/cache
