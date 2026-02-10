# =============================================================================
# Tweede Kamer Open Data Pipeline — Makefile
# =============================================================================
# One-command workflow so the whole team gets identical data every time.
#
#   make setup   — first time only: create venv + install deps
#   make data    — fetch + preprocess everything (the main one)
#   make fetch   — fetch raw data only
#   make process — preprocess only (needs raw data)
#   make summary — print data summary table
#   make clean   — wipe all data and start fresh
#   make list    — show available entity types
# =============================================================================

PYTHON  ?= python3
VENV    := venv
PIP     := $(VENV)/bin/pip
PY      := $(VENV)/bin/python
CONFIG  := config.yaml

.PHONY: help setup data fetch process summary viz clean list nuke

help: ## Show this help
	@echo ""
	@echo "  Tweede Kamer Data Pipeline"
	@echo "  =========================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""

setup: ## Create venv and install dependencies (first time)
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "  ✓ Setup complete. Run 'make data' to fetch & process all data."
	@echo ""

data: ## Run full pipeline: fetch + preprocess all entities
	$(PY) pipeline.py --config $(CONFIG)

fetch: ## Fetch raw data only (skip preprocessing)
	$(PY) pipeline.py --fetch-only --config $(CONFIG)

process: ## Preprocess only (raw data must exist)
	$(PY) pipeline.py --preprocess-only --config $(CONFIG)

summary: ## Show summary of processed data
	$(PY) pipeline.py --summary --config $(CONFIG)

list: ## List all available entity types
	$(PY) pipeline.py --list-entities --config $(CONFIG)

viz: ## Generate data overview dashboard (dashboard.png)
	$(PY) visualize.py
	@echo ""
	@echo "  ✓ Dashboard saved: dashboard.png"
	@echo ""

clean: ## Delete all fetched and processed data
	rm -rf data/raw/*.json data/processed/*.parquet data/processed/*.csv data/processed/_summary.csv
	@echo "  ✓ Data cleaned. Run 'make data' to re-fetch."

nuke: ## Full reset: delete data + venv (start from scratch)
	rm -rf data/raw/*.json data/processed/*.parquet data/processed/*.csv data/processed/_summary.csv
	rm -rf $(VENV)
	@echo "  ✓ Nuked. Run 'make setup && make data' to start fresh."
