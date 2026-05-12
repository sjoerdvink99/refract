VENV := .venv/bin
VENV_FROM_BACKEND := ../../.venv/bin
PYTHON := $(VENV)/python3

.PHONY: install backend frontend download tune eval figures test test-frontend lint clean

install:
	uv pip install -e ".[dev,umap]" --python $(VENV)/python3

backend:
	cd app/backend && PYTHONPATH=../.. $(VENV_FROM_BACKEND)/uvicorn server.main:app --reload --port 8000

frontend:
	cd app/frontend && npm run dev

download:
	$(PYTHON) scripts/download.py --datasets all

tune-%:
	$(PYTHON) scripts/tune.py $* --device auto

eval-%:
	$(PYTHON) scripts/eval.py $* --config configs/hparams/$*.yaml --device auto

eval-all:
	@for ds in cora citeseer pubmed texas cornell wisconsin chameleon squirrel actor roman-empire amazon-ratings minesweeper tolokers questions; do \
		echo "=== $$ds ==="; \
		$(PYTHON) scripts/eval.py $$ds --config configs/hparams/$$ds.yaml --device auto || true; \
	done

figures:
	$(PYTHON) scripts/make_figures.py --legacy-import

test:
	$(VENV)/pytest tests -v

test-frontend:
	cd app/frontend && npm test

lint:
	$(VENV)/ruff check src scripts && $(VENV)/mypy src/refract scripts schemas

clean:
	find . -name "__pycache__" -type d ! -path "./.venv/*" ! -path "./app/frontend/node_modules/*" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" ! -path "./.venv/*" -delete 2>/dev/null; true
