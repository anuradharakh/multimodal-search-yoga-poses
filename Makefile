PYTHON ?= python3
BIN    := .venv/bin

.PHONY: help install pipeline app notebooks test lint format

help:      ## Show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:   ## Create .venv (Python 3.10-3.12) and install requirements
	$(PYTHON) -m venv .venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

pipeline:  ## Run every stage enabled in config.yaml
	$(BIN)/python run_pipeline.py

app:       ## Start the Streamlit app on http://localhost:8501
	$(BIN)/streamlit run app.py --server.port 8501

notebooks: ## Execute the notebooks in place and export HTML to outputs/html/
	cd notebooks && ../$(BIN)/jupyter nbconvert --to notebook --execute --inplace *.ipynb
	$(BIN)/jupyter nbconvert --to html notebooks/*.ipynb --output-dir outputs/html

test:      ## Unit tests (synthetic data, no downloads)
	$(BIN)/pytest

lint:      ## Lint with ruff
	$(BIN)/ruff check .

format:    ## Auto-format with ruff
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .
