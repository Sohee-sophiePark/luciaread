# LuciaRead — developer commands.
# No target reads .env. Live targets (dev with RUN_MODE=live, record) take GEMINI_API_KEY from the
# shell environment: run `set -a; . ./.env; set +a` once in your terminal first.

API_PORT ?= $(shell uv run luciaread port)

.PHONY: setup test lint format eval split train samples api web dev record build-static clean

setup:
	uv sync
	npm --prefix web install

test:
	RUN_MODE=replay uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

eval:
	RUN_MODE=replay uv run python evals/run_evals.py

split:
	uv run luciaread split

train:  # needs data/raw (see data/README.md); writes weights/ and reports/model_card.json
	uv run python -m luciaread.ml.train --task modality --epochs 1
	uv run python -m luciaread.ml.train --task cxr --epochs 5
	uv run python -m luciaread.ml.train --task oct --epochs 5
	uv run python -m luciaread.ml.train --task cxr --epochs 2 --control
	uv run python -m luciaread.ml.train --task oct --epochs 2 --control

samples:
	uv run luciaread samples

api:
	DEV_CONSOLE=1 uv run uvicorn luciaread.api.app:create_app --factory --host 127.0.0.1 --port $(API_PORT) --reload

web:
	API_PORT=$(API_PORT) npm --prefix web run dev

dev:
	@trap 'kill 0' INT TERM; p=$$(uv run luciaread port); $(MAKE) api API_PORT=$$p & $(MAKE) web API_PORT=$$p & wait

record:
	RUN_MODE=record uv run luciaread record

build-static:
	uv run luciaread export web/public
	VITE_STATIC=1 npm --prefix web run build

clean:
	rm -rf .pytest_cache .ruff_cache runs
