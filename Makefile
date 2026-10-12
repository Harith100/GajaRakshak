.PHONY: install run test bench up perception-setup perception-report perception-stream

install:
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt -r perception/requirements.txt

run:                       # laptop mode: in-process bus + SQLite
	.venv/bin/uvicorn gajarakshak.api:app --port 8000

up:                        # full stack: EMQX + Postgres/PostGIS/Timescale + app
	docker compose up --build

test:
	.venv/bin/pytest -q tests

bench:
	.venv/bin/python -m gajarakshak.bench --hours 24 --seeds 5

perception-setup:
	.venv/bin/python -m perception.download_model

perception-report:
	.venv/bin/python -m perception.run report --verify

perception-stream:
	.venv/bin/python -m perception.run stream --mqtt mqtt://localhost:1883 --seconds 120
