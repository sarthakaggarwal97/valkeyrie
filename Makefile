.PHONY: sync format format-check lint typecheck test security synth check clean

sync:
	uv sync --locked

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest

security:
	uv run bandit -c pyproject.toml -r src infra
	uv export --locked --all-groups --no-emit-project --format requirements-txt | \
		uv run pip-audit --strict -r /dev/stdin

synth:
	rm -rf cdk.out
	uv run python infra/app.py

check: format-check lint typecheck test security synth

clean:
	rm -rf cdk.out .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
