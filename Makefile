.PHONY: install run test test-cov lint complexity audit lock format clean check

# Install project in editable mode
install:
	pip install -e ".[dev]"

# Run the agent (pass args after --)
run:
	python -m src.main $(ARGS)

# Run tests
test:
	python -m pytest tests/ -v

# Run tests with coverage and gate
test-cov:
	python -m pytest tests/ -v --cov=src --cov-report=term --cov-report=html --cov-fail-under=70

# Lint with ruff (if installed)
lint:
	@which ruff > /dev/null 2>&1 && ruff check src/ || echo "ruff not installed, skipping lint"
	@which ruff > /dev/null 2>&1 && ruff format --check src/ || true

# Complexity check with lizard
complexity:
	@if which lizard > /dev/null 2>&1; then \
		lizard -C 10 -L 150 --warnings_only -l python src/; \
	else \
		echo "lizard not installed, try: pip install lizard"; \
	fi

# Dependency vulnerability audit
audit:
	@if which pip-audit > /dev/null 2>&1; then \
		pip-audit; \
	else \
		echo "pip-audit not installed, try: pip install pip-audit"; \
	fi

# Generate lock file from pyproject.toml
lock:
	@if which pip-compile > /dev/null 2>&1; then \
		pip-compile --extra=dev --output-file=requirements.txt pyproject.toml -q; \
		echo "Generated requirements.txt"; \
	else \
		echo "pip-compile not installed, try: pip install pip-tools"; \
	fi

# Format code
format:
	@which ruff > /dev/null 2>&1 && ruff format src/ || echo "ruff not installed, skipping format"

# Clean up caches and build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/ .pytest_cache/
	rm -f checkpoints.db

# Full check (install + lint + complexity + audit + test with coverage)
check: install lint complexity audit test-cov
