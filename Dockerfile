FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Optimize uv for container environments
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Copy dependency definitions first for better layer caching
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src

# Install project and dev dependencies (pytest, ruff) from lockfile
RUN uv sync --frozen

# Copy project data, tests, and evaluation suite
COPY data ./data
COPY tests ./tests
COPY eval ./eval

# Ensure virtual environment binaries are on PATH
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["uv", "run", "help-desk-agent"]
CMD ["--help"]
