# Base image (pyproject.toml requires Python >= 3.13)
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first so this layer is cached when only the code changes
COPY uv.lock pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Copy the project and install it
COPY src/ src/
COPY data/ data/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Output directories used by train.py
RUN mkdir -p models reports/figures

ENTRYPOINT ["uv", "run", "--no-sync", "train"]
