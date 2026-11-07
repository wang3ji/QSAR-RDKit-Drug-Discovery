# Pinned base so a rebuild a year from now still produces the same numbers.
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="a2a-qsar"
LABEL org.opencontainers.image.description="Lab-in-the-loop QSAR for the adenosine A2A receptor"
LABEL org.opencontainers.image.source="https://github.com/wang3ji/QSAR-RDKit-Drug-Discovery"
LABEL org.opencontainers.image.licenses="GPL-3.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# RDKit's wheels need a small set of shared libraries that slim images omit.
RUN apt-get update \
 && apt-get install --no-install-recommends -y libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing source does not invalidate the layer.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[plots]"

COPY data/ ./data/
COPY tests/ ./tests/

# A non-root user, because nothing here needs to write outside /app.
RUN useradd --create-home --uid 10001 runner && chown -R runner:runner /app
USER runner

ENTRYPOINT ["a2a-qsar"]
CMD ["--help"]
