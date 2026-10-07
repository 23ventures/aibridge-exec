# aibridge-exec HTTP server. Stateless: no volumes, no database.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[server]" && useradd --create-home --uid 10001 exec
USER exec

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["aibridge-exec", "serve", "--host", "0.0.0.0", "--port", "8000"]
