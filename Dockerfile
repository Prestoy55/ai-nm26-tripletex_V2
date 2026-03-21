FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY tripletex_agent /app/tripletex_agent

RUN pip install --no-cache-dir .

CMD ["sh", "-c", "uvicorn tripletex_agent.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
