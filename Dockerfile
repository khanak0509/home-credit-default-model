FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN sed 's/^xgboost==/xgboost-cpu==/' requirements.txt > requirements-cpu.txt \
    && pip install --no-cache-dir -r requirements-cpu.txt

COPY pyproject.toml config.yaml ./
COPY src src
COPY sql sql
COPY reports reports
RUN pip install --no-cache-dir --no-deps -e .

ENV PYTHONHASHSEED=0
EXPOSE 8000
CMD ["uvicorn", "credit_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
