FROM python:3.12-slim AS builder

# 현재 의존성은 모두 manylinux wheel을 제공한다.
# 소스 빌드가 필요한 의존성이 추가되면 이 단계에 build-essential을 설치한다.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CORPUS_DIR=/app/data

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY generation ./generation
COPY pipeline ./pipeline
COPY retrieval ./retrieval

# corpus는 이미지에 포함하지 않고 호스트 볼륨으로 마운트한다
RUN useradd --create-home riido \
    && mkdir -p /app/data \
    && chown -R riido:riido /app
USER riido

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
