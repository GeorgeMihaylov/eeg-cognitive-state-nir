FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    COGSTATE_STREAMING_CONFIG=/app/configs/streaming.yaml

WORKDIR /app

RUN groupadd --gid 10001 cogstate \
    && useradd --uid 10001 --gid cogstate --no-create-home --shell /usr/sbin/nologin cogstate

COPY requirements-preprocessing.txt requirements-streaming-api.txt requirements-streaming-cli.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        -r requirements-preprocessing.txt \
        -r requirements-streaming-api.txt \
        -r requirements-streaming-cli.txt

COPY --chown=cogstate:cogstate cogstate ./cogstate
COPY --chown=cogstate:cogstate apps ./apps
COPY --chown=cogstate:cogstate configs ./configs
COPY --chown=cogstate:cogstate artifacts ./artifacts

RUN mkdir -p /app/outputs \
    && chown cogstate:cogstate /app/outputs

USER cogstate

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]

CMD ["python", "-m", "apps.streaming_worker.api", "--config", "configs/streaming.yaml", "--host", "0.0.0.0", "--port", "8000"]
