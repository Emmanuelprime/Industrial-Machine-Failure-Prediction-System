FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --no-compile -r requirements.txt \
    && python -m pip check

COPY --chown=10001:10001 server ./server
COPY --chown=10001:10001 models/machine_failure_model.pkl ./models/machine_failure_model.pkl

USER 10001:10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8080/health', timeout=2).read()"]

CMD ["gunicorn", "--bind=0.0.0.0:8080", "--workers=2", "--threads=4", "--timeout=60", "--graceful-timeout=30", "--access-logfile=-", "--error-logfile=-", "server.app:app"]
