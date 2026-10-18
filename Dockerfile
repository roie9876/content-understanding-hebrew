FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 fonts-dejavu-core \
    && apt-get clean

WORKDIR /app
COPY requirements.txt requirements-container.txt ./
RUN pip install --no-cache-dir -r requirements-container.txt

RUN groupadd --gid 10001 inksight \
    && useradd --uid 10001 --gid inksight --no-create-home --shell /usr/sbin/nologin inksight \
    && mkdir -p /data /app/.ui_data /app/analysis_results \
    && chown inksight:inksight /data /app/.ui_data /app/analysis_results

COPY app.py handwriting_pipeline.py segmented_ocr.py public_demo.py container_wsgi.py ./
COPY ui/index.html ui/app.js ui/styles.css ./ui/

ENV AZURE_AUTH_MODE=managed-identity \
    MODEL_BACKEND=azure-responses \
    ENABLE_GOLDEN_COMPARISON=0 \
    PUBLIC_DEMO=1 \
    INKSIGHT_DATA_DIR=/data \
    INKSIGHT_RESULTS_DIR=/data/analysis_results

USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).close()"
CMD ["gunicorn", "--bind=0.0.0.0:8080", "--workers=1", "--threads=8", "--timeout=120", "--access-logfile=-", "--error-logfile=-", "container_wsgi:app"]
