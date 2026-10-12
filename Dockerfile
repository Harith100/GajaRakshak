FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gajarakshak ./gajarakshak
COPY config ./config
COPY dashboard ./dashboard

RUN useradd -r -u 10001 app && mkdir -p /data && chown app /data
USER app
ENV GR_DB=/data/events.db
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "gajarakshak.api:app", "--host", "0.0.0.0", "--port", "8000"]
