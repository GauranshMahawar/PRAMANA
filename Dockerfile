# PRAMANA live assessment service -- the hosted T1 engine (Render).
#
# Not the air-gapped image: that one installs from a signed local wheelhouse with the
# network cable out (see the repository's main Dockerfile). This one is for the public
# evaluation instance and installs from package indexes.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# the exact torch the null was fitted under, CPU build
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pramana/ pramana/
COPY api/__init__.py api/live.py api/live_store.py api/live_app.py api/
COPY backend/assets/ backend/assets/
COPY taxonomy.yaml .

EXPOSE 10000
CMD ["sh", "-c", "uvicorn api.live_app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
