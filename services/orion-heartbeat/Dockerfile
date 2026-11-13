FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY services/orion-heartbeat/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The substrate math is 4x4/16x16 matrices; a multi-threaded OpenBLAS spends
# its time spin-waiting, not computing. Measured 2026-10-30: same wall time,
# 16x the CPU (bursts to 1,600%) with the default thread pool.
ENV OPENBLAS_NUM_THREADS=1

COPY orion /app/orion
COPY services/orion-heartbeat /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7251"]
