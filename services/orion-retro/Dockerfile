FROM python@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89 AS core
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential ca-certificates
RUN git clone https://github.com/libretro/libretro-fceumm.git /src && \
    cd /src && git checkout ac0d00a7fc01166f84c7bac01d1084ebfd2b865e && \
    make -f Makefile.libretro -j2

FROM python@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY services/orion-retro/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=core /src/fceumm_libretro.so /opt/retro/core.so
COPY --from=core /src/Copying /opt/retro/FCEUmm-GPL-2.0.txt
COPY orion/retro /app/orion/retro
COPY services/orion-retro/settings.py /app/settings.py
COPY services/orion-retro/app /app/app
COPY services/orion-retro/evals /app/evals
USER 1000:1000
CMD ["uvicorn", "app.main:app", "--uds", "/ipc/retro.sock", "--workers", "1", "--ws-max-size", "4096", "--ws-max-queue", "4", "--limit-concurrency", "32", "--no-access-log"]
