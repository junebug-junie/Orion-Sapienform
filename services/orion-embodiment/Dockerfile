FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY services/orion-embodiment/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY orion /app/orion
# Family C run-cap override bounds (orion/physiology/run_cap_override.py reads /app/config/physiology/).
COPY config/physiology/run_cap_overrides.yaml /app/config/physiology/run_cap_overrides.yaml
COPY services/orion-embodiment /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8130"]
