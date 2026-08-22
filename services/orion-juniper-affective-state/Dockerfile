FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY services/orion-juniper-affective-state/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/orion-juniper-affective-state/app ./app
COPY orion ./orion

EXPOSE 6611
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "6611"]
