FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip3 install --no-cache-dir --upgrade pip
COPY services/orion-bus/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY orion /app/orion
COPY services/orion-bus /app
CMD ["python", "-m", "app.main"]
