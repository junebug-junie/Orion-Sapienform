FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt             && python -m spacy download en_core_web_sm

COPY tag_service.py .

CMD ["uvicorn", "tag_service:app", "--host", "0.0.0.0", "--port", "8001"]
