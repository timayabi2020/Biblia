# Dockerfile
FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir fastapi uvicorn openai langchain firebase-admin

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]