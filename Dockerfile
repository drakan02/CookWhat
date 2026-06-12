FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    espeak-ng \
    cron \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000