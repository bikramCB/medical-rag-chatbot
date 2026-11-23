FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgl1 libglib2.0-0 libgomp1 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8501
CMD ["./start.sh"]
