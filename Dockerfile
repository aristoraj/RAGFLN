FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY pdfs/ ./pdfs/
COPY start.sh ./start.sh

RUN chmod +x ./start.sh

RUN mkdir -p ./chroma_db

EXPOSE 8000

ENV PORT=8000

CMD ["./start.sh"]
