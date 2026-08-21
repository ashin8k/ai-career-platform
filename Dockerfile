FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose FastAPI port
EXPOSE 8000
# Expose Streamlit port
EXPOSE 8501

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
