FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY . .

# Ensure entrypoint script is executable
RUN chmod +x /app/entrypoint.sh

# Expose Hugging Face default port 7860
EXPOSE 7860

# Launch both FastAPI backend and Streamlit frontend via entrypoint script
CMD ["/app/entrypoint.sh"]
