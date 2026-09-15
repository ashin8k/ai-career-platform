#!/bin/bash

# Start FastAPI backend server in the background on port 8000
PYTHONPATH=. python3 backend/main.py &

# Wait 3 seconds for backend initialization
sleep 3

# Start Streamlit frontend server on Hugging Face port 7860
PYTHONPATH=. python3 -m streamlit run frontend/streamlit_app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true
