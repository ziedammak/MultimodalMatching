#!/bin/bash
source venv/bin/activate

# Stop existing services
docker rm -f triton_infer_server 2>/dev/null || true
docker rm -f mongodb 2>/dev/null || true
pkill -f "streamlit run" 2>/dev/null || true

# Start MongoDB
echo "[RESTART] Starting MongoDB..."
docker run -d --name mongodb -p 27017:27017 \
  -v mongo_data:/data/db \
  mongo:latest
echo "[RESTART] MongoDB started!"

# Start Triton
echo "[RESTART] Starting Triton Server..."
docker run --gpus=1 --rm -d -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $PWD/model_repository:/models \
  --name triton_infer_server \
  nvcr.io/nvidia/tritonserver:23.06-py3 \
  tritonserver --model-repository=/models --strict-model-config=false

# Wait for Triton to initialize
echo "[RESTART] Waiting for Triton to initialize..."
timeout 60 bash -c 'until curl -s localhost:8000/v2/health/ready >/dev/null; do sleep 1; done'
echo "[RESTART] Triton is ready!"

# Fetch product data if missing
python fetch_and_build.py


# Build FAISS index if missing
if [ ! -f "data/faiss_index_multimodal.index" ]; then
    echo "[RESTART] Building FAISS index..."
    python build_faiss_index.py
fi

# Launch Streamlit
echo "[RESTART] Launching Streamlit app..."
streamlit run streamlit_app.py --server.port 8501 --server.enableCORS false &