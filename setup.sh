#!/bin/bash
set -e
echo "[SETUP] Creating Python venv..."
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install onnx transformers sentence-transformers

echo "[SETUP] Exporting text + vision encoders..."
python3 export_text_and_vision_to_onnx.py

echo "[SETUP] Quantizing with TensorRT (via Docker)..."
mkdir -p model_repository/text_encoder/1
mkdir -p model_repository/vision_encoder/1

# Get host CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: //' | sed 's/\..*//')
echo "[INFO] Detected host CUDA Version: ${CUDA_VERSION}.x"

# Select compatible TensorRT version
if [ "$CUDA_VERSION" -ge 12 ]; then
    TRT_TAG="23.06-py3"
elif [ "$CUDA_VERSION" -eq 11 ]; then
    TRT_TAG="22.12-py3"
else
    echo "[ERROR] Unsupported CUDA version: ${CUDA_VERSION}.x"
    exit 1
fi
echo "[INFO] Using TensorRT container: nvcr.io/nvidia/tensorrt:${TRT_TAG}"

# Quantize text encoder
docker run --rm --gpus all -v $PWD:/workspace nvcr.io/nvidia/tensorrt:${TRT_TAG} \
    trtexec --onnx=/workspace/text_encoder.onnx \
            --saveEngine=/workspace/model_repository/text_encoder/1/model.plan \
            --fp16 \
            --minShapes=input_ids:1x32 \
            --optShapes=input_ids:8x32 \
            --maxShapes=input_ids:32x32

# Quantize vision encoder
docker run --rm --gpus all -v $PWD:/workspace nvcr.io/nvidia/tensorrt:${TRT_TAG} \
    trtexec --onnx=/workspace/vision_encoder.onnx \
            --saveEngine=/workspace/model_repository/vision_encoder/1/model.plan \
            --fp16 \
            --minShapes=pixel_values:1x3x224x224 \
            --optShapes=pixel_values:8x3x224x224 \
            --maxShapes=pixel_values:32x3x224x224

echo "[SETUP COMPLETE] Ready to launch Triton."