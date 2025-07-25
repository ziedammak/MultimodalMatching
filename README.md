# Multimodal Matching System

This project implements an end-to-end multimodal product matching system. It supports image, text, or combined inputs to find the most similar product from a catalog using deep learning and vector search techniques.

The system leverages quantized deep learning models served via NVIDIA Triton Inference Server and uses FAISS for similarity search. It also includes a Streamlit-based web interface and a MongoDB-backed metadata and logging system.

---

## Project Objectives

- Efficient multimodal retrieval (text, image, or both)
- Scalable inference using quantized models
- Fully containerized deployment
- Modular and testable design

---

## Model Selection

### Text Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Chosen for its balance between speed and semantic accuracy.
- Outputs 384-dimensional sentence embeddings.
- We use the `[CLS]` token (first position) as the embedding.

### Vision Encoder: `openai/clip-vit-base-patch32`
- Selected for its wide adoption and effective visual representation.
- Produces 512-dimensional embeddings of input images.
- Only the vision tower is used, not the full CLIP architecture.

These models are used independently and fused via concatenation and normalization.

---

## Quantization with TensorRT (FP16)

Quantization reduces model size and improves inference speed. This project uses **FP16 (half-precision floating point)** quantization via **NVIDIA TensorRT**:

- **FP16** uses 16 bits instead of 32, cutting memory and bandwidth in half
- This is ideal for inference workloads on GPUs that support fast FP16 operations
- Both ONNX-exported models are quantized using `trtexec` into `.plan` format

These `.plan` files are then loaded and served by NVIDIA Triton Inference Server.

---

## Architecture Overview

- **Inference**: Quantized models served via Triton Inference Server
- **Vector Search**: FAISS index over combined [text | image] embeddings
- **Metadata Store**: MongoDB holds product info and logs
- **Frontend**: Streamlit app for interactive querying

---

## Setup Instructions

### Step 1: Quantize Models & Build Index

Run this once on a machine with a GPU:

```bash
bash setup.sh
```

This will:
- Export both models to ONNX
- Quantize to FP16 via TensorRT
- Build a FAISS index from product embeddings
- Prepare the `model_repository/` directory for Triton

### Step 2: Launch All Services

Option 1 — via helper script:

```bash
bash restart_services.sh
```

Option 2 — fully containerized using Docker Compose:

```bash
docker-compose up --build
```

This will:
- Start MongoDB
- Start Triton Inference Server
- Start the Streamlit web app (http://localhost:8501)

---

## Search Modes

- **Image Only**: Embeds image and searches using visual features
- **Text Only**: Uses the `[CLS]` token to embed the input and search
- **Image + Text**: Embeddings are fused (weighted concat + normalization)

Fusion weights can be configured in `streamlit_app.py`.

---

## When to Re-run `setup.sh`

Only run `setup.sh` again if you:
- Change the model backbone
- Add new product data (images or metadata)
- Want to adjust quantization or embedding logic

---

## Testing

Basic unit tests are included. To run:

```bash
pip install pytest
pytest
```

This covers schema validation, FAISS matching, Triton client mocking, and preprocessing.

---

## Technologies Used

- Triton Inference Server (model deployment)
- TensorRT (model quantization to FP16)
- FAISS (vector similarity search)
- PyTorch / HuggingFace Transformers
- Streamlit (web UI)
- MongoDB (metadata and logging)
- Docker (deployment)

