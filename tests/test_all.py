import numpy as np
import pytest
from sklearn.preprocessing import normalize
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# FAISS
import faiss
from app.matcher import ProductMatcher

# Schemas
from app.schemas import Product, LogEntry

# Logger
from app.logger import MongoLogger

# Triton Client (mocked)
from app.triton_client import TritonClient

# Image preprocessing
from app.utils import preprocess_image
from PIL import Image

# --- FAISS Matcher Test ---
def test_matcher_faiss():
    dim = 896
    index = faiss.IndexFlatIP(dim)
    data = normalize(np.random.random((10, dim)).astype("float32"), axis=1)
    index.add(data)
    faiss.write_index(index, "tests/tmp_test.index")
    
    matcher = ProductMatcher("tests/tmp_test.index")
    query = normalize(np.random.random((1, dim)).astype("float32"), axis=1)
    indices, scores = matcher.match(query, top_k=3)
    assert len(indices) == 3
    assert scores[0] >= 0

# --- Schema Validation Test ---
def test_product_schema():
    p = Product(id=1, title="Test", description="desc", price=10.0, category="cat", image_path="/tmp/x.jpg")
    assert p.title == "Test"

def test_log_entry_schema():
    log = LogEntry(
        input_type="text",
        query="sneaker",
        result_indices=[0, 1],
        latency=0.5,
        model_version="clip-ViT-B-32",
        score_mean=0.92,
        score_std=0.03
    )
    assert log.model_version.startswith("clip")

# --- Mongo Logger Test ---
def test_logger_insert():
    try:
        logger = MongoLogger(model_version="test-version")
        logger.log("text", "test query", [0, 1], 0.123, {"mean": 0.9, "std": 0.05})
        assert True
    except Exception as e:
        assert False, f"Logging failed: {str(e)}"

# --- Triton Client Mocked Test ---
def test_triton_infer_mock(monkeypatch):
    class DummyResponse:
        def as_numpy(self, name): return np.zeros((1, 384), dtype=np.float32)

    class DummyClient:
        def infer(self, **kwargs): return DummyResponse()

    monkeypatch.setattr("app.triton_client.grpcclient.InferenceServerClient", lambda url: DummyClient())
    client = TritonClient()
    result = client.infer("text_encoder", np.zeros((1, 32), dtype=np.int32), "input_ids", "last_hidden_state")
    assert result.shape == (1, 384)

# --- Image Preprocessing Test ---
def test_preprocess_image_shape():
    img = Image.new("RGB", (300, 300))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, 224, 224)
