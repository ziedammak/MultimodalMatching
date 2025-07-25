import numpy as np
import faiss
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from app.triton_client import TritonClient
import os
import logging
from sklearn.preprocessing import normalize


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load product data
df = pd.read_csv("data/products.csv")
products = df.to_dict(orient="records")
logger.info(f"Loaded {len(products)} products")

# Setup
client = TritonClient("localhost:8001")  # Use gRPC port 8001
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Preprocess image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

combined_embeddings = []
failed_products = []

for idx, product in enumerate(products):
    try:
        logger.info(f"Processing product {idx+1}/{len(products)}: {product['title'][:50]}...")
        
        # Vision
        img_path = product["image_path"]
        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            continue
            
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).numpy().astype(np.float32)
        vision_embedding = client.infer("vision_encoder", img_tensor, "pixel_values", "embedding")
        
        # Text
        text = f"{product['title']} {product['description']}"
        tokens = tokenizer([text], return_tensors="np", padding="max_length", truncation=True, max_length=32)
        input_ids = tokens["input_ids"].astype(np.int32)
        text_embedding = client.infer("text_encoder", input_ids, "input_ids", "last_hidden_state")
        
        # Pool text embeddings (average over sequence length)
        text_embedding = text_embedding[:, 0, :]
        
        # Combine text and image with weights
        alpha = 1.0
        beta = 0.5
        fused = np.concatenate([
            alpha * text_embedding, beta * vision_embedding
        ], axis=1)

        fused = normalize(fused, axis=1)
        combined_embeddings.append(fused[0])
        
    except Exception as e:
        logger.error(f"Failed to process product {product['title']}: {str(e)}")
        failed_products.append(product)

# Check if we have any embeddings
if not combined_embeddings:
    raise RuntimeError("No valid embeddings generated. All products failed processing.")

# Build FAISS index
embedding_matrix = normalize(np.vstack(combined_embeddings).astype(np.float32), axis=1)
index = faiss.IndexFlatIP(embedding_matrix.shape[1])
index.add(embedding_matrix)

os.makedirs("data", exist_ok=True)
faiss.write_index(index, "data/faiss_index_multimodal.index")
logger.info(f"Saved FAISS index with {len(combined_embeddings)} embeddings. "
            f"{len(failed_products)} products failed.")

if failed_products:
    logger.warning("Failed products:")
    for p in failed_products:
        logger.warning(f" - {p['title']} (ID: {p.get('id', 'N/A')})")