import streamlit as st
from app.database import load_products_from_csv
from app.matcher import ProductMatcher
from app.utils import read_image, preprocess_image
from app.triton_client import TritonClient  
from app.logger import MongoLogger
from transformers import AutoTokenizer
from sklearn.preprocessing import normalize
from pymongo import MongoClient
import numpy as np
import time

st.set_page_config(page_title="Multimodal Matcher (Multimodal)", layout="wide")
st.title("Multimodal Product Search (Image, Text or Both)")

products = load_products_from_csv("data/products.csv")
matcher = ProductMatcher("data/faiss_index_multimodal.index")
logger = MongoLogger(model_version="multimodal-clip-minilm")

client = TritonClient("localhost:8001") 
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

uploaded_file = st.file_uploader("Upload a product image:", type=["jpg", "jpeg", "png"])
query_text = st.text_input("Or describe the product:", "")

def build_query_embedding(image=None, text=None):
    vision_embed = np.zeros((1, 512), dtype=np.float32)
    text_embed = np.zeros((1, 384), dtype=np.float32)  # MiniLM dimension

    if image:
        img_tensor = preprocess_image(image)
        vision_embed = client.infer("vision_encoder", img_tensor, input_name="pixel_values", output_name="embedding")

    if text:
        tokens = tokenizer([text], return_tensors="np", padding="max_length", truncation=True, max_length=32)
        input_ids = tokens["input_ids"].astype(np.int32)
        text_embed = client.infer("text_encoder", input_ids, input_name="input_ids", output_name="last_hidden_state")
        #text_embed = np.mean(text_embed, axis=1)  # Pool text embeddings
        text_embed = text_embed[:, 0, :]  #  CLS token only
    if text and image:
        alpha, beta = 1.0, 0.5
    elif text:
        alpha, beta = 1.0, 0.0
    elif image:
        alpha, beta = 0.0, 1.0
    fused = np.concatenate([
    alpha * text_embed, beta * vision_embed], axis=1)

    return normalize(fused, axis=1)    


if uploaded_file or query_text:
    st.subheader("Top Matches")
    start_time = time.time()

    if uploaded_file:
        image = read_image(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
    else:
        image = None

    embedding = build_query_embedding(image=image, text=query_text)
    indices, scores = matcher.match(embedding)
    latency = time.time() - start_time
    score_stats = {"mean": float(scores.mean()), "std": float(scores.std())}

    for idx in indices[:5]:  # Show top K matches
        product = products[idx]
        st.image(product["image_path"], width=150)
        st.markdown(f"**{product['title']}**")
        st.markdown(f"*{product['category']}* — ${product['price']}")
        st.markdown("---")

    logger.log(
        input_type="image+text" if uploaded_file and query_text else "image" if uploaded_file else "text",
        query=query_text or "image_upload",
        result_ids=indices.tolist(),
        latency=latency,
        score_stats=score_stats
    )