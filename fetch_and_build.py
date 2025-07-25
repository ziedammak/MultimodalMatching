import requests
import os
import pandas as pd
from PIL import Image

def fetch_fake_store_products():
    url = "https://fakestoreapi.com/products"
    response = requests.get(url)
    return response.json()

def download_images(products, save_dir="data/images"):
    os.makedirs(save_dir, exist_ok=True)
    for p in products:
        img_url = p['image']
        img_path = os.path.join(save_dir, f"{p['id']}.jpg")
        if not os.path.exists(img_path):
            img_data = requests.get(img_url).content
            with open(img_path, 'wb') as f:
                f.write(img_data)
        p["image_path"] = img_path
    return products

def save_to_csv(products, save_path="data/products.csv"):
    df = pd.DataFrame(products)
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    products = fetch_fake_store_products()
    products = download_images(products)
    save_to_csv(products)
    print(" Data ready in 'data/products.csv' and 'data/images/'")