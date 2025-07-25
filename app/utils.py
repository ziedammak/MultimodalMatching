from .triton_client import TritonClient  
from PIL import Image
import numpy as np
from torchvision import transforms

def read_image(uploaded_file):
    return Image.open(uploaded_file)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)
    return img_tensor