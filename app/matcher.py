
import numpy as np
import faiss
from sklearn.preprocessing import normalize

class ProductMatcher:
    def __init__(self, faiss_index_path):
        self.index = faiss.read_index(faiss_index_path)

    def match(self, embedding, top_k=5):
        normed = normalize(embedding.reshape(1, -1), axis=1)
        scores, indices = self.index.search(normed, top_k)
        return indices[0], scores[0]
