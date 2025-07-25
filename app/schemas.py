from pydantic import BaseModel
from typing import List

class Product(BaseModel):
    id: int
    title: str
    description: str
    price: float
    category: str
    image_path: str

class LogEntry(BaseModel):
    input_type: str
    query: str
    result_indices: List[int]
    latency: float
    model_version: str
    score_mean: float
    score_std: float