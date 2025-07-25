import os
from pymongo import MongoClient
from pydantic import BaseModel
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LogEntry(BaseModel):
    timestamp: datetime
    model_version: str
    input_type: str  # text,image or image+text
    query: str
    result_ids: list[int]
    latency: float
    score_stats: dict

class MongoLogger:
    def __init__(self, model_version, db_name="product_search", collection_name="logs"):
        self.model_version = model_version
        try:
            self.client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            logger.info("Connected to MongoDB successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self.client = None

    def log(self, input_type, query, result_ids, latency, score_stats):
        if not self.client:
            logger.warning("Skipping log entry - MongoDB not available")
            return
            
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            model_version=self.model_version,
            input_type=input_type,
            query=query,
            result_ids=result_ids,
            latency=latency,
            score_stats=score_stats
        )
        
        try:
            self.collection.insert_one(entry.dict())
        except Exception as e:
            logger.error(f"Failed to log to MongoDB: {str(e)}")