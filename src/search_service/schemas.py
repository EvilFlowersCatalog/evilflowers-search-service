from pydantic import BaseModel
from typing import Optional
from pymilvus import FieldSchema, DataType

from config.Config import Config


# API schemas
class IndexRequest(BaseModel):
    document_id: str
    chunks: dict


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    document_id: Optional[str] = None
    page_num: Optional[int] = None


class ElasticsearchSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    document_id: Optional[str] = None


# Milvus collection schema
MILVUS_FIELDS = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="source_page", dtype=DataType.INT64),
    FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="chunk_index", dtype=DataType.INT64),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=Config.MILVUS_MAX_TEXT_LENGTH),
    FieldSchema(name="word_count", dtype=DataType.INT64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=Config.EMBEDDING_DIM),
    FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
]


# Elasticsearch index mapping
ES_MAPPING = {
    "dynamic": True,
    "properties": {
        "document_id": {"type": "keyword"},
        "chunk_id": {"type": "keyword"},
        "text": {"type": "text"},
        "title": {"type": "text"},
        "page": {"type": "integer"},
        "metadata": {"type": "object", "dynamic": True},
        "created_at": {"type": "date"},
    },
}