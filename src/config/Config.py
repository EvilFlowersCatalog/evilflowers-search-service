import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Milvus Vector Database
    MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
    MILVUS_PORT = int(os.getenv('MILVUS_PORT', '19530'))
    MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME', 'document_embeddings')
    MILVUS_LITE_DB=os.getenv('MILVUS_LITE_DB', '/app/data/milvus_lite.db')
    
    # Embeddings
    MODEL_NAME = "sentence-transformers/stsb-xlm-r-multilingual"
    EMBEDDING_DIM = 768
    MAX_SEQUENCE_LENGTH = 512
    BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))
    DEVICE = os.getenv('EMBEDDING_DEVICE', 'cpu')  # 'cuda' or 'cpu'
    
    # Text Processing
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '768'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
    
    # Services
    SEARCH_SERVICE_URL = os.getenv('SEARCH_SERVICE_URL', 'http://localhost:8001')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')