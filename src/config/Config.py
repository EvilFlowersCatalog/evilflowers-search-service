import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Milvus Vector Database
    MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
    MILVUS_PORT = int(os.getenv('MILVUS_PORT', '19530'))
    MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME', 'document_embeddings')

    MILVUS_HOST = os.getenv('MILVUS_HOST')
    MILVUS_PORT = int(os.getenv('MILVUS_PORT'))
    MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME')
    MILVUS_MAX_TEXT_LENGTH = int(os.getenv('MILVUS_MAX_TEXT_LENGTH'))
    
    # Elasticsearch
    ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST', 'http://localhost:9200')
    ELASTICSEARCH_INDEX = os.getenv('ELASTICSEARCH_INDEX', 'evilflowers')
    ELASTICSEARCH_USERNAME = os.getenv('ELASTICSEARCH_USERNAME')
    ELASTICSEARCH_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD')
    ELASTICSEARCH_API_KEY = os.getenv('ELASTICSEARCH_API_KEY')
    ELASTICSEARCH_VERIFY_CERTS = os.getenv('ELASTICSEARCH_VERIFY_CERTS', 'true').lower() in ('1', 'true', 'yes', 'y')
    ELASTICSEARCH_CA_CERTS = os.getenv('ELASTICSEARCH_CA_CERTS')
    ELASTICSEARCH_REQUEST_TIMEOUT = float(os.getenv('ELASTICSEARCH_REQUEST_TIMEOUT', '30'))
    ELASTICSEARCH_SEARCH_FIELDS = os.getenv('ELASTICSEARCH_SEARCH_FIELDS', 'text,content,chunk_text,title,metadata.*,tags').split(',')
    
    # Embeddings
    MODEL_NAME = "sentence-transformers/stsb-xlm-r-multilingual"
    EMBEDDING_DIM = 768
    MAX_SEQUENCE_LENGTH = 512
    BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))
    DEVICE = os.getenv('EMBEDDING_DEVICE', 'cpu')
    
    # Text Processing
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '768'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
    
    # Services
    SEARCH_SERVICE_URL = os.getenv('SEARCH_SERVICE_URL', 'http://localhost:8001')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')