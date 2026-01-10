import os
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from typing import List, Dict, Optional
import numpy as np
import logging
from datetime import datetime

from config.Config import Config

logger = logging.getLogger(__name__)


class MilvusManager:
    """
    Manages Milvus vector database operations.
    Singleton pattern for connection reuse.
    """
    
    _instance = None
    _collection = None
    _connected = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._connected:
            self._connect()
            self._setup_collection()
    
    def _connect(self):
        """Connect to Milvus server or use embedded Milvus Lite"""
        
        use_lite = os.getenv('USE_MILVUS_LITE', 'false').lower() == 'true'
        
        if use_lite:
            logger.info("Using Milvus Lite (embedded mode)")
            logger.info(f"USE_MILVUS_LITE env var: {os.getenv('USE_MILVUS_LITE')}")  # ADD THIS
            logger.info(f"DB file path: {os.getenv('MILVUS_LITE_DB', './milvus_lite.db')}")  # ADD THIS
            try:
                from pymilvus import MilvusClient
                db_file = os.getenv('MILVUS_LITE_DB', './milvus_lite.db')
                
                logger.info(f"Attempting to connect to: {db_file}")  # ADD THIS
                self._client = MilvusClient(uri=db_file)
                self._use_lite = True  # ADD THIS
                self._connected = True
                logger.info(f"✓ Connected to Milvus Lite: {db_file}")
                
            except Exception as e:
                logger.error(f"Failed to connect to Milvus Lite: {e}")
                raise
        else:
            try:
                logger.info(f"Connecting to Milvus server at {Config.MILVUS_HOST}:{Config.MILVUS_PORT}")
                
                connections.connect(
                    alias="default",
                    host=Config.MILVUS_HOST,
                    port=Config.MILVUS_PORT
                )
                
                self._use_lite = False  # ADD THIS
                self._connected = True
                logger.info("✓ Connected to Milvus server")
                
            except Exception as e:
                logger.error(f"Failed to connect to Milvus: {e}")
                raise

    def _setup_collection(self):
        """Create or load collection"""
        collection_name = Config.MILVUS_COLLECTION_NAME
        
        if self._use_lite:
            # MilvusClient API (Milvus Lite)
            if self._client.has_collection(collection_name):
                logger.info(f"Collection {collection_name} already exists")
            else:
                logger.info(f"Creating new collection: {collection_name}")
                self._client.create_collection(
                    collection_name=collection_name,
                    dimension=Config.EMBEDDING_DIM,
                    metric_type="COSINE",
                    auto_id=True
                )
            logger.info(f"✓ Collection {collection_name} is ready")
        else:
            # Old Collection API (Milvus Server)
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="source_page", dtype=DataType.INT64),
                FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="word_count", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=Config.EMBEDDING_DIM),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Document embeddings for semantic search"
            )
            
            if utility.has_collection(collection_name):
                logger.info(f"Loading existing collection: {collection_name}")
                self._collection = Collection(name=collection_name)
            else:
                logger.info(f"Creating new collection: {collection_name}")
                self._collection = Collection(
                    name=collection_name,
                    schema=schema
                )
                self._create_index()
            
            self._collection.load()
            logger.info(f"✓ Collection {collection_name} is ready")

    def _create_index(self):
        """Create index for fast similarity search"""
        
        # Check if using Milvus Lite
        use_lite = os.getenv('USE_MILVUS_LITE', 'false').lower() == 'true'
        
        if use_lite:
            # Milvus Lite: Use AUTOINDEX (simpler, automatically optimized)
            index_params = {
                "metric_type": "COSINE",
                "index_type": "AUTOINDEX",
                "params": {}
            }
            logger.info("Creating AUTOINDEX for Milvus Lite...")
        else:
            # Milvus Server: Use HNSW (faster for large datasets)
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {
                    "M": 16,
                    "efConstruction": 256
                }
            }
            logger.info("Creating HNSW index for Milvus Server...")
        
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        logger.info("✓ Index created")
    
    def insert_embeddings(
        self,
        document_id: str,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ) -> List[int]:
        """Insert embeddings with metadata."""
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match metadata entries")
        
        if len(embeddings) == 0:
            logger.warning("No embeddings to insert")
            return []
        
        current_time = datetime.utcnow().isoformat()
        
        if self._use_lite:
            # MilvusClient API - uses dict format
            data = []
            for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
                data.append({
                    "document_id": document_id,
                    "source_page": meta.get("source_page", -1),
                    "section": meta.get("section", "content"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "text": meta.get("text", "")[:2000],
                    "word_count": meta.get("word_count", 0),
                    "vector": emb.tolist(),
                    "created_at": current_time
                })
            
            logger.info(f"Inserting {len(data)} embeddings for document {document_id}")
            result = self._client.insert(
                collection_name=Config.MILVUS_COLLECTION_NAME,
                data=data
            )
            logger.info(f"✓ Inserted {len(result['ids'])} embeddings")
            return result['ids']
        else:
            # Old Collection API
            entities = [
                [document_id] * len(embeddings),
                [m.get("source_page", -1) for m in metadata],
                [m.get("section", "content") for m in metadata],
                [m.get("chunk_index", 0) for m in metadata],
                [m.get("text", "")[:2000] for m in metadata],
                [m.get("word_count", 0) for m in metadata],
                embeddings.tolist(),
                [current_time] * len(embeddings)
            ]
            
            logger.info(f"Inserting {len(embeddings)} embeddings for document {document_id}")
            insert_result = self._collection.insert(entities)
            self._collection.flush()
            logger.info(f"✓ Inserted {len(insert_result.primary_keys)} embeddings")
            return insert_result.primary_keys

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        document_id: Optional[str] = None,
        page_num: Optional[int] = None
    ) -> List[Dict]:
        """Search for similar embeddings."""
        
        if self._use_lite:
            # MilvusClient API
            filter_expr = None
            if document_id and page_num is not None:
                filter_expr = f'document_id == "{document_id}" && source_page == {page_num}'
            elif document_id:
                filter_expr = f'document_id == "{document_id}"'
            elif page_num is not None:
                filter_expr = f'source_page == {page_num}'
            
            logger.debug(f"Searching for top {top_k} results")
            
            results = self._client.search(
                collection_name=Config.MILVUS_COLLECTION_NAME,
                data=[query_embedding.tolist()],
                limit=top_k,
                filter=filter_expr,
                output_fields=["document_id", "source_page", "section", "chunk_index", "text", "word_count"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": hit.get("id"),
                        "score": hit.get("distance"),
                        "distance": hit.get("distance"),
                        "document_id": hit.get("document_id"),
                        "source_page": hit.get("source_page"),
                        "section": hit.get("section"),
                        "chunk_index": hit.get("chunk_index"),
                        "text": hit.get("text"),
                        "word_count": hit.get("word_count")
                    })
            
            logger.debug(f"Found {len(formatted_results)} results")
            return formatted_results
        else:
            # Old Collection API (same as before)
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 128}
            }
            
            filter_expr = None
            if document_id and page_num is not None:
                filter_expr = f'document_id == "{document_id}" && page_num == {page_num}'
            elif document_id:
                filter_expr = f'document_id == "{document_id}"'
            elif page_num is not None:
                filter_expr = f'page_num == {page_num}'
            
            output_fields = [
                "document_id", "source_page", "section",
                "chunk_index", "text", "word_count"
            ]
            
            logger.debug(f"Searching for top {top_k} results")
            
            results = self._collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields
            )
            
            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "score": hit.score,
                        "distance": hit.distance,
                    }
                    for field in output_fields:
                        result[field] = hit.entity.get(field)
                    formatted_results.append(result)
            
            logger.debug(f"Found {len(formatted_results)} results")
            return formatted_results       
    
    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all embeddings for a document"""
        expr = f'document_id == "{document_id}"'
        
        try:
            logger.info(f"Deleting embeddings for document: {document_id}")
            
            if self._use_lite:
                # MilvusClient API
                result = self._client.delete(
                    collection_name=Config.MILVUS_COLLECTION_NAME,
                    filter=expr
                )
                deleted_count = len(result) if isinstance(result, list) else result.get('delete_count', 0)
            else:
                # Old Collection API
                result = self._collection.delete(expr)
                self._collection.flush()
                deleted_count = result.delete_count
            
            logger.info(f"✓ Deleted {deleted_count} embeddings")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        if self._use_lite:
            # MilvusClient API
            stats = self._client.get_collection_stats(Config.MILVUS_COLLECTION_NAME)
            return {
                "collection_name": Config.MILVUS_COLLECTION_NAME,
                "total_entities": stats.get('row_count', 0),
                "embedding_dim": Config.EMBEDDING_DIM
            }
        else:
            # Old Collection API
            return {
                "collection_name": Config.MILVUS_COLLECTION_NAME,
                "total_entities": self._collection.num_entities,
                "embedding_dim": Config.EMBEDDING_DIM
            }
    
    def close(self):
        """Release resources"""
        if self._use_lite:
            # MilvusClient API - close the client
            if hasattr(self, '_client'):
                self._client.close()
        else:
            # Old Collection API
            if self._collection:
                self._collection.release()
            connections.disconnect("default")
        
        self._connected = False
        logger.info("Disconnected from Milvus")

if __name__ == '__main__':
    milvus = MilvusManager()