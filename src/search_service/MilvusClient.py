import numpy as np
import logging
from datetime import datetime
from pymilvus import connections, Collection, CollectionSchema, utility

from config.Config import Config
from .schemas import MILVUS_FIELDS

logger = logging.getLogger(__name__)

class MilvusClient:
    def __init__(self):
        self._collection = None
        self._connect()
        self._setup_collection()

    def _connect(self):
        logger.info(f"Connecting to Milvus at {Config.MILVUS_HOST}:{Config.MILVUS_PORT}")
        connections.connect(alias="default", host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)
        logger.info("Connected to Milvus")

    def _setup_collection(self):
        collection_name = Config.MILVUS_COLLECTION_NAME
        schema = CollectionSchema(
            fields=MILVUS_FIELDS,
            description=f"Document embeddings (max {Config.MILVUS_MAX_TEXT_LENGTH} chars per chunk)"
        )
        if utility.has_collection(collection_name):
            self._collection = Collection(name=collection_name)
        else:
            self._collection = Collection(name=collection_name, schema=schema)
            self._create_index()
        self._collection.load()
        logger.info(f"Collection {collection_name} ready")

    def _create_index(self):
        self._collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 256}}
        )

    def insert_embeddings(self, document_id: str, embeddings: np.ndarray, metadata: list[dict]) -> list[int]:
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match metadata entries")
        if len(embeddings) == 0:
            logger.warning("No embeddings to insert")
            return []

        current_time = datetime.utcnow().isoformat()
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

        insert_result = self._collection.insert(entities)
        self._collection.flush()
        logger.info(f"Inserted {len(insert_result.primary_keys)} embeddings for {document_id}")
        return insert_result.primary_keys

    def search(self, query_embedding: np.ndarray, top_k: int = 10, document_id: str | None = None, page_num: int | None = None) -> list[dict]:
        filter_expr = None
        if document_id and page_num is not None:
            filter_expr = f'document_id == "{document_id}" && source_page == {page_num}'
        elif document_id:
            filter_expr = f'document_id == "{document_id}"'
        elif page_num is not None:
            filter_expr = f'source_page == {page_num}'

        output_fields = ["document_id", "source_page", "section", "chunk_index", "text", "word_count"]

        results = self._collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )

        return [
            {"id": hit.id, "score": hit.score, "distance": hit.distance, **{f: hit.entity.get(f) for f in output_fields}}
            for hits in results for hit in hits
        ]

    def delete_by_document_id(self, document_id: str) -> int:
        result = self._collection.delete(f'document_id == "{document_id}"')
        self._collection.flush()
        logger.info(f"Deleted {result.delete_count} embeddings for {document_id}")
        return result.delete_count

    def get_stats(self) -> dict:
        return {
            "collection_name": Config.MILVUS_COLLECTION_NAME,
            "total_entities": self._collection.num_entities,
            "embedding_dim": Config.EMBEDDING_DIM,
            "index_type": "HNSW",
            "metric_type": "COSINE",
        }

    def close(self):
        if self._collection:
            self._collection.release()
        connections.disconnect("default")
        logger.info("Disconnected from Milvus")