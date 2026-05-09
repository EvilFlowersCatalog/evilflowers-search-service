import logging
import numpy as np

from search_service.EmbeddingGenerator import EmbeddingGenerator
from search_service.MilvusClient import MilvusClient
from config.Config import Config

logger = logging.getLogger(__name__)


class SemanticService:

    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.milvus_manager = MilvusClient()

    def index_document(self, document_id: str, chunks: list[dict]) -> dict:
        texts = [chunk['metadata']['text'] for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts, normalize=True, show_progress=True)
        metadata = [chunk['metadata'] for chunk in chunks]
        inserted_ids = self.milvus_manager.insert_embeddings(document_id, embeddings, metadata)
        return {
            "document_id": document_id,
            "success": True,
            "chunks_indexed": len(inserted_ids),
            "embedding_dim": Config.EMBEDDING_DIM
        }

    def search(self, query: str, top_k: int = 10, document_id: str | None = None, page_num: int | None = None) -> list[dict]:
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        results = self.milvus_manager.search(query_embedding=query_embedding, top_k=top_k, document_id=document_id, page_num=page_num)
        logger.info(f"Found {len(results)} results for '{query[:50]}'")
        return results

    def delete_document(self, document_id: str) -> dict:
        deleted_count = self.milvus_manager.delete_by_document_id(document_id)
        return {
            "document_id": document_id,
            "success": True,
            "chunks_deleted": deleted_count
        }