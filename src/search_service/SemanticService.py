import logging

from search_service.EmbeddingGenerator import EmbeddingGenerator
from search_service.MilvusClient import MilvusClient
from config.Config import Config

logger = logging.getLogger(__name__)

# service that manages the indexing and searching documents in milvus DB, uses milvus clent to communicate
class SemanticService:    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.milvus_manager = MilvusClient()
    
    def index_document(
        self,
        document_id: str,
        chunks: dict[list]
    ) -> dict:
        # Generate embeddings
        embeddings = self._generate_embeddings(chunks, document_id)

        # Store in Milvus
        chunk_list = chunks
        metadata = [chunk['metadata'] for chunk in chunk_list]
        return self._store_embeddings(document_id, embeddings, metadata)

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_id: str | None = None,
        page_num: str | None = None
    ) -> list[dict]:
        """
        Semantic search across indexed documents.
        
        Args:
            query: Search query in any supported language
            top_k: Number of results
            document_id: Filter by document (optional)
            page_num: Filter by page (optional)
            
        Returns:
            List of search results
        """
        logger.info(f"Searching: '{query[:50]}...'")
        
        # Generate query embedding
        try:
            query_embedding = self.embedding_generator.generate_single_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []
        
        # Search in Milvus
        try:
            results = self.milvus_manager.search(
                query_embedding=query_embedding,
                top_k=top_k,
                document_id=document_id,
                page_num=page_num
            )
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_document(self, document_id: str) -> dict:
        """Delete document from index"""
        try:
            deleted_count = self.milvus_manager.delete_by_document_id(document_id)
            
            return {
                "document_id": document_id,
                "success": True,
                "chunks_deleted": deleted_count
            }
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return {
                "document_id": document_id,
                "success": False,
                "error": str(e)
            }
        
    def _generate_embeddings(self, chunks, document_id=None):
        try:
            embeddings = self.embedding_generator.generate_embeddings(
                chunks,
                normalize=True,
                show_progress=True
            )
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return {
                "document_id": document_id,
                "success": False,
                "chunks_indexed": 0,
                "error": str(e)
            }
        
        return embeddings
    
    def _store_embeddings(self, document_id, embeddings, metadata):
        try:
            inserted_ids = self.milvus_manager.insert_embeddings(
                document_id=document_id,
                embeddings=embeddings,
                metadata=metadata
            )
                        
            return {
                "document_id": document_id,
                "success": True,
                "chunks_indexed": len(inserted_ids),
                "embedding_dim": Config.EMBEDDING_DIM
            }
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            return {
                "document_id": document_id,
                "success": False,
                "chunks_indexed": 0,
                "error": str(e)
            }
