from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging

from search_service.SemanticService import SemanticService
from search_service.SemanticSearch import SemanticSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EvilFlowers Search Service",
    description="Semantic search service using Milvus",
    version="1.0.0"
)

semantic_service: SemanticService | None = None

def get_semantic_service():
    global semantic_service
    if semantic_service is None:
        semantic_service = SemanticService()
    return semantic_service

semantic_search: SemanticSearch | None = None

def get_semantic_search():
    global semantic_search
    if semantic_search is None:
        semantic_search = SemanticSearch(get_semantic_service())
    return semantic_search

class IndexRequest(BaseModel):
    document_id: str
    chunks: dict


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    document_id: Optional[str] = None
    page_num: Optional[int] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        semantic_service = get_semantic_service()

        stats = semantic_service.milvus_manager.get_stats()
        return {
            "status": "healthy",
            "milvus": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/index")
async def index_document(request: IndexRequest):
    """Index document chunks with embeddings"""
    try:
        semantic_service = get_semantic_service()
        result = semantic_service.index_document(
            document_id=request.document_id,
            chunks=request.chunks
        )
        return result
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete document from index"""
    try:
        semantic_service = get_semantic_service()
        result = semantic_service.delete_document(document_id)
        return result
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(request: SearchRequest):
    """Semantic search across documents"""
    try:
        semantic_search = get_semantic_search()
        results = semantic_search.search(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
            page_num=request.page_num
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
