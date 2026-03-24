from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import json

from search_service.SemanticService import SemanticService
from search_service.SemanticService import SemanticService
from search_service.ElasticService import ElasticService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EvilFlowers Search Service",
    description="Semantic search service using Milvus and Elasticsearch",
    version="1.0.0"
)

semantic_service: SemanticService | None = None
elastic_service: ElasticService | None = None
semantic_search: SemanticService | None = None


def get_semantic_service():
    global semantic_service
    if semantic_service is None:
        semantic_service = SemanticService()
    return semantic_service


def get_elasticsearch_client():
    global elastic_service
    if elastic_service is None:
        elastic_service = ElasticService()
    return elastic_service


def get_semantic_search():
    global semantic_search
    if semantic_search is None:
        semantic_search = SemanticService(get_semantic_service())
    return semantic_search


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


# @app.on_event("startup")
# async def on_startup():
#     es_client = get_elasticsearch_client()
#     await es_client.ensure_index()


# @app.on_event("shutdown")
# async def on_shutdown():
#     global elasticsearch_client
#     if elasticsearch_client is not None:
#         await elasticsearch_client.close()
#         elasticsearch_client = None

@app.get("/health")
async def health_check():
    try:
        semantic_service = get_semantic_service()

        milvus_stats = semantic_service.milvus_manager.get_stats()

    except Exception as e:
        milvus_stats = {"status": "unhealthy", "error": str(e)}
    

    es_client = get_elasticsearch_client()
    es_health = await es_client.check_connection()
    es_stats = await es_client.stats_overview()


    
    return {
            "status": "healthy",
            "milvus": milvus_stats,
            "elasticsearch": {
                "connected": es_health,
                **es_stats
            },
        }

@app.post("/index")
async def index_document(request: IndexRequest):
    print("INDEXING DOCUMENT:", request.document_id)

    try:
        # Get clients
        es_client = get_elasticsearch_client()
        semantic_service = get_semantic_service()  # Uncomment when enabling Milvus
        
        # Delete existing document first to prevent duplicates
        # await es_client.delete_document(request.document_id, refresh=True)
        # semantic_service.delete_document(request.document_id)  # Uncomment when enabling Milvus
        
        # Index in Elasticsearch
        es_result = await es_client.index_document(
            document_id=request.document_id,
            chunks=request.chunks["chunks"],
            refresh=True
        )
        
        # Index in Milvus (when enabled)
        milvus_result = semantic_service.index_document(
            document_id=request.document_id,
            chunks=request.chunks["chunks"]
        )

        return {
            "document_id": request.document_id,
            "elasticsearch": es_result,
            "milvus": milvus_result,
        }
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        semantic_service = get_semantic_service()
        es_client = get_elasticsearch_client()

        milvus_result = semantic_service.delete_document(document_id)
        es_result = await es_client.delete_document(document_id, refresh=True)

        return {
            "document_id": document_id,
            "milvus": milvus_result,
            "elasticsearch": es_result,
        }
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/semantic")
async def semantic_search_endpoint(request: SemanticSearchRequest):
    try:
        semantic_search = get_semantic_service()

        results = semantic_search.search(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
            page_num=request.page_num
        )

        return {
            "search_type": "semantic",
            "query": request.query,
            "results": results,
            "total_results": len(results),
        }
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/elasticsearch")
async def elasticsearch_search_endpoint(request: ElasticsearchSearchRequest):
    try:
        es_client = get_elasticsearch_client()

        results = await es_client.search_documents(
            query=request.query,
            document_id=request.document_id,
            size=request.top_k
        )

        return {
            "search_type": "elasticsearch",
            "query": request.query,
            "results": results,
            "total_results": len(results),
        }
    except Exception as e:
        logger.error(f"Elasticsearch search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)
