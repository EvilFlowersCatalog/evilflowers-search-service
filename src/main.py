from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from functools import lru_cache
from typing import Optional
import logging

from search_service.SemanticService import SemanticService
from search_service.ElasticService import ElasticService
from search_service.schemas import IndexRequest, SemanticSearchRequest, ElasticsearchSearchRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@lru_cache
def get_semantic_service() -> SemanticService:
    return SemanticService()


@lru_cache
def get_elastic_service() -> ElasticService:
    return ElasticService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    semantic = get_semantic_service()
    es = get_elastic_service()
    await es.ensure_index()
    yield
    semantic.milvus_manager.close()
    await es.close()


app = FastAPI(title="EvilFlowers Search Service", lifespan=lifespan)


@app.get("/health")
async def health_check(
    semantic: SemanticService = Depends(get_semantic_service),
    es: ElasticService = Depends(get_elastic_service)
):
    try:
        milvus_stats = semantic.milvus_manager.get_stats()
    except Exception as e:
        milvus_stats = {"status": "unhealthy", "error": str(e)}

    es_health = await es.check_connection()
    es_stats = await es.stats_overview()

    return {
        "status": "healthy",
        "milvus": milvus_stats,
        "elasticsearch": {"connected": es_health, **es_stats},
    }


@app.post("/index")
async def index_document(
    request: IndexRequest,
    semantic: SemanticService = Depends(get_semantic_service),
    es: ElasticService = Depends(get_elastic_service)
):
    logger.info(f"Indexing document: {request.document_id}")
    try:
        await es.delete_document(request.document_id, refresh=True)
        semantic.delete_document(request.document_id)

        es_result = await es.index_document(
            document_id=request.document_id,
            chunks=request.chunks["chunks"],
            refresh=True
        )
        milvus_result = semantic.index_document(
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
async def delete_document(
    document_id: str,
    semantic: SemanticService = Depends(get_semantic_service),
    es: ElasticService = Depends(get_elastic_service)
):
    try:
        milvus_result = semantic.delete_document(document_id)
        es_result = await es.delete_document(document_id, refresh=True)

        return {
            "document_id": document_id,
            "milvus": milvus_result,
            "elasticsearch": es_result,
        }
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/semantic")
async def semantic_search(
    request: SemanticSearchRequest,
    semantic: SemanticService = Depends(get_semantic_service)
):
    try:
        results = semantic.search(
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
async def elasticsearch_search(
    request: ElasticsearchSearchRequest,
    es: ElasticService = Depends(get_elastic_service)
):
    try:
        results = await es.search_documents(
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

@app.get("/documents/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    limit: int = 10,
    es: ElasticService = Depends(get_elastic_service)
):
    results = await es.search_documents(query="", document_id=document_id, size=limit)
    return {"document_id": document_id, "chunks": results}