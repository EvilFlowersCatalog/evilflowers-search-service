import os
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from elasticsearch import AsyncElasticsearch, NotFoundError
from elasticsearch.helpers import async_bulk

logger = logging.getLogger(__name__)


class ElasticSearchClient:
    def __init__(
        self,
        host: Optional[str] = None,
        index_name: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[Union[str, Tuple[str, str]]] = None,
        verify_certs: Optional[bool] = None,
        ca_certs: Optional[str] = None,
        request_timeout: Optional[float] = None,
        search_fields: Optional[Sequence[str]] = None,
    ) -> None:
        # Prefer explicit constructor arguments; fall back to environment variables; then to sane defaults.
        self.host = host or os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
        self.index_name = index_name or os.getenv("ELASTICSEARCH_INDEX", "evilflowers")

        # If username/password are passed explicitly as None, fall back to env; otherwise respect the provided value.
        self.username = username if username is not None else os.getenv("ELASTICSEARCH_USERNAME")
        self.password = password if password is not None else os.getenv("ELASTICSEARCH_PASSWORD")

        # API key can be a string or a tuple; if not passed, read from env.
        self.api_key = api_key if api_key is not None else os.getenv("ELASTICSEARCH_API_KEY")

        # TLS verification default: read env ELASTICSEARCH_VERIFY_CERTS; treat "1/true/yes/y" as True.
        self.verify_certs = (
            verify_certs
            if verify_certs is not None
            else (os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").lower() in ("1", "true", "yes", "y"))
        )

        # Optional CA bundle path for verifying Elasticsearch TLS certificates.
        self.ca_certs = ca_certs if ca_certs is not None else os.getenv("ELASTICSEARCH_CA_CERTS")

        # Request timeout for ES operations (seconds).
        self.request_timeout = (
            request_timeout
            if request_timeout is not None
            else float(os.getenv("ELASTICSEARCH_REQUEST_TIMEOUT", "30"))
        )

        # Which fields should be searched by default in multi_match.
        # If not provided, it reads ELASTICSEARCH_SEARCH_FIELDS and splits by comma.
        self.search_fields = list(
            search_fields
            if search_fields is not None
            else (
                os.getenv(
                    "ELASTICSEARCH_SEARCH_FIELDS",
                    "text,content,chunk_text,title,metadata.*,tags",
                ).split(",")
            )
        )

        # Configure Basic Auth only if both username and password are present.
        basic_auth = None
        if self.username and self.password:
            basic_auth = (self.username, self.password)

        # Create the async Elasticsearch client instance.
        # - hosts=[self.host] allows a single-node URL or a load balancer.
        # - basic_auth and api_key are alternative auth methods; ES will use what is provided.
        # - verify_certs/ca_certs control TLS verification when using https.
        # - request_timeout controls max time per request.
        self.client = AsyncElasticsearch(
            hosts=[self.host],
            basic_auth=basic_auth,
            api_key=self.api_key,
            verify_certs=self.verify_certs,
            ca_certs=self.ca_certs,
            request_timeout=self.request_timeout,
        )

    def default_mapping(self) -> Dict[str, Any]:
        # Default index mapping (schema) if you do not provide one to ensure_index().
        # dynamic=True means unknown fields are accepted and mapped automatically.
        return {
            "dynamic": True,
            "properties": {
                "document_id": {"type": "keyword"},   # exact filtering/grouping (not full-text analyzed)
                "chunk_id": {"type": "keyword"},      # exact chunk lookup
                "text": {"type": "text"},             # full-text searchable (analyzed)
                "title": {"type": "text"},            # full-text searchable
                "page": {"type": "integer"},          # numeric page field
                "metadata": {"type": "object", "dynamic": True},  # arbitrary nested metadata
                "created_at": {"type": "date"},       # reserved for timestamps (not currently written by index_* methods)
            },
        }

    async def check_connection(self) -> bool:
        # Ping Elasticsearch to check connectivity.
        try:
            return bool(await self.client.ping())
        except Exception as e:
            # Any exception here means we could not reach ES or auth/TLS failed.
            logger.warning("Elasticsearch ping failed: %s", e)
            return False

    async def ensure_index(
        self,
        mapping: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Check if the index already exists; if so, do nothing.
        exists = await self.client.indices.exists(index=self.index_name)
        if exists:
            return

        # Build the index creation payload.
        body: Dict[str, Any] = {}
        body["mappings"] = mapping if mapping is not None else self.default_mapping()
        if settings is not None:
            body["settings"] = settings

        # Create the index.
        await self.client.indices.create(index=self.index_name, **body)

    async def index_chunk(
        self,
        document_id: str,
        chunk_id: str,
        text: str,
        title: Optional[str] = None,
        page: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        # Ensure the index exists before inserting anything.
        await self.ensure_index()

        # Build the document content stored in Elasticsearch.
        source: Dict[str, Any] = {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "text": text,
        }
        if title is not None:
            source["title"] = title
        if page is not None:
            source["page"] = page
        if metadata is not None:
            source["metadata"] = metadata

        # Deterministic ES _id: allows overwriting the same chunk if you re-index.
        doc_id = f"{document_id}:{chunk_id}"

        # Index a single document.
        # refresh="wait_for" makes the doc searchable immediately after indexing finishes (slower).
        resp = await self.client.index(
            index=self.index_name,
            id=doc_id,
            document=source,
            refresh="wait_for" if refresh else False,
        )
        return {"indexed": True, "id": resp.get("_id", doc_id), "result": resp.get("result")}

    def _normalize_chunks(self, chunks: Any) -> List[Dict[str, Any]]:
        # Accept multiple input shapes for chunks and normalize them into:
        # List[Dict[str, Any]] where each dict contains at least {"chunk_id": ..., "text": ...} or similar.
        if chunks is None:
            return []

        # If already a list, keep only dict elements.
        out: List[Dict[str, Any]] = []
        for item in chunks:
            if isinstance(item, dict):
                print("DISCT ITEM:", item)
                out.append(item)
        return out


    async def index_document(
        self,
        document_id: str,
        chunks: Any,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        # Ensure the index exists before bulk indexing.
        await self.ensure_index()

        # Normalize chunks to a list of dicts.
        items = self._normalize_chunks(chunks)
        if not items:
            return {"indexed": 0, "errors": 0}

        actions = []
        for item in items:
            # Try multiple key variants to find chunk id.
            chunk_id = item.get("metadata", {}).get("chunk_index")
            if not chunk_id:
                # Skip entries that do not have an id.
                continue

            # Try multiple key variants to find the chunk text content.
            text = item.get("text") or item.get("chunk_text") or item.get("content")
            if text is None:
                # Skip entries that do not have any content.
                continue

            title = item.get("title")
            page = item.get("page")
            metadata = item.get("metadata")

            # Build ES document source.
            source: Dict[str, Any] = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "text": str(text),
            }
            if title is not None:
                source["title"] = title
            if page is not None:
                # Page may come as a string; attempt to coerce to int.
                try:
                    source["page"] = int(page)
                except Exception:
                    # If coercion fails, ignore page rather than failing the whole bulk request.
                    pass
            if metadata is not None and isinstance(metadata, dict):
                source["metadata"] = metadata

            # Deterministic id; bulk "index" op will insert or overwrite.
            doc_id = f"{document_id}:{chunk_id}"

            # async_bulk expects actions in this structure.
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_id": doc_id,
                    "_source": source,
                }
            )

        if not actions:
            return {"indexed": 0, "errors": 0}

        # Perform bulk indexing.
        # success is number of successful actions, errors contains per-item errors (if any).
        success, errors = await async_bulk(
            client=self.client,
            actions=actions,
            refresh="wait_for" if refresh else False,
        )

        # Count errors (structure depends on helper; for most cases a list of error items is returned).
        err_count = 0
        if errors:
            err_count = len(errors)

        return {"indexed": int(success), "errors": int(err_count)}

    async def delete_document(self, document_id: str, refresh: bool = False) -> Dict[str, Any]:
        # Ensure the index exists so delete_by_query does not fail on missing index.
        await self.ensure_index()

        # Delete all chunks belonging to the given document_id.
        # conflicts="proceed" tells ES to continue even if there are version conflicts.
        resp = await self.client.delete_by_query(
            index=self.index_name,
            query={"term": {"document_id": document_id}},
            refresh=True if refresh else False,
            conflicts="proceed",
        )

        return {
            "deleted": int(resp.get("deleted", 0) or 0),
            "took_ms": int(resp.get("took", 0) or 0),
        }

    async def search_documents(
        self,
        query: str,
        document_id: Optional[str] = None,
        size: int = 10,
        from_: int = 0,
    ) -> List[Dict[str, Any]]:
        # Ensure index exists for search.
        await self.ensure_index()

        # Trim query; empty query triggers match_all.
        q = (query or "").strip()

        # Optional exact filter (not full-text) to restrict results to a single document.
        filters: List[Dict[str, Any]] = []
        if document_id:
            filters.append({"term": {"document_id": document_id}})

        if q:
            # Full-text query across multiple fields.
            # type="best_fields" means ES picks the best field match per document.
            # operator="and" requires all terms (stricter than default OR).
            # fuzziness="AUTO" allows small typos for supported text fields.
            query_dsl: Dict[str, Any] = {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": q,
                                "fields": self.search_fields,
                                "type": "best_fields",
                                "operator": "and",
                                "fuzziness": "AUTO",
                            }
                        }
                    ],
                    "filter": filters,
                }
            }
        else:
            # If query is empty, return everything (subject to filters).
            query_dsl = {"bool": {"must": [{"match_all": {}}], "filter": filters}}

        # Configure highlight snippets for UI display.
        highlight = {
            "pre_tags": ["<em>"],
            "post_tags": ["</em>"],
            "fields": {
                "text": {"number_of_fragments": 3, "fragment_size": 150},
                "content": {"number_of_fragments": 3, "fragment_size": 150},
                "chunk_text": {"number_of_fragments": 3, "fragment_size": 150},
                "title": {"number_of_fragments": 1, "fragment_size": 150},
            },
        }

        # Execute the search.
        resp = await self.client.search(
            index=self.index_name,
            query=query_dsl,
            size=size,
            from_=from_,
            highlight=highlight,
            source=True,
        )

        # Extract hits list safely.
        hits = (resp or {}).get("hits", {}).get("hits", []) or []
        results: List[Dict[str, Any]] = []

        # Normalize ES hit structure into a cleaner result format for application usage.
        for h in hits:
            src = h.get("_source") or {}
            hl = h.get("highlight") or {}
            results.append(
                {
                    "id": h.get("_id"),
                    "score": h.get("_score"),
                    "document_id": src.get("document_id"),
                    "chunk_id": src.get("chunk_id"),
                    "source": src,
                    "highlight": hl,
                }
            )

        return results

    async def get_chunk(self, document_id: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        # Ensure index exists before get.
        await self.ensure_index()

        # Deterministic id matches how index_chunk/bulk_index_chunks stored documents.
        doc_id = f"{document_id}:{chunk_id}"
        try:
            resp = await self.client.get(index=self.index_name, id=doc_id)
        except NotFoundError:
            # If the chunk does not exist, return None.
            return None
        src = (resp or {}).get("_source") or None
        return src


    async def stats_overview(self) -> Dict[str, Any]:
        await self.ensure_index()
        count_resp = await self.client.count(index=self.index_name)
        agg_resp = await self.client.search(
            index=self.index_name,
            size=0,
            aggs={
                "unique_documents": {
                    "cardinality": {"field": "document_id", "precision_threshold": 40000}
                }
            },
        )
        stats_resp = await self.client.indices.stats(index=self.index_name, metric=["docs", "store"])
        total_chunks = int((count_resp or {}).get("count", 0) or 0)
        unique_docs = int((((agg_resp or {}).get("aggregations") or {}).get("unique_documents") or {}).get("value", 0) or 0)
        prim = (((stats_resp or {}).get("indices") or {}).get(self.index_name) or {}).get("primaries") or {}
        docs = (prim.get("docs") or {})
        store = (prim.get("store") or {})
        return {
            "index": self.index_name,
            "total_chunks": total_chunks,
            "unique_document_ids": unique_docs,
            "deleted_docs": int(docs.get("deleted", 0) or 0),
            "store_size_bytes": int(store.get("size_in_bytes", 0) or 0),
        }

    async def close(self) -> None:
        # Close underlying HTTP connections and cleanup.
        await self.client.close()
