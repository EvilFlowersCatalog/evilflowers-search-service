import os
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from elasticsearch import AsyncElasticsearch, NotFoundError
from elasticsearch.helpers import async_bulk

logger = logging.getLogger(__name__)


class ElasticsearchClient:
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
        self.host = host or os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
        self.index_name = index_name or os.getenv("ELASTICSEARCH_INDEX", "evilflowers")
        self.username = username if username is not None else os.getenv("ELASTICSEARCH_USERNAME")
        self.password = password if password is not None else os.getenv("ELASTICSEARCH_PASSWORD")
        self.api_key = api_key if api_key is not None else os.getenv("ELASTICSEARCH_API_KEY")
        self.verify_certs = (
            verify_certs
            if verify_certs is not None
            else (os.getenv("ELASTICSEARCH_VERIFY_CERTS", "true").lower() in ("1", "true", "yes", "y"))
        )
        self.ca_certs = ca_certs if ca_certs is not None else os.getenv("ELASTICSEARCH_CA_CERTS")
        self.request_timeout = (
            request_timeout
            if request_timeout is not None
            else float(os.getenv("ELASTICSEARCH_REQUEST_TIMEOUT", "30"))
        )
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

        basic_auth = None
        if self.username and self.password:
            basic_auth = (self.username, self.password)

        self.client = AsyncElasticsearch(
            hosts=[self.host],
            basic_auth=basic_auth,
            api_key=self.api_key,
            verify_certs=self.verify_certs,
            ca_certs=self.ca_certs,
            request_timeout=self.request_timeout,
        )

    def default_mapping(self) -> Dict[str, Any]:
        return {
            "dynamic": True,
            "properties": {
                "document_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "text": {"type": "text"},
                "title": {"type": "text"},
                "page": {"type": "integer"},
                "metadata": {"type": "object", "dynamic": True},
                "created_at": {"type": "date"},
            },
        }

    async def check_connection(self) -> bool:
        try:
            return bool(await self.client.ping())
        except Exception as e:
            logger.warning("Elasticsearch ping failed: %s", e)
            return False

    async def ensure_index(
        self,
        mapping: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        exists = await self.client.indices.exists(index=self.index_name)
        if exists:
            return

        body: Dict[str, Any] = {}
        body["mappings"] = mapping if mapping is not None else self.default_mapping()
        if settings is not None:
            body["settings"] = settings

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
        await self.ensure_index()

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

        doc_id = f"{document_id}:{chunk_id}"

        resp = await self.client.index(
            index=self.index_name,
            id=doc_id,
            document=source,
            refresh="wait_for" if refresh else False,
        )
        return {"indexed": True, "id": resp.get("_id", doc_id), "result": resp.get("result")}

    def _normalize_chunks(self, chunks: Any) -> List[Dict[str, Any]]:
        if chunks is None:
            return []

        if isinstance(chunks, list):
            out: List[Dict[str, Any]] = []
            for item in chunks:
                if isinstance(item, dict):
                    out.append(item)
            return out

        if isinstance(chunks, dict):
            out2: List[Dict[str, Any]] = []
            for k, v in chunks.items():
                if isinstance(v, dict):
                    if "chunk_id" not in v:
                        v = {**v, "chunk_id": str(k)}
                    out2.append(v)
                else:
                    out2.append({"chunk_id": str(k), "text": str(v)})
            return out2

        return []

    async def bulk_index_chunks(
        self,
        document_id: str,
        chunks: Any,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        await self.ensure_index()

        items = self._normalize_chunks(chunks)
        if not items:
            return {"indexed": 0, "errors": 0}

        actions = []
        for item in items:
            chunk_id = str(item.get("chunk_id") or item.get("id") or item.get("chunkId") or "")
            if not chunk_id:
                continue

            text = item.get("text") or item.get("chunk_text") or item.get("content")
            if text is None:
                continue

            title = item.get("title")
            page = item.get("page")
            metadata = item.get("metadata")

            source: Dict[str, Any] = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "text": str(text),
            }
            if title is not None:
                source["title"] = title
            if page is not None:
                try:
                    source["page"] = int(page)
                except Exception:
                    pass
            if metadata is not None and isinstance(metadata, dict):
                source["metadata"] = metadata

            doc_id = f"{document_id}:{chunk_id}"

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

        success, errors = await async_bulk(
            client=self.client,
            actions=actions,
            refresh="wait_for" if refresh else False,
        )

        err_count = 0
        if errors:
            err_count = len(errors)

        return {"indexed": int(success), "errors": int(err_count)}

    async def delete_document(self, document_id: str, refresh: bool = False) -> Dict[str, Any]:
        await self.ensure_index()

        resp = await self.client.delete_by_query(
            index=self.index_name,
            query={"term": {"document_id": document_id}},
            refresh="wait_for" if refresh else False,
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
        await self.ensure_index()

        q = (query or "").strip()

        filters: List[Dict[str, Any]] = []
        if document_id:
            filters.append({"term": {"document_id": document_id}})

        if q:
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
            query_dsl = {"bool": {"must": [{"match_all": {}}], "filter": filters}}

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

        resp = await self.client.search(
            index=self.index_name,
            query=query_dsl,
            size=size,
            from_=from_,
            highlight=highlight,
            source=True,
        )

        hits = (resp or {}).get("hits", {}).get("hits", []) or []
        results: List[Dict[str, Any]] = []

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
        await self.ensure_index()
        doc_id = f"{document_id}:{chunk_id}"
        try:
            resp = await self.client.get(index=self.index_name, id=doc_id)
        except NotFoundError:
            return None
        src = (resp or {}).get("_source") or None
        return src

    async def close(self) -> None:
        await self.client.close()
