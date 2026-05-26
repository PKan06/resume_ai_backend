# server/rag/vector_store.py
import logging
from collections import OrderedDict
from server.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:

    def __init__(self, documents):
        self.documents = documents
        self.db = None          # sentinel: None = not ready, True = ready
        self._index = None      # raw Pinecone Index object
        self._embedding_service = None

    # =========================
    # LAZY CONNECT / UPSERT
    # =========================
    def _build_db(self):
        if self.db is not None:
            return

        from pinecone import Pinecone
        from server.services.embedding_service import EmbeddingService

        logger.info("Connecting to Pinecone index '%s'...", settings.PINECONE_INDEX_NAME)

        try:
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self._index = pc.Index(settings.PINECONE_INDEX_NAME)
            self._embedding_service = EmbeddingService.get_instance()

            stats = self._index.describe_index_stats()
            total = stats.get("total_vector_count", 0)
            force = settings.PINECONE_FORCE_REINDEX.lower() == "true"

            if total == 0 or force:
                self._upsert_documents()
            else:
                logger.info("Pinecone has %d vectors — reconnected (no upsert)", total)

            self.db = True

        except Exception as e:
            logger.error("Pinecone init failed: %s", str(e))
            raise

    # =========================
    # FIRST-TIME UPSERT
    # =========================
    def _upsert_documents(self):
        texts = [doc.page_content for doc in self.documents]
        logger.info("Upserting %d chunks to Pinecone...", len(texts))

        vectors_data = self._embedding_service.embed_documents(texts)

        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = [
                {
                    "id": f"doc_{j}",
                    "values": vectors_data[j],
                    "metadata": {"text": texts[j]},
                }
                for j in range(i, min(i + batch_size, len(texts)))
            ]
            self._index.upsert(vectors=batch)

        logger.info("Pinecone upsert complete (%d vectors)", len(texts))

    # =========================
    # MULTI-QUERY RETRIEVAL
    # =========================
    def get_relevant_documents(self, queries, top_k=settings.RETRIEVAL_TOP_K):
        self._build_db()

        if not queries:
            return []

        all_results = []

        for q in queries:
            try:
                query_vector = self._embedding_service.embed_query(q)
                response = self._index.query(
                    vector=query_vector,
                    top_k=5,
                    include_metadata=True,
                )
                for match in response["matches"]:
                    text = match["metadata"]["text"]
                    score = match["score"]   # cosine similarity, 0–1 directly
                    all_results.append((text, score))

            except Exception as e:
                logger.error("Retrieval failed for query '%s': %s", q, str(e))

        if not all_results:
            logger.warning("No retrieval results")
            return []

        # =========================
        # DEDUP (ORDER PRESERVED)
        # =========================
        unique_map = OrderedDict()

        for text, score in all_results:
            if text not in unique_map:
                unique_map[text] = score
            else:
                unique_map[text] = max(unique_map[text], score)

        # =========================
        # SORT BY SCORE
        # =========================
        ranked = sorted(
            unique_map.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # =========================
        # THRESHOLD FILTER
        # =========================
        threshold = getattr(settings, "RAG_SCORE_THRESHOLD", 0.35)

        filtered = [
            (text, score)
            for text, score in ranked
            if score >= threshold
        ]

        if not filtered:
            logger.warning("Threshold removed all chunks — fallback to top results")
            filtered = ranked[:top_k]

        # =========================
        # FINAL SELECTION
        # =========================
        final_chunks = [text for text, _ in filtered[:top_k]]

        logger.info(
            "Selected chunks: %d | Top scores: %s",
            len(final_chunks),
            [round(s, 3) for _, s in ranked[:5]],
        )

        return final_chunks
