# server/rag/vector_store.py
import logging
from collections import OrderedDict
from server.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:

    def __init__(self, documents):
        self.documents = documents
        self.db = None
        self.embeddings = None

    # =========================
    # LAZY BUILD
    # =========================
    def _build_db(self):
        if self.db is not None:
            return

        logger.info("⚡ Building FAISS index (lazy init)...")

        try:
            from server.services.embedding_service import EmbeddingService
            from langchain_community.vectorstores import FAISS

            self.embeddings = EmbeddingService.get_instance()

            self.db = FAISS.from_documents(self.documents, self.embeddings)

            logger.info("✅ FAISS ready")

        except Exception as e:
            logger.error(f"❌ FAISS build failed: {str(e)}")
            raise

    # =========================
    # SAFE RETRIEVER
    # =========================
    def get_retriever(self):
        if self.db is None:
            logger.info("⚡ Lazy building DB via retriever")

        self._build_db()

        return self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.RETRIEVAL_TOP_K}
        )

    # =========================
    # 🔥 MULTI-QUERY + FAISS SCORE BASED RANKING
    # =========================
    def get_relevant_documents(self, queries, top_k=settings.RETRIEVAL_TOP_K):
        self._build_db()

        if not queries:
            return []

        all_results = []

        # =========================
        # 🔥 MULTI QUERY RETRIEVAL
        # =========================
        for q in queries:
            try:
                results = self.db.similarity_search_with_score(q, k=5)

                for doc, score in results:
                    # FAISS L2 → convert to similarity
                    similarity = 1 / (1 + score)
                    all_results.append((doc.page_content, similarity))

            except Exception as e:
                logger.error(f"❌ Retrieval failed for query '{q}': {str(e)}")

        if not all_results:
            logger.warning("⚠️ No retrieval results")
            return []

        # =========================
        # 🔥 DEDUP (ORDER PRESERVED)
        # =========================
        unique_map = OrderedDict()

        for text, score in all_results:
            if text not in unique_map:
                unique_map[text] = score
            else:
                # keep best score
                unique_map[text] = max(unique_map[text], score)

        # =========================
        # 🔥 SORT BY SCORE
        # =========================
        ranked = sorted(
            unique_map.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # =========================
        # 🔥 OPTIONAL THRESHOLD FILTER
        # =========================
        threshold = getattr(settings, "RAG_SCORE_THRESHOLD", 0.35)

        filtered = [
            (text, score)
            for text, score in ranked
            if score >= threshold
        ]

        # fallback if too aggressive filtering
        if not filtered:
            logger.warning("⚠️ Threshold removed all chunks → fallback to top results")
            filtered = ranked[:top_k]

        # =========================
        # 🔥 FINAL SELECTION
        # =========================
        final_chunks = [text for text, _ in filtered[:top_k]]

        logger.info(
            f"🎯 Selected chunks: {len(final_chunks)} | "
            f"Top scores: {[round(s,3) for _,s in ranked[:5]]}"
        )

        return final_chunks