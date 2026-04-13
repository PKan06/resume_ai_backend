# server/rag/vector_store.py
from langchain_community.vectorstores import FAISS
from server.services.embedding_service import EmbeddingService
from server.config import settings
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# 🔥 FAISS uses L2 distance (lower = more similar).
# A score of 0.0 = perfect match. Scores above this threshold are weak matches.
FAISS_SCORE_THRESHOLD = 0.75  # tune this: lower = stricter filtering


class VectorStoreManager:

    def __init__(self, documents):
        self.documents = documents
        self.embeddings = EmbeddingService.get_instance()
        self.db = FAISS.from_documents(self.documents, self.embeddings)

    def get_retriever(self):
        if self.db is None:
            raise ValueError("Vector DB not initialized")
        return self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.RETRIEVAL_TOP_K}
        )
        
        
    def get_relevant_documents(self, query: str):

        results = self.db.similarity_search_with_score(
            query,
            k=5
        )

        # 🔥 NO RE-EMBEDDING
        # FAISS already gives distance → convert to similarity

        scored = []

        for doc, score in results:
            similarity = 1 / (1 + score)  # normalize L2 → similarity
            scored.append((doc.page_content, similarity))

        ranked = sorted(scored, key=lambda x: x[1], reverse=True)

        top_k = 3
        filtered = [text for text, _ in ranked[:top_k]]

        logger.debug(f"🎯 Scores: {[round(s,3) for _,s in ranked]}")

        return filtered