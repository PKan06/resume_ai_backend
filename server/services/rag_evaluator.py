# server/services/rag_evaluator.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class RAGEvaluator:

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self._cache = {}

    # =========================
    # FAST EMBEDDING (CACHED)
    # =========================
    def _embed(self, text: str):
        """
        BUG FIX: Added strict type check. Previously received tuples from
        get_relevant_documents() — str(tuple) was being embedded, producing
        garbage vectors and cosine scores collapsing to ~1.5.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return None

        key = text[:200]  # cache key (cap length)
        if key in self._cache:
            return self._cache[key]

        vec = self.embedding_model.embed_query(text)
        arr = np.array(vec, dtype=np.float32).reshape(1, -1)
        self._cache[key] = arr
        return arr

    # =========================
    # BATCH EMBED (SAFE)
    # =========================
    def _batch_embed(self, texts: list) -> np.ndarray | None:
        """
        Returns stacked ndarray of embeddings, or None if all fail.
        Filters out None results before stacking (avoids np.vstack crash).
        """
        vecs = [self._embed(t) for t in texts if isinstance(t, str) and t.strip()]
        valid = [v for v in vecs if v is not None]
        if not valid:
            return None
        return np.vstack(valid)

    # =========================
    # EVALUATE
    # =========================
    def evaluate(self, data: dict) -> dict:
        try:
            user_query: str = data.get("user_query", "")
            retrieval_queries: list = data.get("retrieval_queries", [])
            answer: str = data.get("answer", "")
            context_chunks: list = data.get("context_chunks", [])

            # 🔥 Guard: skip eval for non-RAG query types
            if not context_chunks or not retrieval_queries:
                raise ValueError("Missing context or queries — skipping eval")

            # 🔥 Skip dummy queries from non-professional routes
            real_queries = [q for q in retrieval_queries if q.strip() and q != "none"]
            if not real_queries:
                raise ValueError("No real retrieval queries — skipping eval")

            # =========================
            # EMBED
            # =========================
            user_vec = self._embed(user_query)
            answer_vec = self._embed(answer)

            if user_vec is None or answer_vec is None:
                raise ValueError("Embedding failed for query or answer")

            query_vecs = self._batch_embed(real_queries)
            chunk_vecs = self._batch_embed(context_chunks)

            if query_vecs is None or chunk_vecs is None:
                raise ValueError("Batch embedding failed")

            # =========================
            # CONTEXT RELEVANCE
            # BUG FIX: Was hardcoded top-2 regardless of chunk count.
            # Now uses ALL chunks with mean-of-max scoring (standard RAG metric).
            # =========================
            sim_matrix = cosine_similarity(query_vecs, chunk_vecs)

            # max similarity per chunk (across all retrieval queries)
            context_scores = np.max(sim_matrix, axis=0)

            # 🔥 Dynamic top-k: use top 50% of chunks, min 2
            n_chunks = len(context_scores)
            top_k = max(2, n_chunks // 2)
            top_k_scores = np.sort(context_scores)[-top_k:]
            context_score = float(np.mean(top_k_scores))

            # =========================
            # ANSWER RELEVANCE
            # Measures: does the answer address what was asked?
            # =========================
            answer_relevance = float(
                cosine_similarity(user_vec, answer_vec)[0][0]
            )

            # =========================
            # FAITHFULNESS
            # Measures: is the answer grounded in retrieved context?
            # =========================
            faithfulness_scores = cosine_similarity(answer_vec, chunk_vecs)[0]
            faithfulness = float(np.max(faithfulness_scores))

            # =========================
            # HALLUCINATION LABEL
            # =========================
            if faithfulness > 0.75:
                hallucination = "LOW"
            elif faithfulness > 0.55:
                hallucination = "MEDIUM"
            else:
                hallucination = "HIGH"

            result = {
                "context_score": round(context_score, 3),
                "answer_relevance": round(answer_relevance, 3),
                "faithfulness": round(faithfulness, 3),
                "hallucination": hallucination,
                "chunks": n_chunks,
                "avg_context_score": round(float(np.mean(context_scores)), 3),
                "top_chunk_score": round(float(np.max(context_scores)), 3),
            }

            logger.info(f"📊 RAG Eval: {result}")
            return result

        except Exception as e:
            logger.error(f"❌ Evaluation failed: {str(e)}")
            result = self._empty()
            result["error"] = str(e)
            return result

    def _empty(self) -> dict:
        return {
            "context_score": 0,
            "answer_relevance": 0,
            "faithfulness": 0,
            "hallucination": "UNKNOWN",
            "chunks": 0,
            "avg_context_score": 0,
            "top_chunk_score": 0,
        }