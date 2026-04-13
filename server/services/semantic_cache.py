# server/services/semantic_cache.py
import json
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Redis-backed Semantic Cache (db=1)

    ✔ TTL expiration (Redis native)
    ✔ LRU eviction (Redis maxmemory-policy)
    ✔ Cosine similarity threshold
    ✔ Hit/miss metrics (in-process)
    ✔ Zero in-process RAM for cache entries
    """

    def __init__(self, embedding_model, redis_client, max_size=500, ttl=300, threshold=0.85):
        self.embedding_model = embedding_model
        self.redis = redis_client
        self.max_size = max_size
        self.ttl = ttl
        self.threshold = threshold

        self.hits = 0
        self.misses = 0

        self.PREFIX = "sem_cache:"

    # =========================
    # INTERNAL: KEY
    # =========================
    def _key(self, query: str) -> str:
        hashed = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"{self.PREFIX}{hashed}"

    # =========================
    # INTERNAL: SAFE EMBEDDING
    # Handles all shapes:
    #   - list[float]       → (768,)   ✅ already flat
    #   - list[list[float]] → (1, 768) ← bge returns this sometimes
    #   - np.ndarray 2D     → (1, 768) ← needs flatten
    # =========================
    def _to_flat_array(self, embedding) -> np.ndarray:
        """
        Always returns a guaranteed 1D float32 ndarray regardless of
        what embed_query() returns (list, nested list, 2D array, tuple).
        """
        arr = np.array(embedding, dtype=np.float32)

        # 🔥 KEY FIX: flatten any shape → always 1D
        arr = arr.flatten()

        if arr.size == 0:
            raise ValueError("Embedding is empty after flatten")

        return arr

    # =========================
    # INTERNAL: SERIALIZE / DESERIALIZE
    # =========================
    def _serialize(self, value: dict, embedding: np.ndarray) -> str:
        # embedding is guaranteed 1D here — tolist() produces List[float]
        return json.dumps({
            "value": value,
            "embedding": embedding.tolist()   # ✅ always List[float] now
        })

    def _deserialize(self, raw) -> dict:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    # =========================
    # INTERNAL: FETCH ALL ENTRIES
    # =========================
    def _fetch_all(self):
        """
        scan_iter — non-blocking, returns individual keys (never tuples).
        Returns three parallel lists: keys, embeddings, values.
        """
        valid_keys = []
        valid_embeddings = []
        valid_values = []

        for key in self.redis.scan_iter(f"{self.PREFIX}*"):

            if isinstance(key, bytes):
                key = key.decode("utf-8")

            raw = self.redis.get(key)
            if raw is None:
                continue

            try:
                entry = self._deserialize(raw)

                # 🔥 re-flatten on load too — defensive against old malformed entries
                emb = self._to_flat_array(entry["embedding"])

                valid_keys.append(key)
                valid_embeddings.append(emb)
                valid_values.append(entry["value"])

            except Exception as e:
                logger.warning(f"⚠️ Skipped malformed cache entry {key}: {e}")
                continue

        return valid_keys, valid_embeddings, valid_values

    # =========================
    # LOOKUP
    # =========================
    def lookup(self, query: str):
        try:
            valid_keys, valid_embeddings, valid_values = self._fetch_all()

            if not valid_embeddings:
                self.misses += 1
                return None

            query_vec_raw = self.embedding_model.embed_query(query)
            if not query_vec_raw:
                self.misses += 1
                return None

            # 🔥 flatten query vec too — same issue applies here
            query_vec = self._to_flat_array(query_vec_raw).reshape(1, -1)

            embeddings_matrix = np.vstack([e.reshape(1, -1) for e in valid_embeddings])

            scores = cosine_similarity(query_vec, embeddings_matrix)[0]  # guaranteed 1D

            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            if best_score >= self.threshold:
                self.hits += 1
                logger.info(f"⚡ CACHE HIT | score={best_score:.3f} | key={valid_keys[best_idx]}")
                return valid_values[best_idx]

            self.misses += 1
            logger.info(f"❌ CACHE MISS | best_score={best_score:.3f}")
            return None

        except Exception as e:
            logger.error(f"❌ SemanticCache.lookup error: {e}")
            self.misses += 1
            return None

    # =========================
    # STORE
    # =========================
    def store(self, query: str, value: dict):
        try:
            raw_embedding = self.embedding_model.embed_query(query)

            if raw_embedding is None or len(raw_embedding) == 0:
                logger.error("❌ Skipped cache store — empty embedding")
                return

            # 🔥 flatten before serialize — fixes the tuple/2D error
            embedding = self._to_flat_array(raw_embedding)

            key = self._key(query)
            payload = self._serialize(value, embedding)

            self.redis.set(key, payload, ex=self.ttl)
            logger.debug(f"💾 Stored in Redis cache: {key}")

        except Exception as e:
            logger.error(f"❌ SemanticCache.store error: {e}")

    # =========================
    # MEMORY USAGE
    # =========================
    def memory_usage(self) -> float:
        try:
            count = sum(1 for _ in self.redis.scan_iter(f"{self.PREFIX}*"))
            # ~3KB per entry (768-dim float32 = 3KB + JSON overhead)
            return round((count * 3) / 1024, 4)
        except Exception:
            return 0.0

    # =========================
    # STATS
    # =========================
    def stats(self) -> dict:
        total = self.hits + self.misses
        try:
            size = sum(1 for _ in self.redis.scan_iter(f"{self.PREFIX}*"))
        except Exception:
            size = -1

        return {
            "size": size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 2) if total else 0,
            "memory_mb": self.memory_usage()
        }