# server/services/similarity.py
import numpy as np


def cosine_similarity(a, b) -> np.ndarray:
    """Pairwise cosine similarity: (n, d) x (m, d) -> (n, m). Zero vectors score 0."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b = np.atleast_2d(np.asarray(b, dtype=np.float32))

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    a = a / np.where(a_norm == 0, 1, a_norm)
    b = b / np.where(b_norm == 0, 1, b_norm)

    return a @ b.T
