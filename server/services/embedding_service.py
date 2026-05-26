from __future__ import annotations

import logging
import threading
import time

import httpx
from langchain_core.embeddings import Embeddings

from server.config import settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.5      # retry waits: 1.5s, 2.25s, 3.375s
_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 30.0


class EmbeddingError(RuntimeError):
    """Raised when Cloudflare Workers AI returns an error or success=false."""


class CloudflareEmbeddings(Embeddings):
    """
    Langchain-compatible embedding client backed by Cloudflare Workers AI.

    Implements embed_query / embed_documents so FAISS.from_documents and all
    existing callers (SemanticCache, RAGEvaluator) work without modification.

    The singleton httpx.Client is thread-safe and reused across calls.
    A threading.Event cancel token can be passed per-call so that an
    in-flight retry loop is aborted immediately when a request is cancelled.
    """

    def __init__(self, account_id: str, api_token: str, model: str) -> None:
        self._model = model
        self._url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/run/{model}"
        )
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=_CONNECT_TIMEOUT,
                read=_READ_TIMEOUT,
                write=10.0,
                pool=5.0,
            ),
            headers=self._headers,
        )

    # ------------------------------------------------------------------
    # Internal: HTTP POST with retry + cancel-token support
    # ------------------------------------------------------------------
    def _post_with_retry(
        self,
        texts: list[str],
        cancel: threading.Event | None = None,
    ) -> list[list[float]]:
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            if cancel and cancel.is_set():
                raise EmbeddingError("Embedding request cancelled by caller.")

            try:
                response = self._client.post(self._url, json={"text": texts})

                if 400 <= response.status_code < 500:
                    raise EmbeddingError(
                        f"Cloudflare returned HTTP {response.status_code}: "
                        f"{response.text[:400]}"
                    )

                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                payload = response.json()
                if not payload.get("success", False):
                    raise EmbeddingError(
                        f"Cloudflare embedding failed (success=false): "
                        f"{payload.get('errors', [])}"
                    )

                return payload["result"]["data"]

            except EmbeddingError:
                raise

            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    wait = _BACKOFF_BASE ** attempt
                    logger.warning(
                        "Cloudflare embedding attempt %d/%d failed (%s). "
                        "Retrying in %.1fs...",
                        attempt, _MAX_RETRIES, exc, wait,
                    )
                    # Honour cancel during the backoff sleep; fall back to plain sleep.
                    if cancel:
                        if cancel.wait(timeout=wait):
                            raise EmbeddingError("Embedding request cancelled during retry backoff.")
                    else:
                        time.sleep(wait)
                else:
                    logger.error(
                        "Cloudflare embedding failed after %d attempts: %s",
                        _MAX_RETRIES, exc,
                    )

        raise EmbeddingError(
            f"Cloudflare embedding unreachable after {_MAX_RETRIES} retries"
        ) from last_exc

    # ------------------------------------------------------------------
    # Langchain Embeddings interface — sync
    # ------------------------------------------------------------------
    def embed_query(
        self,
        text: str,
        cancel: threading.Event | None = None,
    ) -> list[float]:
        return self._post_with_retry([text], cancel=cancel)[0]

    def embed_documents(
        self,
        texts: list[str],
        cancel: threading.Event | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[batch_start: batch_start + _BATCH_SIZE]
            logger.debug(
                "Embedding batch %d (%d texts)",
                batch_start // _BATCH_SIZE + 1, len(batch),
            )
            results.extend(self._post_with_retry(batch, cancel=cancel))
        return results

    # ------------------------------------------------------------------
    # Langchain Embeddings interface — async
    # Async callers get true async HTTP; asyncio.CancelledError aborts
    # the underlying connection immediately via AsyncClient context manager.
    # ------------------------------------------------------------------
    async def aembed_query(self, text: str) -> list[float]:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=10.0, pool=5.0
            ),
            headers=self._headers,
        ) as client:
            response = await client.post(self._url, json={"text": [text]})
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success", False):
                raise EmbeddingError(
                    f"Cloudflare async embed_query failed: {payload.get('errors', [])}"
                )
            return payload["result"]["data"][0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=10.0, pool=5.0
            ),
            headers=self._headers,
        ) as client:
            for batch_start in range(0, len(texts), _BATCH_SIZE):
                batch = texts[batch_start: batch_start + _BATCH_SIZE]
                response = await client.post(self._url, json={"text": batch})
                response.raise_for_status()
                payload = response.json()
                if not payload.get("success", False):
                    raise EmbeddingError(
                        f"Cloudflare async embed_documents failed: {payload.get('errors', [])}"
                    )
                results.extend(payload["result"]["data"])
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the underlying sync HTTP client. Called on app shutdown."""
        self._client.close()


# =========================
# SINGLETON SERVICE
# =========================
class EmbeddingService:

    _instance: CloudflareEmbeddings | None = None

    @classmethod
    def get_instance(cls) -> CloudflareEmbeddings:
        if cls._instance is None:
            logger.info("Initializing CloudflareEmbeddings singleton...")
            cls._instance = CloudflareEmbeddings(
                account_id=settings.CLOUDFLARE_ACCOUNT_ID,
                api_token=settings.CLOUDFLARE_API_TOKEN,
                model=settings.CLOUDFLARE_EMBEDDING_MODEL,
            )
            logger.info(
                "CloudflareEmbeddings ready (%s, 384-dim)",
                settings.CLOUDFLARE_EMBEDDING_MODEL,
            )
        return cls._instance
