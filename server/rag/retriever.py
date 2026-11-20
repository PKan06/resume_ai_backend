# server/rag/retriever.py

import asyncio
import logging

logger = logging.getLogger(__name__)


class Retriever:
    """
    🔍 Retriever

    queries -> relevant resume chunks.
    Kept independent of routing/generation so it can be evaluated on its own.
    """

    def __init__(self, documents):
        self.documents = documents

        # 🔥 LAZY VECTOR STORE — built on first query
        self._vector_manager = None

    def _get_vector_manager(self):
        if self._vector_manager is None:
            from server.rag.helper.vector_store import VectorStoreManager
            self._vector_manager = VectorStoreManager(self.documents)

        return self._vector_manager

    def retrieve(self, queries: list[str]) -> list[str]:
        try:
            return self._get_vector_manager().get_relevant_documents(queries)

        except Exception as e:
            logger.error(f"❌ Retrieval error: {str(e)}")
            return []

    async def aretrieve(self, queries: list[str]) -> tuple[str, list[str]]:
        loop = asyncio.get_running_loop()

        try:
            chunks = await loop.run_in_executor(None, self.retrieve, queries)
        except Exception as e:
            logger.error(f"❌ Async retrieval crash: {str(e)}")
            return "", []

        if not chunks:
            logger.warning("⚠️ No chunks retrieved")

        return "\n\n".join(chunks), chunks


# =========================
# STANDALONE CHECK
# =========================
if __name__ == "__main__":
    # Usage (from backend/):  python -m server.rag.retriever "query 1" "query 2"
    import io
    import sys
    from server.config import settings
    from server.rag.helper.document_processor import DocumentProcessor

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console + emojis
    logging.basicConfig(level=logging.INFO)

    queries = sys.argv[1:] or ["what are his top 5 technical skills", "what project have he worked on?"]
    documents = DocumentProcessor(settings.PDF_PATH).load_and_process()
    retriever = Retriever(documents)

    context_text, chunks = asyncio.run(retriever.aretrieve(queries))

    print(f"\nQueries: {queries}\nRetrieved {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- [{i}] ---\n{chunk}\n")
