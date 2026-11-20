# server/rag/normalizer.py

import asyncio
import logging

from langsmith import traceable

logger = logging.getLogger(__name__)


class Normalizer:
    """
    🧹 Normalizer

    raw user question -> clean, grammatical sentence (same meaning).
    Used so that the same search question hits the cache / router.
    Kept independent (no cache, no side effects) so it can be evaluated on its own.
    """

    def __init__(self, llm):
        self.llm = llm

    @traceable(name="normalize_query", run_type="llm")
    async def normalize(self, question: str) -> str:

        prompt = f"""
                You are a text normalization system.

                Task:
                Rewrite the user query into a clean, grammatically correct sentence.

                Rules:
                - ALWAYS return a valid sentence
                - NEVER return empty
                - If no correction needed → return EXACT SAME sentence
                - Output MUST be plain text (no quotes, no explanation)

                User Query:
                {question}

                Final Answer:
            """

        response = await self.llm.ainvoke(prompt)

        if response == "__RATE_LIMIT__":
            raise Exception("RATE_LIMIT")

        if not response or response.strip() == "":
            return question

        return response.strip()


# =========================
# STANDALONE CHECK
# =========================
if __name__ == "__main__":
    # Usage (from backend/):  python -m server.rag.normalizer "your messy question" "another one"
    import io
    import sys
    from server.llm.langchain_llm import LangchainLLM

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console + emojis
    logging.basicConfig(level=logging.INFO)

    questions = sys.argv[1:] or [
        "wat r his backnd skils",
        "tell me abt his projcts",
        "What is his work experience?",  # already clean -> should come back unchanged
    ]
    normalizer = Normalizer(LangchainLLM())

    async def main():
        for q in questions:
            print(f"\nRaw:        {q}\nNormalized: {await normalizer.normalize(q)}")

    asyncio.run(main())
