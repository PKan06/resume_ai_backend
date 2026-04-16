# server/rag/rag_chain.py

from server.llm.langchain_llm import LangchainLLM


class RagChain:
    """
    🔥 Simplified RAG Chain

    - No retriever
    - No prompt templates
    - Only LLM interface

    Reason:
    Retrieval + prompt building is handled in assistant.py
    """

    def __init__(self, assistant_name, person_name):

        self.assistant_name = assistant_name
        self.person_name = person_name

        # 🔥 Only LLM
        self.llm = LangchainLLM()

    # =========================
    # STREAM
    # =========================
    def stream(self, prompt, config=None):

        return self.llm.astream(
            prompt,
            config={
                "run_name": config["run_name"] if config else "rag_stream",
                "metadata": {"component": "rag_chain"},
            },
        )

    # =========================
    # NON-STREAM
    # =========================
    def invoke(self, prompt, config=None):

        return self.llm.ainvoke(
            prompt,
            config={
                "run_name": config["run_name"] if config else "rag_invoke",
                "metadata": {"component": "rag_chain"},
            },
        )