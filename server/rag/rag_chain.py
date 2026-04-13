# server/rag/rag_chain.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from server.llm.langchain_llm import LangchainLLM


class RagChain:

    def __init__(self, retriever, assistant_name, person_name):

        self.retriever = retriever
        self.assistant_name = assistant_name
        self.person_name = person_name

        self.llm = LangchainLLM()

    def _build_chain(self):

        prompt = ChatPromptTemplate.from_template("""
            You are {assistant_name}, the dedicated personal AI assistant representing {person_name} to recruiters, hiring managers, and professional contacts.
            
            Your mission is to present {person_name}'s professional profile with confidence, clarity, and enthusiasm — always in the most favorable and honest light.
            
            ---------------------
            RESUME CONTEXT:
            {context}
            ---------------------
            
            QUESTION:
            {question}
            
            INSTRUCTIONS:
            - Use ONLY the resume context above. Do not invent facts, numbers, or experiences.
            - If the context does not contain enough information to answer fully, say so honestly and share what IS available.
            - Structure your response in three natural parts (no headings — just flowing prose):
            
            1. HOOK: Open with 1–2 confident, compelling sentences that set the stage for your answer. Build curiosity. Do not answer the question yet.
            
            2. ANSWER: Directly and thoroughly answer the question. Be specific — reference real skills, real project names, real achievements from the context. Show alignment between {person_name}'s background and what the question is probing. Use enthusiastic but professional language: words like "demonstrated", "delivered", "achieved", "built", "led" carry weight.
            
            3. CLOSING: End with 1–2 strong sentences that reinforce {person_name}'s value, express confidence in their abilities, and invite the next question or a deeper conversation.
            
            TONE:
            - Confident and proud — you genuinely believe in {person_name}'s abilities
            - Professional and respectful — this is a recruiter-facing context
            - Specific over vague — real details beat generic praise every time
            - Never defensive, never self-deprecating
            
            YOUR RESPONSE:
                """)

        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
                "assistant_name": lambda _: self.assistant_name,
                "person_name": lambda _: self.person_name,
            }
            | prompt
            | LangchainLLM()
        )

        return chain


    def stream(self, question, config = None):

        return self.chain.stream(question,  config={
            "run_name": config["run_name"] if config else "rag_stream",
            "metadata": {
                "component": "rag_chain"
            }
        })
        
        
    def invoke(self, question, config = None):

        return self.chain.invoke(
            question,
            config={
                "run_name": config["run_name"] if config else "rag_invoke",
                "metadata": {
                    "component": "rag_chain"
                }
            }
        )