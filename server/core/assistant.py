# server/core/assistant.py (controller for the assistant logic)
# === ADD TO TOP OF assistant.py imports ===
import logging
import asyncio
import json
import datetime
from collections import OrderedDict
from langsmith import traceable
import redis

from server.services.semantic_cache import SemanticCache
from server.config import settings

from server.rag.document_processor import DocumentProcessor
from server.rag.profile_extractor import ProfileExtractor
from server.rag.rag_chain import RagChain

from server.services.embedding_service import EmbeddingService
from server.services.rag_evaluator import RAGEvaluator


logger = logging.getLogger(__name__)


class FastAPIPersonalAssistant:

    def __init__(self, pdf_path, model_name, assistant_name):

        self.pdf_path = pdf_path
        self.model_name = model_name
        self.assistant_name = assistant_name

        self.documents = None
        self.retriever = None
        self.rag_chain = None

        self.person_name = None
        self.person_profile = {}

        self.routing_cache = OrderedDict()
        self.max_cache_size = 100

        self.routing_embeddings = []
        self.routing_questions = []
        self.routing_results = []

        self.user_state = {
            "last_seen": None,
            "conversation_count": 0,
            "last_intent": None,
        }

        self._initialize_system()

    def _initialize_system(self):

        processor = DocumentProcessor(self.pdf_path)

        # 🔥 RAW docs for profile extraction (unsplit, unenriched)
        raw_documents = processor.load_raw()

        # 🔥 Processed docs for RAG (split + enriched)

        self.documents = processor.load_and_process()

        extractor = ProfileExtractor(raw_documents)
        self.person_name, self.person_profile = extractor.extract()

        # 🔥 LAZY VECTOR STORE (CRITICAL FIX)
        self.vector_manager = None

        # retriever will be created dynamically when needed
        self.retriever = None

        self.rag_chain = RagChain(self.assistant_name, self.person_name)

        # 🔥 embedding model
        self.embedding_model = EmbeddingService.get_instance()

        # 🔥 Redis client (Upstash — no persistent connection, HTTP-based)
        if settings.REDIS_ENABLED:
            try:
                redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    password=settings.REDIS_PASSWORD,
                    db=0,
                    decode_responses=True,
                    ssl=True,
                )

                self.semantic_cache = SemanticCache(
                    embedding_model=self.embedding_model,
                    redis_client=redis_client,
                    max_size=500,
                    ttl=300,
                    threshold=0.85,
                )
                logger.info("✅ Semantic cache initialized")

            except Exception as e:
                logger.error(f"❌ Redis init failed: {e}")
                self.semantic_cache = None
        else:
            logger.info("ℹ️ Semantic cache disabled: REDIS_HOST not configured")
            self.semantic_cache = None

        # 🔥 cache service
        """
            ✔ small chunks (350)
            ✔ overlap (80)
            ✔ semantic tagging
            ✔ natural language enrichment
            ✔ clean text normalization
        """

        # 🔥 evaluator for dynamic thresholding and cache management
        self.evaluator = RAGEvaluator(self.embedding_model)

        logger.info("Assistant initialized successfully")

    # 🔥 sync → async bridge
    async def _stream_sync(self, generator):
        for token in generator:
            yield token
            await asyncio.sleep(0)

    # 🔥 query normalization SO THAT TO HAVE SAME search question when to run the llm
    @traceable(name="normalize_query", run_type="llm")
    async def _normalize_query(self, question: str):

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

        response = await self.rag_chain.llm.ainvoke(prompt)

        if response == "__RATE_LIMIT__":
            raise Exception("RATE_LIMIT")

        if not response or response.strip() == "":
            return question

        return response.strip()

    @traceable(name="routing_llm_parallel", run_type="llm")
    async def _route_query(self, question: str):
        """
        this function does the following:
        1. starts query normalization in the background (async)
        2. checks cache with RAW question (fast path)
        3. waits for normalization to complete
        4. checks cache with normalized question (semantic path)
        5. if still no cache hit, runs the routing LLM to classify the query
        """
        # 🔥 STEP 1 — start normalization in background
        normalize_task = asyncio.create_task(self._normalize_query(question))

        # 🔥 STEP 2 — try cache with RAW question first (fast path)
        cached = self.semantic_cache.lookup(question) if self.semantic_cache else None

        if cached:
            logger.info("⚡ Cache HIT (raw query)")
            normalize_task.cancel()  # 🔥 kill unnecessary task
            return {**cached, "_cache": True}

        # 🔥 STEP 3 — wait for normalization
        try:
            normalized = await normalize_task
        except asyncio.CancelledError:
            normalized = question

        cached = self.semantic_cache.lookup(normalized) if self.semantic_cache else None
        if cached:
            logger.info("⚡ Cache HIT (semantic)")
            return {**cached, "_cache": True}

        logger.info("❌ CACHE MISS → calling LLM")
        prompt = f"""
            You are a query classifier for a personal AI assistant representing a professional candidate.
            
            Classify the user query into EXACTLY one of these types:
            - "greeting"      → the user is saying hello, hi, hey, good morning, or any form of welcome
            - "professional"  → the user is asking about skills, experience, education, projects, achievements, career, resume, qualifications, certifications, or anything directly related to the candidate's professional profile
            - "general"       → the user is asking a general knowledge question not related to the candidate's profile (e.g. about technology, coding, the world, etc.)
            - "out_of_scope"  → the user is asking something completely irrelevant, inappropriate, or outside any reasonable assistant scope
            
            Return ONLY this exact JSON — no explanation, no markdown, no extra text:
            {{
                "type": "greeting | professional | general | out_of_scope",
                "intent": "one short phrase describing what the user wants",
                "requires_retrieval": true | false,
                "queries": ["search query 1", "search query 2", "search query 3"]
            }}
            
            Rules for requires_retrieval:
            - true  → if type is "professional" (needs resume context)
            - false → for all other types
            
            Rules for queries:
            - For "professional" → write 3 specific search queries to retrieve relevant resume sections
            - For all others     → return ["none"]
            
            User Query:
            {normalized}
        """

        response = await self.rag_chain.llm.ainvoke(prompt)

        # 🔥 HANDLE RATE LIMIT PROPERLY
        if response == "__RATE_LIMIT__":
            raise Exception("RATE_LIMIT")

        if not response or response.strip() == "":
            raise ValueError("Empty LLM response")

        response = response.strip()

        if not response.startswith("{"):
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                response = response[start : end + 1]
            else:
                raise ValueError("No JSON found")

        result = json.loads(response)

        if self.semantic_cache:
            self.semantic_cache.store(normalized, result)
            logger.info(f"Cache Stats: {self.semantic_cache.stats()}")

        result["_cache"] = False
        return result

    async def _retrieve_context_async(self, queries, qtype, requires):

        if not requires:
            return "", []

        loop = asyncio.get_event_loop()

        def retrieve():
            try:
                if self.vector_manager is None:
                    from server.rag.vector_store import VectorStoreManager
                    self.vector_manager = VectorStoreManager(self.documents)

                return self.vector_manager.get_relevant_documents(queries)

            except Exception as e:
                logger.error(f"❌ Retrieval error: {str(e)}")
                return []

        try:
            chunks = await loop.run_in_executor(None, retrieve)
        except Exception as e:
            logger.error(f"❌ Async retrieval crash: {str(e)}")
            return "", []

        if not chunks:
            logger.warning("⚠️ No chunks retrieved")

        return "\n\n".join(chunks), chunks

    # 🔥 prompt builder
    def _build_prompt(self, question, context, qtype, intent):

        if qtype == "greeting":
            return f"""
                You are {self.assistant_name}, the dedicated personal AI assistant representing {self.person_name}.
                
                A visitor has just greeted you. Your job is to:
                1. Warmly greet them back in a natural, friendly, and professional tone
                2. Introduce yourself clearly — mention that you are the personal assistant for {self.person_name}
                3. Give the visitor a clear idea of what they can ask you about. For example:
                - {self.person_name}'s technical skills and expertise
                - Work experience and past projects
                - Education and certifications
                - Achievements and contributions
                - Anything about their professional background
                4. Invite them to ask their first question with an encouraging closing line
                
                Keep the response concise, warm, and inviting. Do NOT make it a bullet list — write it as natural flowing text.
                
                Visitor's message:
                {question}
                
                Your response:
                """

        if qtype == "professional":
            return f"""
                You are {self.assistant_name}, the personal AI assistant representing {self.person_name} to recruiters and professional contacts.
                
                Your role is to showcase {self.person_name}'s professional profile with confidence, clarity, and enthusiasm.
                
                Resume context (use ONLY this — do not invent facts):
                ---------------------
                {context}
                ---------------------
                
                The visitor's question:
                {question}
                
                Intent detected: {intent}
                
                Write your response in exactly THREE parts — do not use headings, just flow naturally:
                
                PART 1 — SUSPENSE / HOOK (1–2 sentences):
                Open with a compelling, confident statement that builds curiosity about {self.person_name}'s answer to this question. Make the reader want to keep reading. Do NOT answer the question yet — just set the stage.
                
                PART 2 — CLARIFICATION / ALIGNMENT (the main body):
                Now directly and thoroughly answer the question using the resume context. Be specific — mention real skills, real project details, real achievements from the context. Show exactly how {self.person_name} aligns with what the visitor is looking for. Use a confident, enthusiastic tone. If the context does not contain enough detail, say so honestly — do not fabricate.
                
                PART 3 — CONCLUSION (1–2 sentences):
                End with a strong, positive closing that reinforces {self.person_name}'s value and invites a follow-up question or next step.
                
                Your response:
                """

        if qtype == "general":
            return f"""
                You are {self.assistant_name}, the personal AI assistant for {self.person_name}.
                
                The visitor has asked a general question that is not directly about {self.person_name}'s professional profile.
                
                The visitor's question:
                {question}
                
                Your response should:
                1. Be honest and transparent — tell them whether this question is something you can helpfully answer or not
                2. If it IS reasonably within scope (e.g. a general tech question, coding question, or something you can answer briefly) — go ahead and answer it concisely and helpfully
                3. If it is NOT within your scope — do NOT just refuse coldly. Instead:
                - Acknowledge the question warmly
                - Explain that your primary focus is representing {self.person_name}'s professional profile
                - Guide them to the kind of questions you CAN help with, for example:
                    * {self.person_name}'s technical skills
                    * Work experience and projects
                    * Education and certifications
                    * Professional achievements
                4. End with an inviting question that encourages them to ask something relevant
                
                Keep the tone friendly, helpful, and never dismissive.
                
                Your response:
                """

        if qtype == "out_of_scope":
            return f"""
                You are {self.assistant_name}, the personal AI assistant for {self.person_name}.
                
                The visitor has asked something that is completely outside the scope of what you are allowed or designed to answer.
                
                The visitor's question:
                {question}
                
                Your response must:
                1. Clearly and politely state that this question is outside your scope — be direct, not vague
                2. Do NOT apologize excessively — be confident and professional
                3. Briefly explain what you ARE here for:
                - You exist to represent {self.person_name} professionally
                - You can speak to their skills, experience, projects, achievements, and background
                4. Guide the visitor toward relevant questions they CAN ask, with 2–3 short examples
                5. End with a warm invitation to redirect the conversation
                
                Do not answer the out-of-scope question in any way. Stay firmly within your purpose.
                
                Your response:
                """

        # Fallback (should not reach here given routing, but safe default)
        return f"""
            You are {self.assistant_name}, personal assistant of {self.person_name}.
            
            Answer the following question using the context provided. Be professional and helpful.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
        """

    # 🔥 greeting
    def _generate_greeting(self):

        hour = datetime.datetime.now().hour

        if hour < 12:
            return f"Good morning! 👋 I'm {self.assistant_name}, the personal assistant for {self.person_name}. Feel free to ask me about their skills, experience, or projects!"
        elif hour < 18:
            return f"Good afternoon! 👋 I'm {self.assistant_name}, here to represent {self.person_name}. What would you like to know about their professional background?"
        else:
            return f"Good evening! 👋 I'm {self.assistant_name}, {self.person_name}'s personal assistant. Ask me anything about their qualifications or experience!"

    # 🔥 streaming with interrupt
    async def stream_with_interrupt(self, prompt, cancel_event):

        try:
            async for token in self.rag_chain.llm.astream(
                prompt, config={"cancel_event": cancel_event}
            ):

                if cancel_event.is_set():
                    # 🔥 HARD INTERRUPT
                    logger.error("⚡ Interrupt signal received. Stopping stream.")
                    yield "[DONE]\n\n"
                    return

                # 🔥 HANDLE RATE LIMIT SIGNAL
                if token == "__RATE_LIMIT__":
                    yield "⚠️ The system is currently overloaded. Please try again shortly.\n\n"
                    yield "[DONE]\n\n"
                    return

                yield token
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            # 🔥 IMPORTANT
            logger.error("⚡ Streaming cancelled by client.")
            yield "[DONE]\n\n"
            return
        except Exception as e:
            logger.error(f"❌ Streaming error: {str(e)}")
            yield "⚠️ Something went wrong. Please try again.\n\n"
            yield "[DONE]\n\n"

    def get_streaming_response_sync(self, question, cancel_event):
        """
        🔥 Sync wrapper for thread execution
        """
        return self.get_streaming_response(question, cancel_event)

    # 🚀 MAIN PIPELINE (SINGLE PASS)
    @traceable(name="single_pass_pipeline", run_type="llm")
    async def get_streaming_response(self, question, cancel_event):
        try:
            route = await self._route_query(question)

        except Exception as e:

            if str(e) == "RATE_LIMIT":

                yield (
                    "⚠️ We're currently experiencing high traffic and couldn't process your request.\n\n"
                    "Please try again in a few moments.\n"
                )
                yield f"[META]{json.dumps({'type': 'rag_eval', 'data': self.evaluator._empty()})}\n"
                yield "[DONE]\n\n"
                return

            raise

        qtype = route["type"]
        queries = route["queries"]
        intent = route["intent"]
        requires = route["requires_retrieval"]
        cache_hit = route.get("_cache", False)

        # 🔥 EMIT CACHE META — frontend picks this up for per-bubble badge
        stats = self.semantic_cache.stats() if self.semantic_cache else {
            "enabled": False,
            "size": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate": 0,
            "memory_mb": 0.0,
        }

        cache_meta = json.dumps(
            {
                "type": "cache",
                "hit": cache_hit,
                "stats": stats,
                "threshold": self.semantic_cache.threshold if self.semantic_cache else None,
                "ttl": self.semantic_cache.ttl if self.semantic_cache else None,
            }
        )

        yield f"[META]{cache_meta}\n"

        if qtype == "greeting":
            greeting = self._generate_greeting()
            yield f"{greeting}\n\n"
            yield f"[META]{json.dumps({'type': 'rag_eval', 'data': self.evaluator._empty()})}\n"
            yield "[DONE]"
            return

        # 🔥 STEP 1 — start retrieval
        retrieve_task = asyncio.create_task(
            self._retrieve_context_async(queries, qtype, requires)
        )

        # STEP 2 — send early token (VERY IMPORTANT)
        yield "⏳ Fetching relevant information...\n\n"
        await asyncio.sleep(0)
        
        logger.info("📡 Retrieval started")

        try:
            context_text, context_chunks = await asyncio.wait_for(
                asyncio.shield(retrieve_task),  # 🔥 CRITICAL FIX
                timeout=0.8
            )
        except asyncio.TimeoutError:
            logger.warning("⏳ Retrieval timeout → waiting full result")
            
            # 🔥 SAFE: task is NOT cancelled now
            context_text, context_chunks = await retrieve_task

        logger.info(f"📦 Retrieved chunks: {len(context_chunks)}")
        # 🔥 STEP 4 — full prompt
        final_prompt = self._build_prompt(question, context_text, qtype, intent)
        full_response = ""

        # 🔥 STEP 5 — continue streaming full answer
        async for token in self.stream_with_interrupt(final_prompt, cancel_event):
            if "[DONE]" in token:
                continue
            full_response += token
            yield f"{token}\n\n"

        eval_input = {
            "user_query": question,  # original
            "retrieval_queries": queries,  # or normalized
            "answer": full_response,
            "context_chunks": context_chunks,
        }
        eval_result = self.evaluator.evaluate(eval_input)

        meta = {"type": "rag_eval", "data": eval_result}

        yield f"[META]{json.dumps(meta)}\n"
        yield "[DONE]\n\n"

    def get_static_greeting(self) -> str:
        """A static greeting that doesn't rely on the time of day, for testing or fallback purposes."""
        return (
            f"Hi there! 👋 I'm {self.assistant_name}, the personal AI assistant for {self.person_name[:16]}. "
            f"I'm here to help you learn everything about {self.person_name[:16]}'s professional background — "
            f"their skills, projects, experience, education, and achievements. "
            f"Feel free to ask me anything!"
        )
