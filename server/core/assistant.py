# server/core/assistant.py (setup + query routing + pipeline orchestration)
import logging
import asyncio
import json
import time
from langsmith import traceable
import redis

from server.services.semantic_cache import SemanticCache
from server.config import settings

from server.rag.helper.document_processor import DocumentProcessor
from server.rag.helper.profile_extractor import ProfileExtractor
from server.llm.langchain_llm import LangchainLLM
from server.rag.normalizer import Normalizer
from server.rag.retriever import Retriever
from server.rag.generator import Generator

from server.services.embedding_service import EmbeddingService
from server.services.rag_evaluator import RAGEvaluator


logger = logging.getLogger(__name__)


class FastAPIPersonalAssistant:

    def __init__(self, pdf_path, model_name, assistant_name):

        self.pdf_path = pdf_path
        self.model_name = model_name
        self.assistant_name = assistant_name

        self.documents = None
        self.llm = LangchainLLM()
        self.person_name = "Unknown"

        self._initialize_system()

    def _initialize_system(self):

        processor = DocumentProcessor(self.pdf_path)

        # 🔥 RAW docs for profile extraction (unsplit, unenriched)
        raw_documents = processor.load_raw()

        # 🔥 Processed docs for RAG (split + enriched)

        self.documents = processor.load_and_process()

        extractor = ProfileExtractor(raw_documents)
        self.person_name, _ = extractor.extract()

        # 🔥 RAG components: normalizer -> retriever (lazy vector store) -> generator
        self.normalizer = Normalizer(self.llm)
        self.retriever = Retriever(self.documents)
        self.generator = Generator(self.llm, self.assistant_name, self.person_name)

        # 🔥 embedding model
        self.embedding_model = EmbeddingService.get_instance()

        # 🔥 Redis client (Redis Cloud, TCP) — optional, degrades to no-cache on failure
        if settings.REDIS_ENABLED and settings.REDIS_HOST:
            try:
                redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    username=settings.REDIS_USERNAME,
                    password=settings.REDIS_PASSWORD,
                    db=0,
                    decode_responses=True,
                    ssl=False,
                )
                
                redis_client.ping()  # client is lazy; verify connectivity once at startup
                
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

        # 🔥 evaluator for dynamic thresholding and cache management
        self.evaluator = RAGEvaluator(self.embedding_model)

        logger.info("Assistant initialized successfully")

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
        normalize_task = asyncio.create_task(self.normalizer.normalize(question))

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

        response = await self.llm.ainvoke(prompt)

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

    # 🔥 retrieval only when the route needs resume context
    async def _retrieve(self, queries, requires):
        if not requires:
            return "", []
        return await self.retriever.aretrieve(queries)

    async def run_eval_query(self, question: str, generate_answer: bool = False) -> dict:
        start_time = time.perf_counter()
        answer = ""
        context_text = ""
        context_chunks = []
        eval_result = self.evaluator._empty()

        try:
            route = await self._route_query(question)
            qtype = route["type"]
            queries = route["queries"]
            intent = route["intent"]
            requires = route["requires_retrieval"]
            cache_hit = route.get("_cache", False)

            context_text, context_chunks = await self._retrieve(queries, requires)

            if generate_answer:
                answer = await self.generator.generate(question, context_text, qtype, intent)

                eval_input = {
                    "user_query": question,
                    "retrieval_queries": queries,
                    "answer": answer,
                    "context_chunks": context_chunks,
                }
                eval_result = self.evaluator.evaluate(eval_input)

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return {
                "question": question,
                "route_type": qtype,
                "intent": intent,
                "retrieval_queries": queries,
                "context_chunks": context_chunks,
                "answer": answer,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "rag_eval": eval_result,
            }

        except Exception as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"Eval query failed: {e}")
            return {
                "question": question,
                "route_type": "error",
                "intent": "",
                "retrieval_queries": [],
                "context_chunks": context_chunks,
                "answer": answer,
                "cache_hit": False,
                "latency_ms": latency_ms,
                "rag_eval": eval_result,
                "error": str(e),
            }

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
            greeting = self.generator.time_greeting()
            yield f"{greeting}\n\n"
            yield f"[META]{json.dumps({'type': 'rag_eval', 'data': self.evaluator._empty()})}\n"
            yield "[DONE]"
            return

        # 🔥 STEP 1 — start retrieval
        retrieve_task = asyncio.create_task(
            self._retrieve(queries, requires)
        )

        # STEP 2 — send early token (VERY IMPORTANT)
        # yield "⏳ Fetching relevant information...\n\n"
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
        final_prompt = self.generator.build_prompt(question, context_text, qtype, intent)
        full_response = ""

        # 🔥 STEP 5 — continue streaming full answer
        async for token in self.generator.stream(final_prompt, cancel_event):
            if "[DONE]" in token:
                continue
            full_response += token
            yield token

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
