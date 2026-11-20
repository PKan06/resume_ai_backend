# server/main.py
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse, FileResponse, JSONResponse
import asyncio
import logging
import time
import json
import os

from server.core.assistant import FastAPIPersonalAssistant
from server.config import settings
from server.core.session_manager import SessionManager
from server.schemas.chat import ChatRequest
from server.logger import setup_logging

# =========================
# LOGGING SETUP
# =========================
setup_logging()
logger = logging.getLogger(__name__)

session_manager = SessionManager()
assistant = None  # 🔥 lazy load


# =========================
# 🔥 BACKGROUND LOADER (FIX)
# =========================
async def load_assistant():
    global assistant

    try:
        logger.info("⏳ Loading assistant in background...")

        loop = asyncio.get_event_loop()

        assistant_instance = await loop.run_in_executor(
            None,
            lambda: FastAPIPersonalAssistant(
                pdf_path=settings.PDF_PATH,
                assistant_name=settings.ASSISTANT_NAME,
                model_name=settings.MODEL_NAME,
            ),
        )

        assistant = assistant_instance

        logger.info("✅ Assistant loaded")

    except Exception as e:
        logger.error(f"❌ Assistant load failed: {str(e)}")


# =========================
# 🔥 LIFESPAN (NON-BLOCKING FIX)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 App starting...")

    # 🔥 DO NOT BLOCK — run in background
    asyncio.create_task(load_assistant())

    yield

    from server.services.embedding_service import EmbeddingService
    if EmbeddingService._instance is not None:
        EmbeddingService._instance.close()
        logger.info("CloudflareEmbeddings HTTP client closed.")


app = FastAPI(lifespan=lifespan)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat")
async def chat(payload: ChatRequest, request: Request):
    question = payload.message.strip()
    session_id = payload.session_id

    if not question:
        return StreamingResponse(
            iter([
                "You missed the most important part: the question. "
                "Please type what you'd like to know.\n",
                "[DONE]\n",
            ]),
            media_type="text/plain",
        )

    # 🔥 guard while loading
    if assistant is None:
        return StreamingResponse(
            iter(["⏳ Assistant is still loading, please wait...\n", "[DONE]\n"]),
            media_type="text/plain",
        )

    logger.info(f"📩 Incoming request | session={session_id} | msg={question}")

    session_manager.interrupt(session_id)
    session = session_manager.get_session(session_id)

    async def event_stream():

        current = asyncio.current_task()
        session.current_task = current

        start_time = time.time()
        first_token_sent = False

        try:

            stream_gen = assistant.get_streaming_response(
                question,
                session.cancel_event
            )

            async for chunk in stream_gen:

                if await request.is_disconnected():
                    logger.info(f"🔌 Client disconnected | session={session_id}")
                    return

                now = time.time()

                if isinstance(chunk, str) and "[DONE]" in chunk:
                    continue

                if isinstance(chunk, str) and chunk.startswith("[META]"):
                    yield chunk + "\n"
                    continue

                if not first_token_sent and isinstance(chunk, str) and chunk.strip():
                    toft = now - start_time
                    yield f"[META]{json.dumps({'type': 'toft','value': round(toft,3)})}\n"
                    first_token_sent = True

                yield chunk

            tolt = time.time() - start_time
            if not first_token_sent:
                yield f"\n[META]{json.dumps({'type': 'toft','value': round(tolt,3)})}\n"
            yield f"\n[META]{json.dumps({'type': 'tolt','value': round(tolt,3)})}\n"
            yield "[DONE]\n"

        except asyncio.CancelledError:
            if session.cancel_event.is_set():
                logger.info(f"⚡ Request cancelled by user | session={session_id}")
            else:
                logger.info(f"🔌 Stream torn down by infra | session={session_id}")
                
            # 🔥 ensure frontend gets something
            yield "⚠️ Connection interrupted. Please try again.\n"
            yield "[DONE]\n"

        except Exception as e:
            logger.error(f"❌ Streaming Error: {str(e)}")
            yield "⚠️ System error. Please try again.\n"
            yield "[DONE]\n"

        finally:
            if session.current_task is current:
                session.current_task = None

    return StreamingResponse(event_stream(), media_type="text/plain")


@app.get("/greet")
async def greet():

    async def greeting_stream():
        if assistant is None:
            yield "⏳ Assistant is still loading, please wait...\n[DONE]\n"
            return
        greeting = assistant.get_static_greeting()

        # stream character by character — fake typing effect
        for char in greeting:
            yield char
            await asyncio.sleep(0.02)  # 20ms per char → natural typing speed

        yield "\n[DONE]\n"

    return StreamingResponse(greeting_stream(), media_type="text/plain")


# =========================
# SERVE RESUME PDF
# =========================


@app.get("/resume")
async def serve_resume():
    pdf_path = settings.PDF_PATH
    if not os.path.exists(pdf_path):
        return PlainTextResponse("Resume not found.", status_code=404)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=resume.pdf"},
    )


# =========================
# HEALTH / READINESS
# =========================
@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "service": "resume-ai-backend",
    }


@app.get("/readyz")
async def readyz():
    assistant_loaded = assistant is not None
    resume_available = os.path.exists(settings.PDF_PATH)
    cache_available = bool(
        assistant_loaded and getattr(assistant, "semantic_cache", None)
    )
    ready = assistant_loaded and resume_available

    payload = {
        "status": "ready" if ready else "not_ready",
        "assistant_loaded": assistant_loaded,
        "resume_available": resume_available,
        "model_name": settings.MODEL_NAME,
        "cache_enabled": settings.REDIS_ENABLED,
        "cache_available": cache_available,
    }

    return JSONResponse(
        payload,
        status_code=200 if ready else 503,
    )
