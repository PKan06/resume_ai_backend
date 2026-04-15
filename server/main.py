from fastapi import FastAPI, Request
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse, FileResponse
import asyncio
import logging
import time
import json
import os

from server.core.assistant import FastAPIPersonalAssistant
from server.config import settings
from server.core.session_manager import SessionManager
import server.logger as app_logger
from server.logger import setup_logging

# =========================
# LOGGING SETUP
# =========================
setup_logging()
logger = logging.getLogger(__name__)

session_manager = SessionManager()
assistant = None  # 🔥 lazy load


# =========================
# LIFESPAN (CRITICAL FIX)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global assistant
    logger.info("⏳ Loading assistant...")

    loop = asyncio.get_event_loop()

    # 🔥 load heavy models in background thread
    assistant = await loop.run_in_executor(
        None,
        lambda: FastAPIPersonalAssistant(
            pdf_path=settings.PDF_PATH,
            assistant_name=settings.ASSISTANT_NAME,
            model_name=settings.MODEL_NAME,
        ),
    )

    logger.info("✅ Assistant loaded")
    yield


app = FastAPI(lifespan=lifespan)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat")
async def chat(request: Request):

    # 🔥 guard while loading
    if assistant is None:
        return StreamingResponse(
            iter(["⏳ Assistant is still loading, please wait...\n", "[DONE]\n"]),
            media_type="text/plain",
        )

    try:
        body = await request.json()
    except Exception:
        return StreamingResponse(
            iter(["❌ Invalid JSON request\n", "[DONE]\n"]), media_type="text/plain"
        )

    question = body.get("message", "").strip()
    session_id = body.get("session_id", "default")

    logger.info(f"📩 Incoming request | session={session_id} | msg={question}")

    session_manager.interrupt(session_id)
    session = session_manager.get_session(session_id)

    async def event_stream():

        current = asyncio.current_task()
        session.current_task = current

        start_time = time.time()
        first_token_sent = False

        try:
            # 🔥 CRITICAL FIX — RUN PIPELINE IN THREAD
            loop = asyncio.get_event_loop()

            async def get_stream():
                return await loop.run_in_executor(
                    None,
                    lambda: assistant.get_streaming_response_sync(
                        question, session.cancel_event
                    ),
                )

            stream_gen = await get_stream()

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
        greeting = assistant.get_static_greeting()

        # stream character by character — fake typing effect
        for char in greeting:
            yield char
            await asyncio.sleep(0.02)  # 20ms per char → natural typing speed

        yield "\n[DONE]\n"

    return StreamingResponse(greeting_stream(), media_type="text/plain")


# =========================
# LOG STREAM — real time, optional level filter
# /logs/stream              → all levels
# /logs/stream?type=debug   → DEBUG and above
# /logs/stream?type=info    → INFO and above
# /logs/stream?type=warning → WARNING and above
# /logs/stream?type=error   → ERROR and above
# =========================
@app.get("/logs/stream")
async def stream_logs(type: str = "all"):
    log_file = app_logger.LOG_FILE

    if log_file is None or not log_file.exists():
        return PlainTextResponse("No log file found.", status_code=404)

    # 🔥 level filter map
    LEVEL_MAP = {
        "all": None,
        "debug": "DEBUG",
        "info": "INFO",
        "warning": "WARNING",
        "warn": "WARNING",
        "error": "ERROR",
        "critical": "CRITICAL",
    }

    LEVEL_ORDER = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    filter_level = LEVEL_MAP.get(type.lower())

    def line_passes(line: str) -> bool:
        """Returns True if line meets the minimum level filter."""
        if filter_level is None:
            return True
        for lvl in LEVEL_ORDER[LEVEL_ORDER.index(filter_level) :]:
            if f"| {lvl}" in line:
                return True
        return False

    async def log_generator():
        with open(log_file, "r", encoding="utf-8") as f:
            # send full history first
            for line in f:
                if line.strip() and line_passes(line):
                    yield f"data: {line.rstrip()}\n\n"

            # then tail in real time
            while True:
                line = f.readline()
                if line:
                    if line.strip() and line_passes(line):
                        yield f"data: {line.rstrip()}\n\n"
                else:
                    await asyncio.sleep(0.4)

    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# =========================
# LOG DOWNLOAD
# =========================
@app.get("/logs/download")
async def download_logs():
    log_file = app_logger.LOG_FILE
    if log_file is None or not log_file.exists():
        return PlainTextResponse("No log file found.", status_code=404)
    return FileResponse(str(log_file), media_type="text/plain", filename=log_file.name)


# =========================
# SERVE RESUME PDF
# =========================


@app.get("/resume")
async def serve_resume():
    pdf_path = "data/Tushar_Kankhedia_Resume.pdf"
    if not os.path.exists(pdf_path):
        return PlainTextResponse("Resume not found.", status_code=404)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=resume.pdf"},
    )
