# server/main.py
from fastapi import FastAPI, Request, WebSocket
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

# voice
from server.voice_ws import voice_stream

# =========================
# LOGGING SETUP
# =========================
setup_logging()
logger = logging.getLogger(__name__)

# =========================
# APP INIT
# =========================
app = FastAPI()

session_manager = SessionManager()

assistant = FastAPIPersonalAssistant(
    pdf_path=settings.PDF_PATH,
    assistant_name=settings.ASSISTANT_NAME,
    model_name=settings.MODEL_NAME
)

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

    try:
        body = await request.json()
    except Exception:
        return StreamingResponse(
            iter(["❌ Invalid JSON request\n", "[DONE]\n"]),
            media_type="text/plain"
        )

    question = body.get("message", "").strip()
    session_id = body.get("session_id", "default")

    logger.info(f"📩 Incoming request | session={session_id} | msg={question}")

    # 🔥 interrupt previous request
    session_manager.interrupt(session_id)
    session = session_manager.get_session(session_id)

    async def event_stream():

        # =========================
        # BUG FIX #1: Register the streaming task HERE, inside event_stream,
        # not in the outer chat() handler. This is the actual task that
        # performs the streaming work and needs to be tracked/cancelled.
        # asyncio.current_task() in chat() returns the HTTP handler coroutine,
        # which finishes immediately after returning StreamingResponse —
        # leaving the real streaming task untracked and un-cancellable.
        # =========================
        current = asyncio.current_task()
        session.current_task = current
        logger.debug(f"🎯 Streaming task registered | session={session_id} | task={current.get_name()}")

        start_time = time.time()
        first_token_sent = False

        try:
            async for chunk in assistant.get_streaming_response(
                question,
                session.cancel_event
            ):

                # =========================
                # BUG FIX #3: Check for client disconnect before yielding.
                # When the client closes the connection, FastAPI/Starlette
                # will raise an exception on the next yield — but we can
                # detect this early with request.is_disconnected() to avoid
                # a false CancelledError being logged as a user interruption.
                # =========================
                if await request.is_disconnected():
                    logger.info(f"🔌 Client disconnected (not a user interrupt) | session={session_id}")
                    return

                now = time.time()

                # =========================
                # INTERCEPT [DONE] FROM ASSISTANT — don't forward it
                # main.py owns the [DONE] signal
                # =========================
                if isinstance(chunk, str) and "[DONE]" in chunk:
                    continue

                # =========================
                # HANDLE META FROM ASSISTANT
                # =========================
                if isinstance(chunk, str) and chunk.startswith("[META]"):
                    yield chunk + "\n"
                    continue

                # =========================
                # TOFT (FIRST TOKEN)
                # =========================
                if not first_token_sent:
                    toft = now - start_time
                    meta = {
                        "type": "toft",
                        "value": round(toft, 3)
                    }
                    yield f"[META]{json.dumps(meta)}\n"
                    first_token_sent = True

                # =========================
                # NORMAL TOKEN
                # =========================
                yield chunk

            # =========================
            # TOLT (END)
            # =========================
            tolt = time.time() - start_time
            meta = {
                "type": "tolt",
                "value": round(tolt, 3)
            }
            yield f"\n[META]{json.dumps(meta)}\n"

            yield "[DONE]\n"

        except asyncio.CancelledError:
            # =========================
            # BUG FIX: Distinguish between a genuine user-triggered barge-in
            # (cancel_event is set) vs. an infrastructure-level cancellation
            # (uvicorn tearing down the connection, task GC, etc.).
            # Only log as "Request cancelled" when the user actually interrupted.
            # =========================
            if session.cancel_event.is_set():
                logger.info(f"⚡ Request cancelled by user | session={session_id}")
            else:
                logger.info(f"🔌 Stream torn down by infrastructure (not user) | session={session_id}")
            yield "[DONE]\n"
            return

        except Exception as e:
            logger.error(f"❌ Streaming Error: {str(e)}")
            yield "⚠️ System error. Please try again.\n"
            yield "[DONE]\n"

        finally:
            # =========================
            # Always clear the task reference when done so stale
            # task handles don't accumulate in the session.
            # =========================
            if session.current_task is current:
                session.current_task = None

    # =========================
    # BUG FIX #1 (cont): Do NOT set session.current_task here.
    # The correct place is inside event_stream() above, where
    # asyncio.current_task() refers to the actual streaming task.
    # Setting it here captures the HTTP handler coroutine, which is
    # already finishing as StreamingResponse is returned.
    # =========================
    return StreamingResponse(
        event_stream(),
        media_type="text/plain"
    )


@app.get("/greet")
async def greet():

    async def greeting_stream():
        greeting = assistant.get_static_greeting()
        
        # stream character by character — fake typing effect
        for char in greeting:
            yield char
            await asyncio.sleep(0.02)  # 20ms per char → natural typing speed
        
        yield "\n[DONE]\n"

    return StreamingResponse(
        greeting_stream(),
        media_type="text/plain"
    )

# =========================
# VOICE STREAM ENDPOINT
# =========================
@app.websocket("/ws/voice")
async def voice_ws_endpoint(ws: WebSocket):
    await voice_stream(ws, assistant, session_manager)
    
    


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
        "all":     None,
        "debug":   "DEBUG",
        "info":    "INFO",
        "warning": "WARNING",
        "warn":    "WARNING",
        "error":   "ERROR",
        "critical":"CRITICAL",
    }

    LEVEL_ORDER = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    filter_level = LEVEL_MAP.get(type.lower())

    def line_passes(line: str) -> bool:
        """Returns True if line meets the minimum level filter."""
        if filter_level is None:
            return True
        for lvl in LEVEL_ORDER[LEVEL_ORDER.index(filter_level):]:
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
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# =========================
# LOG DOWNLOAD
# =========================
@app.get("/logs/download")
async def download_logs():
    log_file = app_logger.LOG_FILE
    if log_file is None or not log_file.exists():
        return PlainTextResponse("No log file found.", status_code=404)
    return FileResponse(
        str(log_file),
        media_type="text/plain",
        filename=log_file.name
    )
    
    

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
        headers={"Content-Disposition": "inline; filename=resume.pdf"}
    )