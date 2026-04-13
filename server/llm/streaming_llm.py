# server/llm/streaming_llm.py
from server.config import settings
import asyncio
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class LLMRateLimitError(Exception):
    pass


class StreamingLLM:

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.MODEL_NAME
        self.system_instruction = (
            "You are Alex, an AI assistant. Answer ONLY from provided context. "
            "If unsure, say you don't know."
        )

    # =========================
    # NON-STREAM (FOR ROUTING)
    # =========================
    async def complete(self, prompt):
        try:
            def _call():
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        temperature=0.3,
                    )
                )
                return response.text or ""

            result = await asyncio.to_thread(_call)
            return result.strip()

        except Exception as e:
            logger.error(f"❌ Gemini complete() error: {str(e)}")
            if "429" in str(e) or "quota" in str(e).lower():
                raise LLMRateLimitError("Gemini rate limit")
            return ""

    # =========================
    # STREAMING (FOR FINAL ANSWER)
    # =========================
    async def stream(self, prompt, cancel_event=None):
        """
        BUG FIX #2: The original implementation called asyncio.to_thread()
        to get the stream object, then iterated over it synchronously with
        `for chunk in stream`. This is safe but means cancellation can only
        be checked between chunks — which is fine.

        The real problem was that asyncio.to_thread() for the generator
        creation itself was unnecessary overhead, and more importantly the
        synchronous `for` loop inside an async generator means any
        CancelledError injected by the event loop between awaits would surface
        as an unhandled exception bubbling up through LangchainLLM and getting
        logged as a user cancellation.

        Fix: wrap the synchronous chunk iteration in a try/except for
        CancelledError so we can distinguish a real cancellation (cancel_event
        set) from infrastructure teardown, and yield cleanly in both cases.
        """
        try:
            def _stream_generator():
                return self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=prompt,
                )

            # Run the blocking SDK call that returns the stream object
            # in a thread so we don't block the event loop.
            stream_obj = await asyncio.to_thread(_stream_generator)

            has_output = False

            for chunk in stream_obj:

                # =========================
                # BUG FIX #2a: Check cancel_event first — this is the
                # user-triggered barge-in path. If set, exit cleanly.
                # =========================
                if cancel_event and cancel_event.is_set():
                    logger.info("⚡ Stream cancelled by cancel_event")
                    return

                # =========================
                # BUG FIX #2b: Yield control back to the event loop after
                # each chunk. The synchronous `for chunk in stream_obj` loop
                # never yields — meaning the event loop can't process other
                # tasks (like disconnect detection) between chunks.
                # This single await makes the generator cooperative.
                # =========================
                await asyncio.sleep(0)

                # Re-check after yielding to event loop — a cancellation
                # may have been injected during the sleep(0).
                if cancel_event and cancel_event.is_set():
                    logger.info("⚡ Stream cancelled after sleep(0)")
                    return

                if not chunk:
                    continue

                text = getattr(chunk, "text", None)

                if text:
                    has_output = True
                    yield text

            if has_output:
                return

        except asyncio.CancelledError:
            # =========================
            # BUG FIX #2c: CancelledError here means the event loop tore
            # down this coroutine — NOT necessarily a user barge-in.
            # Re-raise so the caller (LangchainLLM._stream_with_retry) can
            # handle it correctly. Do NOT yield anything here.
            # =========================
            logger.debug("🔌 StreamingLLM.stream() received CancelledError — propagating")
            raise

        except Exception as e:
            err = str(e)
            logger.warning(f"⚠️ Gemini stream error: {err}")

            if any(x in err.lower() for x in ["503", "unavailable", "overload", "quota", "429"]):
                raise LLMRateLimitError("Gemini rate limit")

            yield "⚠️ Something went wrong."