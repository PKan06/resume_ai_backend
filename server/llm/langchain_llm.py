from langchain_core.runnables import Runnable
from server.llm.streaming_llm import StreamingLLM, LLMRateLimitError
from langsmith import traceable
import asyncio
import logging

logger = logging.getLogger(__name__)


class LangchainLLM(Runnable):

    def __init__(self):
        self.llm = StreamingLLM()

    def _extract_prompt(self, input):
        if hasattr(input, "to_string"):
            return input.to_string()
        return str(input)

    # =========================
    # SYNC WRAPPER (SAFE)
    # =========================
    @traceable(run_type="llm", name="llm_invoke")
    def invoke(self, input, config=None):

        loop = asyncio.get_event_loop()

        if loop.is_running():
            raise RuntimeError(
                "invoke() called inside async loop. Use ainvoke() instead."
            )

        return loop.run_until_complete(
            self.ainvoke(input, config)
        )

    # =========================
    # RETRY LOGIC (CRITICAL)
    # =========================
    async def _complete_with_retry(self, prompt, retries=10):

        for attempt in range(retries):
            try:
                response = await self.llm.complete(prompt)

                if response and isinstance(response, str) and response.strip():
                    return response.strip()

            except LLMRateLimitError:
                logger.warning(f"⚠️ Rate limit (attempt {attempt+1})")

                # exponential backoff
                await asyncio.sleep(0.5 * (attempt + 1))

                continue

            except Exception as e:
                logger.error(f"❌ LLM error (attempt {attempt+1}): {str(e)}")

                await asyncio.sleep(0.4) # brief pause before retrying(400ms)

        # 🔥 FINAL FAILURE
        return "__FAILED__"

    # =========================
    # ASYNC INVOKE (NON-STREAM)
    # =========================
    @traceable(run_type="llm", name="llm_ainvoke")
    async def ainvoke(self, input, config=None):

        prompt = self._extract_prompt(input)

        response = await self._complete_with_retry(prompt)

        # 🔥 HANDLE FINAL FAILURE
        if response == "__FAILED__":
            return "__RATE_LIMIT__"

        return response
    
    
    # =========================
    # STREAMING Flow for each Attempt
    # =========================
    async def _single_stream_attempt(self, prompt, cancel_event=None):

        has_output = False

        async for token in self.llm.stream(prompt, cancel_event):

            # 🔥 INTERRUPT CHECK
            if cancel_event and cancel_event.is_set():
                logger.info("⚡ Stream interrupted")
                return

            # 🔥 VALID TOKEN FILTER
            if token and token.strip():
                has_output = True
                yield token

        # ⚠️ If stream completes but empty
        # if not has_output:
        #     logger.warning("⚠️ Empty stream response")
        #     yield "I'm here — could you rephrase that?"
    
    # =========================
    # STREAMING WITH RETRY (CRITICAL)
    # =========================    
    async def _stream_with_retry(self, prompt, cancel_event=None):

        max_retries = 5
        base_delay = 0.6

        for attempt in range(max_retries):

            has_output = False

            try:
                logger.info(f"🚀 Streaming attempt {attempt + 1}")

                async for token in self._single_stream_attempt(prompt, cancel_event):

                    has_output = True
                    yield token

                # ✅ SUCCESS → exit retry loop
                if has_output:
                    return

                # ⚠️ EMPTY RESPONSE → retry
                logger.warning(f"⚠️ Empty response (attempt {attempt+1})")

                if attempt == max_retries - 1:
                    yield "I'm here — could you rephrase that?"
                    return

                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

            except LLMRateLimitError:
                logger.warning(f"⚠️ Rate limit (attempt {attempt + 1})")

                if attempt == max_retries - 1:
                    yield "__RATE_LIMIT__"
                    return

                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                logger.info("⚡ Streaming cancelled")
                return

            except Exception as e:
                logger.error(f"❌ Streaming error (attempt {attempt+1}): {str(e)}")

                if attempt == max_retries - 1:
                    yield "⚠️ Something went wrong. Please try again."
                    return

                await asyncio.sleep(0.3)

        # 🔥 ultra fallback
        yield "__RATE_LIMIT__"
        
        
        
    # =========================
    # STREAMING (SAFE)
    # =========================
    @traceable(run_type="llm", name="llm_stream")
    async def astream(self, input, config=None):

        prompt = self._extract_prompt(input)

        cancel_event = None
        if config:
            cancel_event = config.get("cancel_event")

        async for token in self._stream_with_retry(prompt, cancel_event):
            yield token