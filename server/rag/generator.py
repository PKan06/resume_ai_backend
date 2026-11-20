# server/rag/generator.py

import asyncio
import datetime
import logging

logger = logging.getLogger(__name__)


class Generator:
    """
    ✍️ Generator

    (question, context, qtype, intent) -> prompt -> answer (streamed or full).
    Kept independent of routing/retrieval so it can be evaluated on its own
    (e.g. against gold context).
    """

    def __init__(self, llm, assistant_name, person_name):
        self.llm = llm
        self.assistant_name = assistant_name
        self.person_name = person_name

    # 🔥 prompt builder
    def build_prompt(self, question, context, qtype, intent):

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

    # 🔥 time-based greeting (no LLM call)
    def time_greeting(self):

        hour = datetime.datetime.now().hour

        if hour < 12:
            return f"Good morning! 👋 I'm {self.assistant_name}, the personal assistant for {self.person_name}. Feel free to ask me about their skills, experience, or projects!"
        elif hour < 18:
            return f"Good afternoon! 👋 I'm {self.assistant_name}, here to represent {self.person_name}. What would you like to know about their professional background?"
        else:
            return f"Good evening! 👋 I'm {self.assistant_name}, {self.person_name}'s personal assistant. Ask me anything about their qualifications or experience!"

    # 🔥 streaming with interrupt
    async def stream(self, prompt, cancel_event):

        try:
            async for token in self.llm.astream(
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

    # 🔥 non-streaming answer (used by eval)
    async def generate(self, question, context, qtype, intent) -> str:

        prompt = self.build_prompt(question, context, qtype, intent)
        answer = await self.llm.ainvoke(prompt)

        if answer == "__RATE_LIMIT__":
            return ""

        return answer


# =========================
# STANDALONE CHECK
# =========================
if __name__ == "__main__":
    # Usage (from backend/):  python -m server.rag.generator "your question"
    import io
    import sys
    from server.config import settings
    from server.llm.langchain_llm import LangchainLLM

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console + emojis
    logging.basicConfig(level=logging.INFO)

    question = " ".join(sys.argv[1:]) or "What are the candidate's main skills?"

    # Fixed sample context -> tests the generator alone (no retrieval / Pinecone).
    # Edit it, or switch qtype to "general" / "out_of_scope" to try other prompts.
    sample_context = (
        "Skills: Python, FastAPI, LangChain, Redis, Pinecone, Docker, AWS EC2.\n"
        "Project: Real-time voice AI resume assistant with RAG and semantic caching."
    )

    generator = Generator(LangchainLLM(), settings.ASSISTANT_NAME, "the candidate")

    async def main():
        prompt = generator.build_prompt(question, sample_context, "professional", "skills overview")
        print(f"\nQuestion: {question}\n\nAnswer (streaming):\n")
        async for token in generator.stream(prompt, asyncio.Event()):
            print(token, end="", flush=True)
        print()

    asyncio.run(main())
