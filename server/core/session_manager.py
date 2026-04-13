# server/core/session_manager.py
import asyncio
from typing import Dict


class SessionState:
    def __init__(self):
        self.current_task: asyncio.Task = None
        self.cancel_event: asyncio.Event = asyncio.Event()


class SessionManager:

    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def get_session(self, session_id: str) -> SessionState:

        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState()

        session = self.sessions[session_id]

        # 🔥 ensure fresh event if already used
        if session.cancel_event.is_set():
            session.cancel_event = asyncio.Event()

        return session

    def interrupt(self, session_id: str):

        session = self.get_session(session_id)

        # Signal cancellation
        session.cancel_event.set()

        # Cancel running task
        if session.current_task:
            session.current_task.cancel()

        # Reset event for next request
        session.cancel_event = asyncio.Event()