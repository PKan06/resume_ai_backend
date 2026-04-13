# server/voice_ws.py
from fastapi import WebSocket
from server.services.tts_client import TTSClient

tts_client = TTSClient()

async def voice_stream(websocket: WebSocket, assistant, session_manager):
    await websocket.accept()

    data = await websocket.receive_json()
    question = data["message"]
    session_id = data.get("session_id", "default")

    # 🔥 interrupt previous session
    session_manager.interrupt(session_id)
    session = session_manager.get_session(session_id)

    try:
        # 🔥 LLM → TTS SERVICE → AUDIO
        async for audio_bytes in tts_client.stream_audio(
            assistant.get_streaming_response(
                question,
                session.cancel_event
            )
        ):
            await websocket.send_bytes(audio_bytes)

        await websocket.send_text("[DONE]")

    except Exception as e:
        await websocket.send_text(f"ERROR: {str(e)}")