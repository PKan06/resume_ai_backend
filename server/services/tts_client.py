# server/services/tts_client.py

import websockets
import asyncio

class TTSClient:

    def __init__(self, url="ws://localhost:9000/ws/tts"):
        self.url = url

    async def stream_audio(self, text_stream):

        async with websockets.connect(self.url) as ws:

            async def sender():
                try:
                    async for token in text_stream:
                        await ws.send(token)

                    await ws.send("[DONE]")

                except Exception as e:
                    print("Sender error:", e)

            async def receiver():
                try:
                    while True:
                        msg = await ws.recv()

                        if isinstance(msg, bytes):
                            yield msg

                        elif msg == "[DONE]":
                            break

                except Exception as e:
                    print("Receiver error:", e)

            sender_task = asyncio.create_task(sender())

            async for audio in receiver():
                yield audio

            await sender_task