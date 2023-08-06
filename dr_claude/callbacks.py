from typing import List, Any
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import LLMResult
import asyncio
from uuid import UUID
from fastapi import WebSocket
from loguru import logger


class PatientHandler(BaseCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        text = response.generations[0][0].text
        logger.info("sending patient text: {}", text)
        asyncio.run(self.websocket.send_json({"patient": text}))


class DoctorHandler(BaseCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        text = response.generations[0][0].text
        logger.info("sending doctor text: {}", text)
        asyncio.run(self.websocket.send_json({"doctor": text}))


def send_json_sync(websocket: WebSocket, text: str, patient: bool) -> None:
    async def async_send_json(websocket, text):
        await websocket.send_json({"patient": text})

    async def async_send_doc_json(websocket, text):
        await websocket.send_json({"doctor": text})

    loop = asyncio.get_event_loop()
    if patient:
        loop.run_until_complete(async_send_json(websocket, text))
    else:
        loop.run_until_complete(async_send_doc_json(websocket, text))
