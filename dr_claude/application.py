from starlette.templating import _TemplateResponse
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates

import json
from loguru import logger

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "up"}


@app.get("/")
async def get(request: Request) -> _TemplateResponse:
    return templates.TemplateResponse("chat_history_testing.html", {"request": request})


templates = Jinja2Templates(directory="templates")


@app.websocket("/dr_claude")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        try:
            message = await receive_message(websocket)
            logger.info("Received {}", message)
            # await handle_message(message)
        except WebSocketDisconnect:
            logger.info("websocket disconnect")
            break
        except Exception as e:
            logger.error(e)


async def receive_message(websocket: WebSocket):
    """
    This is a temporary fix to enable backwards compatability with the old
    chat history format. It will be removed once the chat history format is
    updated.
    """
    unparsed_message = await websocket.receive()
    message_text: str = unparsed_message["text"]
    try:
        message_dict = json.loads(message_text)
    except Exception:
        raise ValueError("Failed to parse message")
    return message_dict
