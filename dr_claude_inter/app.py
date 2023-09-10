"""The primary entrypoint for running the Dr Claude app.

Run `make app` to run.
Alternatively, run with `poetry run uvicorn dr_claude.app:app`
"""

from typing import Optional, Tuple, List
import json
import nest_asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import transformers
from loguru import logger


from dr_claude_inter import datamodels
from dr_claude_inter.kb import kb_reading


transformers.set_seed(42)

app = FastAPI()

nest_asyncio.apply()
