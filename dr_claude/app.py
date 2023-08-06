"""HTTP app for Dr Claude."""

from typing import Optional, Tuple, List
import json
import nest_asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import transformers
from loguru import logger


from dr_claude import datamodels
from dr_claude.kb import kb_reading


transformers.set_seed(42)

app = FastAPI()

nest_asyncio.apply()
