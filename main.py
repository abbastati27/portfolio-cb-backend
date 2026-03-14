from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag import answer_question
from memory import get_history, save_message

import logging

# -----------------------------------
# LOGGING
# -----------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat-api")

app = FastAPI()

# -----------------------------------
# CORS
# -----------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# REQUEST MODEL
# -----------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    logger.info("Health check endpoint called")
    return {"status": "running"}


# -----------------------------------
# CHAT ENDPOINT (STREAMING)
# -----------------------------------

@app.post("/chat")
async def chat(data: ChatRequest):

    logger.info(f"Incoming message: {data.message}")

    history = get_history(data.session_id)

    save_message(data.session_id, "user", data.message)

    stream = answer_question(data.message, history)

    async def generate():

        full_answer = ""

        logger.info("Streaming response started")

        for token in stream:

            full_answer += token

            yield token

        logger.info("Streaming finished")

        save_message(data.session_id, "assistant", full_answer)

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )