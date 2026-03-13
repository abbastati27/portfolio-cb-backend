from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag import answer_question
from memory import get_history, save_message

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
    return {"status": "running"}


# -----------------------------------
# CHAT ENDPOINT (STREAMING)
# -----------------------------------

@app.post("/chat")
def chat(data: ChatRequest):

    history = get_history(data.session_id)

    save_message(data.session_id, "user", data.message)

    stream = answer_question(data.message, history)

    def generate():

        full_answer = ""

        for token in stream:

            full_answer += token
            yield token

        save_message(data.session_id, "assistant", full_answer)

    return StreamingResponse(generate(), media_type="text/plain")