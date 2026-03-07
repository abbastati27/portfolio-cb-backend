from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer_question
from memory import get_history, save_message
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
def chat(data: ChatRequest):

    history = get_history(data.session_id)

    answer = answer_question(data.message, history)

    save_message(data.session_id, "user", data.message)
    save_message(data.session_id, "assistant", answer)

    return {"answer": answer}