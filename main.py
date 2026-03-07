from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer_question
from memory import get_history, save_message

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/chat")
def chat(data: ChatRequest):

    history = get_history(data.session_id)

    answer = answer_question(data.message, history)

    save_message(data.session_id, "user", data.message)
    save_message(data.session_id, "assistant", answer)

    return {"answer": answer}