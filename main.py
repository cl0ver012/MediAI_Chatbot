import uvicorn
from fastapi import FastAPI, Request, Form
from config import load_env, ModelType
from mediai_bot import setup_llm
from pydantic import BaseModel
from typing import List

class Message(BaseModel):
    role: str
    content: str
    

load_env()

app = FastAPI()

llm = setup_llm()

@app.get("/")
def read_root(request: Request):
    return {"message": "Welcome to the Mental Health Chatbot API!"}


@app.post("/prompt")
def process_prompt(messages: List[Message]):
    response = llm.do(query_type="test", messages=messages, model_type=ModelType.gpt4o)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)