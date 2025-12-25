import os
from fastapi import FastAPI
from pydantic import BaseModel
import google.genai as genai

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY belum diset")

client = genai.Client(api_key=API_KEY)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

def generate_reply(text: str) -> str:
    prompt = f"""
Kamu adalah AI konselor psikologi yang cerdas dan kontekstual.
Jawab langsung, relevan, dan tidak mengulang template.

User:
{text}
"""
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    return response.text.strip()

@app.post("/chat")
def chat(req: ChatRequest):
    return {"reply": generate_reply(req.message)}
