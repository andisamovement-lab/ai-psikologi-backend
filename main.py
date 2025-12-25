import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.genai as genai

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY belum diset")

client = genai.Client(api_key=API_KEY)

app = FastAPI()

# CORS (WAJIB)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

def generate_reply(text: str) -> str:
    prompt = f"""
Kamu adalah AI konselor psikologi yang cerdas dan kontekstual.

Aturan:
- Jawab sesuai konteks
- Jangan mengulang template
- Gunakan bahasa user

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
