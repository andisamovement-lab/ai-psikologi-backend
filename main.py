import os
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

# =====================
# KONFIGURASI
# =====================
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY belum diset")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-flash"

app = FastAPI()

# =====================
# SCHEMA
# =====================
class ChatRequest(BaseModel):
    message: str

# =====================
# AI LOGIC
# =====================
def generate_reply(user_message: str) -> str:
    prompt = f"""
Kamu adalah AI konselor psikologi yang cerdas, empatik, dan solutif.

Aturan:
- Jawab SESUAI konteks pertanyaan user
- Jangan mengulang kalimat yang sama
- Jika user curhat → beri empati + solusi
- Jika user bertanya → beri jawaban langsung
- Gunakan bahasa yang SAMA dengan bahasa user

User:
{user_message}

Jawaban terbaik:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()

# =====================
# API ENDPOINT
# =====================
@app.post("/chat")
def chat(req: ChatRequest):
    reply = generate_reply(req.message)
    return {"reply": reply}
