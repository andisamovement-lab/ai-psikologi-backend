# main.py – DeepSeek dengan deteksi multibahasa
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
from langdetect import detect, LangDetectException

# ------------------- DeepSeek config -------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY belum diset")

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}

# ------------------- FastAPI setup --------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

def _detect_language(text: str) -> str:
    """
    Mengembalikan kode bahasa ISO‑639‑1.
    Jika deteksi gagal, fallback ke 'en'.
    """
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "en"

def _build_prompt(user_msg: str, lang: str) -> str:
    """
    Prompt yang menegaskan peran konselor dan meminta balasan
    dalam bahasa yang sama dengan user.
    """
    if lang.startswith("id"):          # bahasa Indonesia
        instruksi = """Kamu adalah AI konselor psikologi yang cerdas dan kontekstual.
Berikan jawaban yang:
- Sesuai konteks pertanyaan
- Tidak mengulang template
- Gunakan bahasa Indonesia yang natural."""
    else:                               # default ke bahasa Inggris
        instruksi = """You are an intelligent, contextual psychology counselor AI.
Provide answers that:
- Fit the context of the question
- Do not repeat templates
- Use the same language as the user."""
    return f"""{instruksi}
User: {user_msg}
AI:"""

async def generate_reply(text: str) -> str:
    # 1️⃣ deteksi bahasa
    lang = _detect_language(text)

    # 2️⃣ buat prompt sesuai bahasa
    prompt = _build_prompt(text, lang)

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a professional psychology counselor."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }

    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(DEEPSEEK_URL, headers=HEADERS, json=payload, timeout=30.0)
            r.raise_for_status()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Koneksi ke DeepSeek gagal: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code,
                                detail=f"Error DeepSeek: {exc.response.text}") from exc

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Respons tidak dikenali dari DeepSeek")

@app.post("/chat")
async def chat(req: ChatRequest):
    reply = await generate_reply(req.message)
    return {"reply": reply}
