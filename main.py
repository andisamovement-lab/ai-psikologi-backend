from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client
from cachetools import TTLCache
import requests
from bs4 import BeautifulSoup
import re

app = FastAPI()

# ================= CONFIG =================
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
client = Client(HF_MODEL)

CACHE = TTLCache(maxsize=500, ttl=3600)

PSYCHOLOGY_SITES = [
    "https://www.psychologytoday.com",
    "https://positivepsychology.com",
    "https://www.verywellmind.com",
]

CRISIS_KEYWORDS = [
    "bunuh diri", "ingin mati", "tidak ingin hidup",
    "self harm", "melukai diri", "suicide"
]

# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"]
)

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

# ---------- UTIL ----------
def is_crisis(text: str) -> bool:
    text = text.lower()
    return any(k in text for k in CRISIS_KEYWORDS)

def crawl_psychology(query: str) -> str:
    summaries = []

    for site in PSYCHOLOGY_SITES:
        try:
            res = requests.get(site, timeout=5)
            soup = BeautifulSoup(res.text, "html.parser")

            paragraphs = soup.find_all("p")
            content = " ".join(p.get_text() for p in paragraphs[:10])

            if query.lower() in content.lower():
                summaries.append(content[:500])
        except:
            continue

    return " ".join(summaries)

def generate_ai_response(prompt: str, knowledge: str) -> str:
    system_prompt = f"""
Kamu adalah AI konselor psikologi.
Bahasa lembut, empatik, tidak menghakimi.
Tidak mendiagnosis.
Jika krisis, arahkan ke bantuan profesional.

Referensi psikologi:
{knowledge}

Pertanyaan pengguna:
{prompt}
"""

    return client.predict(system_prompt, api_name="/predict")

# ---------- API ----------
@app.post("/chat")
async def chat(data: dict):
    message = data.get("message", "").strip()

    if not message:
        return {"reply": "Silakan ceritakan apa yang sedang kamu rasakan."}

    if message in CACHE:
        return {"reply": CACHE[message]}

    # 🚨 CRISIS MODE
    if is_crisis(message):
        response = (
            "Aku sangat menyesal kamu merasa seperti ini. Kamu tidak sendirian.\n\n"
            "Jika kamu berada di Indonesia, kamu bisa menghubungi:\n"
            "📞 Hotline Kemenkes: 1500-454\n"
            "📞 Sejiwa 119 ext. 8\n\n"
            "Jika kamu bersedia, ceritakan apa yang membuatmu merasa seperti ini."
        )
        CACHE[message] = response
        return {"reply": response}

    # 🔍 KNOWLEDGE CRAWLING
    knowledge = crawl_psychology(message)

    try:
        reply = generate_ai_response(message, knowledge)
    except:
        reply = (
            "Aku di sini untuk mendengarkan. "
            "Tidak apa-apa merasa bingung atau lelah. "
            "Ceritakan lebih lanjut jika kamu mau."
        )

    CACHE[message] = reply
    return {"reply": reply}

