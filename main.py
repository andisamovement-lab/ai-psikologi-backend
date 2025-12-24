from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests, time, re
from bs4 import BeautifulSoup
from cachetools import TTLCache

app = FastAPI()

# ================= BASIC CONFIG =================
HF_API = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

CACHE = TTLCache(maxsize=300, ttl=600)
MEMORY = {}
RATE_LIMIT = {}
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 20

CRISIS_KEYWORDS = [
    "bunuh diri", "ingin mati", "tidak ingin hidup",
    "suicide", "self harm", "melukai diri"
]

PSY_SITES = [
    "https://www.psychologytoday.com",
    "https://www.verywellmind.com",
    "https://positivepsychology.com",
    "https://www.healthline.com",
    "https://www.mind.org.uk",
    "https://www.nimh.nih.gov",
    "https://www.apa.org",
    "https://www.samhsa.gov",
    "https://www.helpguide.org",
    "https://www.psychcentral.com"
]

# ================= MIDDLEWARE =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= UTIL =================
def is_crisis(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in CRISIS_KEYWORDS)

def rate_limited(ip: str) -> bool:
    now = time.time()
    hits = RATE_LIMIT.get(ip, [])
    hits = [h for h in hits if now - h < RATE_LIMIT_WINDOW]
    if len(hits) >= RATE_LIMIT_MAX:
        RATE_LIMIT[ip] = hits
        return True
    hits.append(now)
    RATE_LIMIT[ip] = hits
    return False

def crawl_psychology(query: str) -> str:
    snippets = []
    for site in PSY_SITES:
        try:
            r = requests.get(site, timeout=4)
            soup = BeautifulSoup(r.text, "html.parser")
            p = soup.find_all("p")
            text = " ".join(x.get_text() for x in p[:5])
            if query.lower() in text.lower():
                snippets.append(text[:300])
        except:
            continue
    return " ".join(snippets)

def ai_generate(prompt: str) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.6,
            "max_new_tokens": 300,
            "return_full_text": False
        }
    }
    r = requests.post(HF_API, json=payload, timeout=45)
    if r.status_code != 200:
        raise Exception("AI unavailable")
    return r.json()[0]["generated_text"]

# ================= API =================
@app.post("/chat")
async def chat(request: Request, data: dict):
    ip = request.client.host
    message = data.get("message", "").strip()

    if not message:
        return {"reply": "Silakan ceritakan apa yang sedang kamu rasakan."}

    if rate_limited(ip):
        return {"reply": "Mohon tunggu sebentar sebelum mengirim pesan lagi."}

    if message in CACHE:
        return {"reply": CACHE[message]}

    if is_crisis(message):
        reply = (
            "Aku sangat menyesal kamu merasa seperti ini. Kamu tidak sendirian.\n\n"
            "📞 Indonesia:\n"
            "- Sejiwa 119 ext. 8\n"
            "- Hotline Kemenkes 1500-454\n\n"
            "Jika kamu mau, ceritakan apa yang membuatmu merasa sangat berat."
        )
        CACHE[message] = reply
        return {"reply": reply}

    memory = MEMORY.get(ip, "")
    knowledge = crawl_psychology(message)

    prompt = f"""
Kamu adalah AI konselor psikologi.
Gunakan bahasa lembut, empatik, tidak menghakimi.
Tidak mendiagnosis medis.

Riwayat singkat:
{memory}

Referensi psikologi:
{knowledge}

Pertanyaan:
{message}
"""

    try:
        reply = ai_generate(prompt)
    except:
        reply = (
            "Aku mungkin belum bisa memberi jawaban terbaik, "
            "tapi aku di sini untuk mendengarkanmu."
        )

    MEMORY[ip] = (memory + " " + message)[-1000:]
    CACHE[message] = reply
    return {"reply": reply}

@app.get("/")
def health():
    return {"status": "ok"}
