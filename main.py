from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests, re
from bs4 import BeautifulSoup
from cachetools import TTLCache

app = FastAPI()

# ================= CONFIG =================
# Gemma open-source model (via HF inference)
HF_API = "https://api-inference.huggingface.co/models/google/gemma-7b-it"

CACHE = TTLCache(maxsize=500, ttl=900)
MEMORY = {}

CRISIS_KEYWORDS = [
    "bunuh diri", "ingin mati", "tidak ingin hidup",
    "suicide", "self harm", "melukai diri"
]

PSY_SITES = [
    "https://www.psychologytoday.com",
    "https://www.verywellmind.com",
    "https://www.psychcentral.com",
    "https://positivepsychology.com",
    "https://www.healthline.com",
    "https://www.helpguide.org",
    "https://www.nimh.nih.gov",
    "https://www.apa.org",
    "https://www.ncbi.nlm.nih.gov",
    "https://www.frontiersin.org",
]

# ================= MIDDLEWARE =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= UTIL =================
def detect_language(text: str) -> str:
    if re.search(r"[a-zA-Z]", text) and not re.search(r"[à-ÿ]", text):
        if any(w in text.lower() for w in ["the", "and", "is", "are", "i feel"]):
            return "en"
    return "id"

def is_crisis(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in CRISIS_KEYWORDS)

def crawl_psychology() -> str:
    results = []
    for site in PSY_SITES:
        try:
            r = requests.get(site, timeout=4)
            soup = BeautifulSoup(r.text, "html.parser")
            for p in soup.find_all("p")[:3]:
                txt = p.get_text().strip()
                if len(txt) > 120:
                    results.append(txt)
            if len(results) >= 3:
                break
        except:
            continue
    return " ".join(results[:3])

def build_prompt(user_text: str, memory: str, lang: str) -> str:
    if lang == "en":
        return f"""
You are a professional mental health counselor.
Be empathetic, calm, and non-judgmental.
Do NOT give medical diagnosis.

Conversation memory:
{memory}

Psychology reference:
{crawl_psychology()}

User message:
"{user_text}"

Respond in English with warmth and clarity.
Ask one gentle reflective question at the end.
"""
    else:
        return f"""
Kamu adalah konselor kesehatan mental profesional.
Gunakan bahasa Indonesia yang lembut, empatik, dan tidak menghakimi.
JANGAN memberi diagnosis medis.

Riwayat percakapan:
{memory}

Referensi psikologi:
{crawl_psychology()}

Ucapan pengguna:
"{user_text}"

Berikan jawaban yang menenangkan dan relevan.
Ajukan satu pertanyaan reflektif di akhir.
"""

def ai_generate(prompt: str) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.6,
            "max_new_tokens": 350,
            "return_full_text": False
        }
    }

    r = requests.post(HF_API, json=payload, timeout=60)

    if r.status_code != 200:
        raise Exception("AI overload")

    data = r.json()

    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]

    raise Exception("Invalid AI response")

def smart_answer(user_text: str, memory: str) -> str:
    lang = detect_language(user_text)
    prompt = build_prompt(user_text, memory, lang)

    try:
        answer = ai_generate(prompt)
        if answer and len(answer.strip()) > 40:
            return answer
        raise Exception()
    except:
        if lang == "en":
            return (
                "I can sense that something feels heavy for you right now. "
                "If you want, we can talk about it slowly, one step at a time."
            )
        else:
            return (
                "Aku bisa merasakan ada hal yang cukup berat untukmu saat ini. "
                "Jika kamu mau, kita bisa membahasnya perlahan."
            )

# ================= API =================
@app.post("/chat")
async def chat(request: Request, data: dict):
    message = data.get("message", "").strip()
    ip = request.client.host

    if not message:
        return {"reply": "Aku di sini. Silakan ceritakan apa yang kamu rasakan."}

    if message in CACHE:
        return {"reply": CACHE[message]}

    if is_crisis(message):
        reply = (
            "Aku sangat menyesal kamu merasa seberat ini. Kamu tidak sendirian.\n\n"
            "📞 Indonesia:\n"
            "- Sejiwa 119 ext. 8\n"
            "- Kemenkes 1500-454\n\n"
            "Jika kamu mau, ceritakan apa yang membuat semuanya terasa sangat berat."
        )
        CACHE[message] = reply
        return {"reply": reply}

    memory = MEMORY.get(ip, "")
    reply = smart_answer(message, memory)

    MEMORY[ip] = (memory + " " + message)[-1500:]
    CACHE[message] = reply
    return {"reply": reply}

@app.get("/")
def health():
    return {"status": "ok"}
