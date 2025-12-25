from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests, re
from bs4 import BeautifulSoup
from cachetools import TTLCache

app = FastAPI()

# ================= CONFIG =================
HF_API = "https://api-inference.huggingface.co/models/google/gemma-7b-it"

CACHE = TTLCache(maxsize=500, ttl=900)
MEMORY = {}

PSY_SITES = [
    "https://www.psychologytoday.com",
    "https://www.verywellmind.com",
    "https://www.psychcentral.com",
    "https://positivepsychology.com",
    "https://www.healthline.com",
]

CRISIS_KEYWORDS = [
    "bunuh diri", "ingin mati", "tidak ingin hidup",
    "suicide", "self harm", "melukai diri"
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
    if re.search(r"\b(the|and|is|are|what|how|why|i feel)\b", text.lower()):
        return "en"
    return "id"

def is_crisis(text: str) -> bool:
    return any(k in text.lower() for k in CRISIS_KEYWORDS)

def ai_generate(prompt: str, max_tokens=350, temperature=0.7) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "return_full_text": False
        }
    }
    r = requests.post(HF_API, json=payload, timeout=60)
    if r.status_code != 200:
        raise Exception("AI overload")
    data = r.json()
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    raise Exception("Invalid response")

# ================= LAYER 1: FACT EXTRACTION =================

def extract_facts(user_text: str, memory: str) -> str:
    prompt = f"""
Ekstrak FAKTA penting dari percakapan berikut.
HANYA ringkasan fakta, TANPA saran atau empati.

Percakapan sebelumnya:
{memory}

Ucapan terbaru user:
"{user_text}"

Fakta penting:
"""
    try:
        return ai_generate(prompt, max_tokens=200, temperature=0.3)
    except:
        return ""

# ================= LAYER 2: KNOWLEDGE RETRIEVAL =================

def retrieve_psychology_context(facts: str) -> str:
    keywords = []
    f = facts.lower()

    if "dikhianati" in f or "betrayal" in f:
        keywords += ["betrayal", "trust", "infidelity"]
    if "lelah" in f or "emotion" in f:
        keywords += ["emotional exhaustion", "stress"]
    if not keywords:
        keywords = ["emotional pain", "relationships"]

    snippets = []

    for site in PSY_SITES:
        try:
            r = requests.get(site, timeout=4)
            soup = BeautifulSoup(r.text, "html.parser")
            for p in soup.find_all("p"):
                t = p.get_text().lower()
                if any(k in t for k in keywords) and len(t) > 150:
                    snippets.append(p.get_text().strip())
            if len(snippets) >= 2:
                break
        except:
            continue

    return " ".join(snippets[:2])

# ================= LAYER 3: REASONED RESPONSE =================

def build_reasoned_prompt(user_text, facts, knowledge, lang):
    if lang == "id":
        return f"""
Kamu adalah konselor AI yang sangat cerdas dan bernalar.

FAKTA USER:
{facts}

PENGETAHUAN PSIKOLOGI RELEVAN:
{knowledge}

ATURAN WAJIB:
- Akui emosi secara SPESIFIK (bukan umum)
- Kaitkan jawaban dengan fakta (misal pengkhianatan)
- Jelaskan secara psikologis singkat & jelas
- Jangan mengulang kalimat generik
- Jangan bertanya hal yang sama berulang
- Percakapan harus maju

Ucapan user:
"{user_text}"

Jawaban cerdas Bahasa Indonesia:
"""
    else:
        return f"""
You are an emotionally intelligent counselor.

USER FACTS:
{facts}

RELEVANT PSYCHOLOGY:
{knowledge}

RULES:
- Be specific, not generic
- Connect emotions to facts
- Explain briefly and clearly
- Move the conversation forward

User input:
"{user_text}"

Intelligent answer:
"""

def generate_smart_reply(user_text: str, memory: str) -> str:
    lang = detect_language(user_text)

    facts = extract_facts(user_text, memory)
    knowledge = retrieve_psychology_context(facts)
    prompt = build_reasoned_prompt(user_text, facts, knowledge, lang)

    try:
        return ai_generate(prompt)
    except:
        return (
            "Aku mendengarmu. Situasi ini tampaknya cukup menyakitkan. "
            "Kita bisa membahasnya perlahan, satu bagian yang paling berat dulu."
            if lang == "id"
            else
            "I hear you. This situation sounds painful. We can unpack it slowly, one part at a time."
        )

# ================= API =================

@app.post("/chat")
async def chat(request: Request, data: dict):
    message = data.get("message", "").strip()
    ip = request.client.host

    if not message:
        return {"reply": "Aku di sini. Ceritakan apa yang sedang kamu alami."}

    if message in CACHE:
        return {"reply": CACHE[message]}

    if is_crisis(message):
        reply = (
            "Aku sangat menyesal kamu merasa seberat ini.\n\n"
            "📞 Bantuan Indonesia:\n"
            "- Sejiwa 119 ext. 8\n"
            "- Kemenkes 1500-454\n\n"
            "Jika kamu mau, ceritakan apa yang membuat rasa ini muncul."
        )
        CACHE[message] = reply
        return {"reply": reply}

    memory = MEMORY.get(ip, "")
    reply = generate_smart_reply(message, memory)

    MEMORY[ip] = (memory + "\n" + message)[-2000:]
    CACHE[message] = reply
    return {"reply": reply}

@app.get("/")
def health():
    return {"status": "ok"}
