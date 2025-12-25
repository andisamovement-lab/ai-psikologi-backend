from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests, re, hashlib
from bs4 import BeautifulSoup
from cachetools import TTLCache

app = FastAPI()

# ================= CONFIG =================
HF_API = "https://api-inference.huggingface.co/models/google/gemma-7b-it"

CACHE = TTLCache(maxsize=1000, ttl=900)
SESSION = {}

PSY_SITES = [
    "https://www.psychologytoday.com",
    "https://www.verywellmind.com",
    "https://www.psychcentral.com",
    "https://www.healthline.com",
]

CRISIS = [
    "bunuh diri", "ingin mati", "suicide", "self harm", "melukai diri"
]

# ================= MIDDLEWARE =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CORE UTILS =================

def detect_language(text):
    return "en" if re.search(r"\b(what|why|how|should|i feel)\b", text.lower()) else "id"

def classify_intent(text):
    t = text.lower()

    if re.search(r"(apa yang harus|harus saya lakukan|what should i do)", t):
        return "ACTION_REQUEST"

    if re.search(r"(kenapa|mengapa|why)", t):
        return "WHY_QUESTION"

    if len(t.split()) <= 4:
        return "SHORT_EMOTION"

    if re.search(r"(lelah|capek|penuh|sedih|bingung|dikhianati|stres)", t):
        return "EMOTIONAL_STATEMENT"

    return "GENERAL"

def is_crisis(text):
    return any(k in text.lower() for k in CRISIS)

def ai_generate(prompt, temp=0.7, tokens=400):
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temp,
            "max_new_tokens": tokens,
            "return_full_text": False
        }
    }
    r = requests.post(HF_API, json=payload, timeout=60)
    if r.status_code != 200:
        raise Exception("HF overload")
    return r.json()[0]["generated_text"].strip()

# ================= LAYER 1 — FACT EXTRACTION =================

def extract_facts(user_text, history):
    prompt = f"""
Ringkas FAKTA OBJEKTIF dari percakapan ini.
JANGAN memberi saran.
JANGAN empati.

Riwayat:
{history}

Ucapan terbaru:
"{user_text}"

Fakta:
"""
    try:
        return ai_generate(prompt, temp=0.2, tokens=200)
    except:
        return ""

# ================= LAYER 2 — KNOWLEDGE RETRIEVAL =================

def retrieve_knowledge(facts):
    keywords = []
    f = facts.lower()

    if "dikhianati" in f:
        keywords += ["betrayal", "trust", "relationship trauma"]
    if "lelah" in f or "penuh pikiran" in f:
        keywords += ["mental overload", "stress", "emotional exhaustion"]
    if not keywords:
        keywords = ["emotional distress"]

    snippets = []

    for site in PSY_SITES:
        try:
            html = requests.get(site, timeout=4).text
            soup = BeautifulSoup(html, "html.parser")
            for p in soup.find_all("p"):
                txt = p.get_text().lower()
                if any(k in txt for k in keywords) and len(txt) > 120:
                    snippets.append(p.get_text().strip())
            if len(snippets) >= 2:
                break
        except:
            continue

    return " ".join(snippets[:2])

# ================= LAYER 3 — RESPONSE PLANNER =================

def build_prompt(user_text, facts, knowledge, intent, lang, last_reply):

    anti_repeat = f"""
ATURAN PENTING:
- JANGAN mengulang struktur atau kalimat berikut:
"{last_reply}"
"""

    if lang == "id":

        if intent == "ACTION_REQUEST":
            task = """
User meminta TINDAKAN.
WAJIB:
- Berikan 2–3 langkah konkret
- Tidak boleh hanya empati
- Tidak bertanya balik
"""

        elif intent in ["EMOTIONAL_STATEMENT", "SHORT_EMOTION"]:
            task = """
User menyatakan kondisi emosional.
WAJIB:
- Validasi spesifik
- Jelaskan penyebab psikologis singkat
- Boleh 1 pertanyaan relevan
"""

        elif intent == "WHY_QUESTION":
            task = """
User bertanya ALASAN.
WAJIB:
- Jelaskan sebab psikologis
- Jangan normatif
"""

        else:
            task = "Tanggapi secara bernalar dan relevan."

        return f"""
Kamu adalah AI konselor yang CERDAS dan KONTEKSTUAL.

FAKTA USER:
{facts}

PENGETAHUAN RELEVAN:
{knowledge}

INTENT USER:
{intent}

{anti_repeat}

INSTRUKSI:
{task}

Ucapan user:
"{user_text}"

Jawaban terbaik:
"""

    else:
        return f"""
You are an intelligent counselor.

FACTS:
{facts}

KNOWLEDGE:
{knowledge}

INTENT:
{intent}

Avoid repeating:
"{last_reply}"

User:
"{user_text}"

Answer intelligently:
"""

# ================= RESPONSE ENGINE =================

def generate_reply(user_text, session):
    lang = detect_language(user_text)
    intent = classify_intent(user_text)

    history = session.get("history", "")
    last_reply = session.get("last_reply", "")

    facts = extract_facts(user_text, history)
    knowledge = retrieve_knowledge(facts)

    prompt = build_prompt(
        user_text, facts, knowledge, intent, lang, last_reply
    )

    reply = ai_generate(prompt)

    # anti jawaban kosong / generik
    if len(reply) < 60:
        raise Exception("Weak response")

    return reply

# ================= API =================

@app.post("/chat")
async def chat(request: Request, data: dict):
    msg = data.get("message", "").strip()
    sid = request.client.host

    if not msg:
        return {"reply": "Aku di sini. Silakan ceritakan apa yang sedang kamu alami."}

    if is_crisis(msg):
        return {"reply":
            "Aku sangat menyesal kamu merasa seberat ini.\n\n"
            "📞 Indonesia:\n"
            "- Sejiwa 119 ext. 8\n"
            "- Kemenkes 1500-454\n\n"
            "Jika kamu mau, ceritakan apa yang membuatmu merasa seperti ini."
        }

    session = SESSION.get(sid, {"history": "", "last_reply": ""})

    try:
        reply = generate_reply(msg, session)
    except:
        reply = (
            "Situasi ini jelas tidak mudah. Kita bisa membaginya menjadi bagian yang lebih kecil agar lebih bisa ditangani."
        )

    session["history"] += f"\nUser: {msg}"
    session["last_reply"] = reply
    SESSION[sid] = session

    return {"reply": reply}

@app.get("/")
def health():
    return {"status": "ok"}
