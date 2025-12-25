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
]

# ================= MIDDLEWARE =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CORE LOGIC =================

def detect_language(text: str) -> str:
    if re.search(r"\b(the|and|is|are|what|how|why|i feel)\b", text.lower()):
        return "en"
    return "id"

def detect_intent(text: str) -> str:
    t = text.lower()

    if len(t.split()) <= 3:
        return "short_emotion"

    if any(w in t for w in ["apa", "bagaimana", "harus", "what", "how", "should"]):
        return "question"

    if any(w in t for w in ["lelah", "capek", "sedih", "bingung", "tired", "sad", "confused"]):
        return "emotional"

    return "general"

def is_crisis(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in CRISIS_KEYWORDS)

def crawl_psychology() -> str:
    results = []
    for site in PSY_SITES:
        try:
            r = requests.get(site, timeout=4)
            soup = BeautifulSoup(r.text, "html.parser")
            for p in soup.find_all("p")[:2]:
                txt = p.get_text().strip()
                if len(txt) > 120:
                    results.append(txt)
            if len(results) >= 2:
                break
        except:
            continue
    return " ".join(results)

def build_prompt(user_text: str, memory: str, lang: str, intent: str) -> str:
    base_rules = (
        "You are an intelligent, emotionally aware counselor.\n"
        "DO NOT repeat the same sentence structure as previous responses.\n"
        "Each response must move the conversation forward.\n"
        "Avoid generic phrases.\n"
    )

    knowledge = crawl_psychology()

    if lang == "en":
        if intent == "short_emotion":
            task = (
                "User expresses a short emotional state.\n"
                "Validate specifically and ask a deeper open-ended question."
            )
        elif intent == "question":
            task = (
                "User asks for guidance.\n"
                "Give 2–3 concrete options.\n"
                "Explain briefly.\n"
                "Remain empathetic."
            )
        else:
            task = (
                "User shares context.\n"
                "Reflect and respond meaningfully."
            )

        return f"""{base_rules}
Conversation memory:
{memory}

Psychology insight:
{knowledge}

Task:
{task}

User input:
"{user_text}"

Final answer (English):
"""

    else:
        if intent == "short_emotion":
            task = (
                "User menyampaikan emosi singkat.\n"
                "Validasi secara spesifik dan ajukan pertanyaan terbuka."
            )
        elif intent == "question":
            task = (
                "User meminta arahan.\n"
                "Berikan 2–3 langkah konkret.\n"
                "Tetap empatik dan realistis."
            )
        else:
            task = (
                "User sedang berbagi cerita.\n"
                "Tanggapi dengan refleksi dan relevansi."
            )

        return f"""{base_rules}
Riwayat percakapan:
{memory}

Wawasan psikologi:
{knowledge}

Instruksi:
{task}

Ucapan user:
"{user_text}"

Jawaban akhir (Bahasa Indonesia):
"""

def ai_generate(prompt: str) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.75,
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
    intent = detect_intent(user_text)

    prompt = build_prompt(user_text, memory, lang, intent)

    try:
        answer = ai_generate(prompt)
        if answer and len(answer.strip()) > 60:
            return answer
        raise Exception()
    except:
        if lang == "en":
            return "I want to understand you better. Could you share a little more about what’s weighing on you?"
        else:
            return "Aku ingin memahami kamu lebih dalam. Bisa ceritakan sedikit lagi apa yang paling membebani pikiranmu?"

# ================= API =================
@app.post("/chat")
async def chat(request: Request, data: dict):
    message = data.get("message", "").strip()
    ip = request.client.host

    if not message:
        return {"reply": "Aku di sini. Ceritakan apa yang sedang kamu rasakan."}

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
