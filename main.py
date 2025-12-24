from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests, time, re
from bs4 import BeautifulSoup
from cachetools import TTLCache

app = FastAPI()

# ================= CONFIG =================
HF_API = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

CACHE = TTLCache(maxsize=500, ttl=900)      # cache jawaban
MEMORY = {}                                 # memori percakapan per IP

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
    "https://www.mind.org.uk",
    "https://www.samhsa.gov",
    "https://www.ncbi.nlm.nih.gov",
    "https://www.frontiersin.org",
    "https://www.mentalhealth.org.uk",
    "https://www.anxietycanada.com",
    "https://www.verywellhealth.com",
    "https://www.psychologytools.com",
    "https://www.goodtherapy.org",
    "https://www.mentalhelp.net",
    "https://www.therapistaid.com",
    "https://www.psychology.org",
    "https://www.medicalnewstoday.com",
    "https://www.sciencedaily.com",
    "https://www.betterhelp.com",
    "https://www.talkspace.com",
    "https://www.psychologyresearch.com"
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

def crawl_psychology(query: str) -> str:
    results = []
    for site in PSY_SITES:
        try:
            r = requests.get(site, timeout=4)
            soup = BeautifulSoup(r.text, "html.parser")
            for p in soup.find_all("p")[:5]:
                text = p.get_text().strip()
                if len(text) > 120:
                    results.append(text)
            if len(results) >= 5:
                break
        except:
            continue
    return " ".join(results[:5])

def cbt_act_prompt(user_text: str) -> str:
    return f"""
Gunakan pendekatan CBT dan ACT:
- Validasi emosi tanpa menyangkal
- Identifikasi pikiran otomatis (CBT)
- Normalisasi pengalaman manusiawi
- Dorong penerimaan & nilai hidup (ACT)
- Ajukan 1 pertanyaan reflektif di akhir

Ucapan klien:
"{user_text}"
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
    r = requests.post(HF_API, json=payload, timeout=45)
    if r.status_code != 200:
        raise Exception("AI overload")
    return r.json()[0]["generated_text"]

def smart_answer(user_text: str, memory: str) -> str:
    knowledge = crawl_psychology(user_text)

    prompt = f"""
Kamu adalah konselor psikologi profesional.
Gunakan bahasa lembut, empatik, tidak menghakimi.
Jangan memberi diagnosis medis.

Riwayat percakapan singkat:
{memory}

Pengetahuan psikologi:
{knowledge}

Instruksi konseling:
{cbt_act_prompt(user_text)}

Jawaban konselor:
"""

    try:
        ai = ai_generate(prompt)
        if ai and len(ai.strip()) > 50:
            return ai
        raise Exception()
    except:
        # 🔥 fallback cerdas dari hasil crawl
        if knowledge:
            return (
                "Perasaan yang kamu alami sangat manusiawi dan valid.\n\n"
                f"{knowledge[:800]}\n\n"
                "Dari semua hal ini, bagian mana yang paling kamu rasakan saat ini?"
            )
        else:
            return (
                "Aku bisa merasakan ada sesuatu yang cukup menguras emosimu. "
                "Kalau kamu mau, kita bisa bahas perlahan satu hal yang paling berat."
            )

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
