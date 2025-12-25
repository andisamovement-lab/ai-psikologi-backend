from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os, re

# ================= CONFIG =================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.7,
        "max_output_tokens": 500,
    }
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= SESSION MEMORY =================
SESSIONS = {}
MAX_HISTORY = 6
MAX_SUMMARY_LEN = 800

# ================= UTIL =================
def detect_language(text):
    return "en" if re.search(r"\b(what|why|how|should|i feel)\b", text.lower()) else "id"

def classify_intent(text):
    t = text.lower()
    if "apa yang harus" in t or "harus saya lakukan" in t:
        return "ACTION"
    if "kenapa" in t or "mengapa" in t:
        return "WHY"
    return "STATEMENT"

# ================= SUMMARY =================
def summarize_history(history_text: str) -> str:
    prompt = f"""
Ringkas percakapan berikut menjadi konteks singkat.
Fokus pada:
- kondisi emosional user
- masalah utama
- hal penting yang sudah terjadi

Percakapan:
{history_text}

Ringkasan:
"""
    try:
        r = model.generate_content(prompt)
        return r.text.strip()[:MAX_SUMMARY_LEN]
    except:
        return history_text[-MAX_SUMMARY_LEN:]

# ================= PROMPT =================
def build_prompt(user_text, session, intent, lang):
    summary = session.get("summary", "")
    history = "\n".join(session.get("history", []))

    if lang == "id":
        if intent == "ACTION":
            task = "Berikan solusi konkret, jelas, dan realistis."
        elif intent == "WHY":
            task = "Jelaskan penyebabnya secara psikologis dan masuk akal."
        else:
            task = "Tanggapi dengan empati yang spesifik dan relevan."

        return f"""
Kamu adalah AI konselor cerdas seperti ChatGPT.

KONTEKS RINGKAS:
{summary}

RIWAYAT TERAKHIR:
{history}

UCAPAN USER:
"{user_text}"

TUGAS:
{task}

Jawaban terbaik:
"""
    else:
        return f"""
You are an intelligent counselor.

Context:
{summary}

Conversation:
{history}

User:
"{user_text}"

Respond clearly and helpfully:
"""

# ================= CORE =================
def generate_reply(user_text, session):
    lang = detect_language(user_text)
    intent = classify_intent(user_text)

    prompt = build_prompt(user_text, session, intent, lang)

    response = model.generate_content(prompt)
    if not response or not response.text:
        raise Exception("Empty response")

    return response.text.strip()

# ================= API =================
@app.post("/chat")
async def chat(request: Request, data: dict):
    user_text = data.get("message", "").strip()
    sid = request.client.host

    if not user_text:
        return {"reply": "Aku di sini. Silakan ceritakan apa yang sedang kamu alami."}

    session = SESSIONS.get(sid, {"history": [], "summary": ""})

    reply = generate_reply(user_text, session)

    # simpan history
    session["history"].append(f"User: {user_text}")
    session["history"].append(f"AI: {reply}")

    # summarization jika terlalu panjang
    if len(session["history"]) > MAX_HISTORY:
        combined = "\n".join(session["history"])
        session["summary"] = summarize_history(combined)
        session["history"] = []

    SESSIONS[sid] = session

    return {"reply": reply}

@app.get("/")
def health():
    return {"status": "ok"}
