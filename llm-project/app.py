"""FastAPI application for LawLens, a local Indian legal information assistant."""

from contextlib import asynccontextmanager
import json
import logging
from typing import Any

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import DATA_DIR, INCIDENTS_FILE, OLLAMA_HOST, OLLAMA_MODEL, STATIC_DIR
from rag import LegalDocumentIndex
from safety import analyze_scam_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
index = LegalDocumentIndex()
client = ollama.Client(host=OLLAMA_HOST)


@asynccontextmanager
async def lifespan(_: FastAPI):
    index.rebuild()
    yield


app = FastAPI(title="LawLens - Local Indian Legal Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class ScamRequest(BaseModel):
    text: str = Field(min_length=3, max_length=10000)


class IncidentRequest(BaseModel):
    incident_category: str = Field(default="", max_length=100)
    date_time: str = Field(default="", max_length=100)
    amount: str = Field(default="", max_length=50)
    transaction_reference: str = Field(default="", max_length=200)
    provider: str = Field(default="", max_length=200)
    suspect_identifier: str = Field(default="", max_length=300)
    evidence_checklist: list[str] = Field(default_factory=list, max_length=30)
    complaint_status: str = Field(default="", max_length=100)
    notes: str = Field(default="", max_length=3000)


def ollama_status() -> dict[str, Any]:
    try:
        response = client.list()
        models = response.get("models", []) if isinstance(response, dict) else response.models
        names = []
        for model in models:
            name = model.get("name") if isinstance(model, dict) else getattr(model, "model", None)
            if name:
                names.append(name)
        available = any(name == OLLAMA_MODEL or name.split(":")[0] == OLLAMA_MODEL.split(":")[0]
                        for name in names)
        return {"connected": True, "model": OLLAMA_MODEL, "model_available": available,
                "models": names, "message": None if available else
                f"Model {OLLAMA_MODEL} is missing. Run: ollama pull {OLLAMA_MODEL}"}
    except Exception:
        return {"connected": False, "model": OLLAMA_MODEL, "model_available": False,
                "models": [], "message":
                "Ollama is unavailable. Start Ollama, then run "
                f"'ollama pull {OLLAMA_MODEL}' in PowerShell."}


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    index.ensure_current()
    return {"status": "ok", "ollama": ollama_status(), "index": index.status(),
            "local_only": True}


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    index.ensure_current()
    return index.status()


@app.post("/api/documents/reindex")
def reindex() -> dict[str, Any]:
    return index.rebuild()


@app.post("/api/analyze-scam")
def analyze_scam(request: ScamRequest) -> dict[str, Any]:
    return analyze_scam_text(request.text)


@app.get("/api/incidents")
def list_incidents() -> dict[str, Any]:
    if not INCIDENTS_FILE.exists():
        return {"incidents": [], "local_file": str(INCIDENTS_FILE)}
    try:
        return {"incidents": json.loads(INCIDENTS_FILE.read_text(encoding="utf-8")),
                "local_file": str(INCIDENTS_FILE)}
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Could not read incident notes: %s", exc)
        raise HTTPException(status_code=500, detail="Local incident notes could not be read.")


@app.post("/api/incidents")
def save_incident(request: IncidentRequest) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        incidents = json.loads(INCIDENTS_FILE.read_text(encoding="utf-8")) \
            if INCIDENTS_FILE.exists() else []
        incidents.append(request.model_dump())
        INCIDENTS_FILE.write_text(json.dumps(incidents, indent=2), encoding="utf-8")
        return {"saved": True, "local_file": str(INCIDENTS_FILE)}
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Could not save incident notes: %s", exc)
        raise HTTPException(status_code=500, detail="Local incident notes could not be saved.")


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Please enter a legal question.")
    intent = index.detect_intent(question)
    results = index.retrieve(question)
    if not results:
        return {
            "answer": "I could not find sufficiently relevant material in the documents currently loaded. "
                      "Please add an appropriate Indian-law source or rephrase the question.",
            "sources": [], "supported": False,             "retrieved_count": 0, "intent": intent,
        }
    status = ollama_status()
    if not status["connected"] or not status["model_available"]:
        raise HTTPException(status_code=503, detail=status["message"])
    context = index.context_for(results)
    try:
        response = client.chat(model=OLLAMA_MODEL, messages=[
            {"role": "system", "content":
             "You are LawLens, a cautious Indian legal information assistant. "
             "Use only the supplied retrieved passages for source-backed legal claims. "
             "Clearly distinguish what a source says from general explanation. "
             "If evidence is insufficient or conflicting, say so. Never invent citations, "
             "sections, cases, quotations, amendments, or page numbers. Do not claim that "
             "legislation is current unless verified. Answer questions about any uploaded "
             "Indian-law or cybercrime document, not only fraud. Give general legal information, not "
             "personalized legal advice, and mention source filenames and PDF pages. "
             "Answer the user's actual question first, then give urgent next steps before background. "
             "Only when the detected intent is recent_financial_fraud, begin with a concise numbered list: call 1930 promptly, "
             "report at the official portal, and contact the affected bank through its official channel. "
             "For other intents, do not add cyber-fraud emergency advice unless the retrieved passages support it. "
             "For legal provisions, explain the supported text, plain-English meaning, missing facts, and uncertainty about currency. "
             "Then give evidence-preservation and account-safety guidance only when relevant. "
             "Never promise recovery of money. Do not state that evidence preservation is legally "
             "required unless the passage explicitly says so. For recent financial fraud, the "
             "trusted curated guidance is the priority and unrelated passages cannot override it."},
            {"role": "user", "content":
             f"Question: {question}\nDetected intent: {intent}\n\n"
             f"Retrieved passages:\n{context}\n\n"
             "Answer clearly. Ground every important claim in a source passage. "
             "For each important step, cite only the filename and page/section whose excerpt supports it. "
             "If the sources do not support a claim, state the limitation instead of guessing."},
        ])
        answer = response["message"]["content"]
    except Exception:
        logger.exception("Ollama request failed")
        raise HTTPException(status_code=503,
                            detail="Ollama could not generate an answer. Check that it is running and the model is installed.")
    return {
        "answer": answer, "supported": True,
        "sources": [item.as_source(score, intent) for item, score in results],
        "retrieved_count": len(results),
        "intent": intent,
    }
