# LawLens — Your Local Indian Legal Assistant

LawLens is a small, local-only legal information assistant. It reads PDF and TXT files from `legal_docs`, retrieves relevant passages with TF-IDF, and sends only those passages to a local Ollama model. It does **not** use a cloud AI API.

> **Important:** This is general legal information, not legal advice. The supplied documents can be historical or incomplete. LawLens does not verify that legislation is current, does not invent citations, and cannot replace a qualified lawyer.

## Quick start on Windows PowerShell

Open PowerShell in this project folder:

```powershell
cd "C:\Users\MUHAMMED KHALANDAR\OneDrive\Desktop\MLH\llm-project"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` and activate again. You can also use `.venv\Scripts\python.exe` directly without activating.

Install Ollama from [ollama.com](https://ollama.com), then verify it:

```powershell
ollama --version
ollama pull qwen2.5:3b
ollama list
```

`qwen2.5:3b` must appear in `ollama list`. Start the Ollama desktop application before chatting. Downloading packages and the model needs internet access initially; normal document processing and inference run locally afterward.

Put the Indian Contract Act PDF and any other `.pdf` or `.txt` files in `legal_docs`. The included `contrant_act.pdf` is retained as the initial demo document. A PDF is indexed page-by-page, so source cards show its PDF page. TXT files show a chunk identifier instead.

The project also includes `emergency_financial_fraud_guidance.txt`, a deliberately small curated source for urgent UPI/digital-payment fraud. Its 1930 and portal instruction cites the official National Cyber Crime Reporting Portal (`https://www.cybercrime.gov.in/UploadMedia/instructions_citizenreportingcyberfrauds.pdf`, verified 2026-10-11); its banking safety sections identify their RBI source URL in the file. This is guidance, not a recovery promise or legal advice.

`financial_fraud_first_steps.txt` is the canonical source used when a question describes money stolen, deducted, unauthorised transactions, UPI scams, or a hacked bank account. It records the official NCRP and RBI URLs and verification date alongside each section. LawLens places its report-first and bank-contact passages before generic portal/manual passages, removes same-page duplicates, and avoids money-mule passages unless the question asks about money mules.

## LawLens 2.0 features

- Explainable cybercrime intent routing for financial fraud, card fraud, phishing/fake KYC, OTP/account takeover, shopping/payment scams, investment/crypto scams, job/task scams, social-media impersonation, blackmail, cyberstalking, identity theft, complaint checking, evidence preservation, prevention, legal provisions, money mules, and out-of-scope questions.
- A local scam-message checker at the bottom of the page. It examines pasted text only; it does not open URLs, download files, or send text to a remote service. Warning signals are indicators, not proof that a message is fraudulent.
- A local incident-notes panel. Notes are saved to `data/incidents.json` (ignored by Git). Never store passwords, OTPs, UPI PINs, or banking credentials.
- Source cards show category, excerpt, supported-claim description, filename/page or chunk, and official provenance when available. Retrieval scores are not confidence or correctness probabilities.

The current lightweight intent classifier is rule-based and explainable. It is not a trained classifier. Ambiguous questions use general retrieval; users should provide more detail when the scenario is unclear. No automatic government-portal filing, bank contact, account freeze, or money recovery is performed.

## Architecture

```text
Browser UI
  ├─ chat / scam checker / local incident notes
  ↓
FastAPI (app.py)
  ├─ intent routing + safe local APIs
  ├─ Ollama client (127.0.0.1:11434)
  ↓
DocumentIndex (rag.py)
  ├─ PDF/TXT extraction with page metadata
  ├─ section-aware chunks + TF-IDF retrieval
  ├─ intent boosts, trusted emergency source, diversity filters
  ↓
qwen2.5:3b (local only)
```

## Four-minute demo

1. Start Ollama and LawLens, then show the document library.
2. Ask: `someone stole 1000 rs from my account`. Show the report-first answer and curated source.
3. Ask: `How do I preserve WhatsApp evidence?` and show evidence-oriented PDF pages.
4. Paste an urgent fake-KYC message into the scam checker; show warning signals without opening its URL.
5. Save a small incident note, point out that it is stored locally, and show the legal/privacy disclaimer.
6. Ask `What is a money mule?` to demonstrate intent-specific RBI retrieval.

## Evaluation

`tests/evaluation_queries.json` contains 30 representative queries across supported intents. Run the executable evaluation and unit suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The evaluation checks the lightweight classifier against expected labels; retrieval quality still depends on the documents available in `legal_docs`, and the model can only ground claims supported by retrieved passages.

Start the server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Open **http://127.0.0.1:8000** in your browser. The document sidebar automatically notices changed files; use the refresh button to reindex explicitly.

## Testing

Tests do not require Ollama:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover chunking, TXT and corrupt-PDF handling, empty files, retrieval thresholds, unrelated questions, and FastAPI validation/health behavior. The chat endpoint still needs Ollama for a real answer.

## Troubleshooting

- **`No module named ...`:** activate `.venv` or run the exact `.venv\Scripts\python.exe` command shown above, then run `pip install -r requirements.txt`.
- **Wrong Python in VS Code:** select `.venv\Scripts\python.exe` with `Python: Select Interpreter`.
- **Ollama unavailable:** start the Ollama application, then check `ollama list`. LawLens reports the exact `ollama pull qwen2.5:3b` command when the model is missing.
- **No documents:** confirm the files are directly inside `legal_docs`, not a nested folder, and click refresh.
- **PDF says no extractable text:** it is probably scanned/image-only. LawLens reports this rather than silently ignoring it; OCR the PDF before adding it again.
- **Unsupported question:** LawLens returns an explicit insufficient-material answer when no chunk meets the configurable retrieval threshold. Similarity scores are retrieval signals, not probabilities of correctness.

## Architecture

`app.py` provides FastAPI endpoints and serves the static UI. `rag.py` discovers files, extracts PDF pages with `pypdf`, creates overlapping chunks, and builds a scikit-learn TF-IDF index. For each question it applies a configurable top-k (`LAW_LENS_TOP_K`, default 5) and threshold (`LAW_LENS_RETRIEVAL_THRESHOLD`, default 0.08), limits duplicate chunks from one file, and caps the prompt context. Ollama receives only the question and retrieved passages. This boundary can later be replaced with local semantic embeddings without changing the API.

Before ranking, LawLens classifies the question into a small intent set (recent financial fraud, reporting, evidence preservation, legal provisions, prevention, complaint checking, or general). Recent financial-fraud intent boosts the curated emergency guide and immediate-reporting section, while suppressing unrelated WhatsApp-export passages. Retrieved sources include an excerpt, supported-claim description, section, and provenance; cosine values are retrieval scores, not correctness probabilities.

Useful tuning examples:

```powershell
$env:LAW_LENS_RETRIEVAL_THRESHOLD = "0.12"
$env:LAW_LENS_TOP_K = "4"
```

Higher thresholds are stricter and produce more “insufficient material” responses; lower thresholds retrieve more loosely related text. Tune with the actual documents rather than treating a score as a correctness percentage.

## Demo flow

1. Start Ollama and the FastAPI server.
2. Show `contrant_act.pdf` in the document library.
3. Ask “What makes an agreement a contract?” and point to the returned PDF page sources.
4. Ask about an unrelated topic, such as a medical diagnosis, and show the insufficient-material response.
5. Mention that the answer was generated by `qwen2.5:3b` running on the local machine.
