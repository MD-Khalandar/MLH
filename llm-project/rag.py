"""Local document ingestion and TF-IDF retrieval for LawLens."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import logging
import re
from typing import Any

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    LEGAL_DOCS_DIR,
    MAX_CONTEXT_CHARS,
    RETRIEVAL_THRESHOLD,
    TOP_K,
    EMERGENCY_GUIDANCE_SOURCE,
    FINANCIAL_FIRST_STEPS_SOURCE,
    NCRP_SOURCE_URL,
    NCRP_VERIFIED_DATE,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    source: str
    page: int | None
    chunk_id: str
    text: str
    file_type: str
    section_heading: str | None = None
    trusted: bool = False
    source_url: str | None = None
    verified_date: str | None = None

    def claim_for(self, intent: str) -> str:
        if self.section_heading:
            return f"Supports the guidance in the section “{self.section_heading}”."
        if intent == "preserving_digital_evidence":
            return "Provides passage-level information about preserving or submitting digital evidence."
        if intent == "reporting_cybercrime":
            return "Provides passage-level information about reporting cybercrime."
        if intent == "legal_provisions":
            return "Provides passage-level legal or statutory information, if present in the excerpt."
        return "Provides the specific information shown in the excerpt; it is not evidence for claims outside that passage."

    @property
    def category(self) -> str:
        name = self.source.casefold()
        if self.trusted or "fraud" in name or "beware" in name:
            return "emergency guidance"
        if "manual" in name or "report" in name:
            return "official reporting instructions"
        return "uploaded legal/cybercrime source"

    def as_source(self, score: float, intent: str) -> dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "score": round(score, 4),
            "file_type": self.file_type,
            "excerpt": self.text[:500],
            "claim_supported": self.claim_for(intent),
            "trusted_guidance": self.trusted,
            "source_url": self.source_url,
            "verified_date": self.verified_date,
            "category": self.category,
        }


@dataclass
class FileStatus:
    name: str
    file_type: str
    status: str
    chunks: int = 0
    pages: int | None = None
    message: str | None = None


def split_into_chunks(text: str, size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into bounded overlapping chunks without infinite loops."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap must be smaller")
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in
                  re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip()
        if current and len(candidate) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n{paragraph}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    if any(len(chunk) > size for chunk in chunks):
        flattened = " ".join(paragraphs)
        chunks = []
        start = 0
        while start < len(flattened):
            end = min(start + size, len(flattened))
            chunks.append(flattened[start:end].strip())
            if end >= len(flattened):
                break
            start = end - overlap
    return chunks


def _file_fingerprint(directory: Path) -> str:
    entries = []
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() in {".pdf", ".txt"} and path.is_file():
            stat = path.stat()
            entries.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("|".join(entries).encode()).hexdigest()


class LegalDocumentIndex:
    def __init__(self, docs_dir: Path = LEGAL_DOCS_DIR) -> None:
        self.docs_dir = docs_dir
        self.chunks: list[DocumentChunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: Any = None
        self.file_statuses: list[FileStatus] = []
        self.errors: list[str] = []
        self.fingerprint = ""
        self.last_indexed: str | None = None

    def _load_pdf(self, path: Path) -> tuple[list[DocumentChunk], FileStatus]:
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                try:
                    if reader.decrypt("") == 0:
                        return [], FileStatus(path.name, "PDF", "error",
                                              message="Encrypted PDF could not be opened.")
                except Exception:
                    return [], FileStatus(path.name, "PDF", "error",
                                          message="Encrypted PDF could not be opened.")
            chunks: list[DocumentChunk] = []
            nonempty_pages = 0
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                page_chunks = split_into_chunks(text)
                if page_chunks:
                    nonempty_pages += 1
                for number, chunk in enumerate(page_chunks, start=1):
                    chunks.append(DocumentChunk(
                        path.name, page_number, f"p{page_number}-c{number}",
                        chunk, "PDF",
                    ))
            if not chunks:
                return [], FileStatus(
                    path.name, "PDF", "warning", pages=len(reader.pages),
                    message="No extractable text found; this may be a scanned PDF and may need OCR.",
                )
            return chunks, FileStatus(path.name, "PDF", "indexed",
                                      len(chunks), len(reader.pages),
                                      f"{nonempty_pages} page(s) with text")
        except Exception as exc:
            logger.warning("Could not process PDF %s: %s", path, exc)
            return [], FileStatus(path.name, "PDF", "error",
                                  message=f"Could not process PDF: {exc}")

    def _load_txt(self, path: Path) -> tuple[list[DocumentChunk], FileStatus]:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                return [], FileStatus(path.name, "TXT", "error", message=str(exc))
        except OSError as exc:
            return [], FileStatus(path.name, "TXT", "error", message=str(exc))
        trusted = path.name in {EMERGENCY_GUIDANCE_SOURCE, FINANCIAL_FIRST_STEPS_SOURCE}
        section_heading: str | None = None
        source_url = NCRP_SOURCE_URL if trusted else None
        verified_date = NCRP_VERIFIED_DATE if trusted else None
        pieces: list[tuple[str, str | None, str | None, str | None]] = []
        current: list[str] = []
        for line in text.splitlines():
            heading_match = re.match(r"^\[SECTION:\s*(.+?)\]\s*$", line.strip(), re.I)
            if heading_match:
                if current:
                    pieces.extend(
                        (piece, section_heading, source_url, verified_date)
                        for piece in split_into_chunks("\n\n".join(current))
                    )
                    current = []
                section_heading = heading_match.group(1)
                continue
            source_match = re.match(r"^Source URL:\s*(\S+)", line.strip(), re.I)
            if source_match:
                source_url = source_match.group(1)
                continue
            date_match = re.match(r"^Verified date:\s*(\S+)", line.strip(), re.I)
            if date_match:
                verified_date = date_match.group(1)
                continue
            current.append(line)
        if current:
            pieces.extend(
                (piece, section_heading, source_url, verified_date)
                for piece in split_into_chunks("\n\n".join(current))
            )
        if not pieces:
            return [], FileStatus(path.name, "TXT", "warning",
                                  message="The text file is empty.")
        chunks = [
            DocumentChunk(path.name, None, f"c{number}", piece, "TXT",
                          heading, trusted, piece_url, piece_date)
            for number, (piece, heading, piece_url, piece_date)
            in enumerate(pieces, start=1)
        ]
        return chunks, FileStatus(path.name, "TXT", "indexed", len(chunks))

    def rebuild(self) -> dict[str, Any]:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[DocumentChunk] = []
        statuses: list[FileStatus] = []
        self.errors = []
        paths = sorted(
            (p for p in self.docs_dir.iterdir()
             if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}),
            key=lambda p: p.name.lower(),
        )
        for path in paths:
            loaded, status = self._load_pdf(path) if path.suffix.lower() == ".pdf" \
                else self._load_txt(path)
            chunks.extend(loaded)
            statuses.append(status)
            if status.status == "error":
                self.errors.append(f"{status.name}: {status.message}")
        self.chunks = chunks
        self.file_statuses = statuses
        self.fingerprint = _file_fingerprint(self.docs_dir)
        self.last_indexed = __import__("datetime").datetime.now().isoformat(timespec="seconds")
        if chunks:
            self.vectorizer = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2), stop_words="english",
            )
            self.matrix = self.vectorizer.fit_transform([item.text for item in chunks])
        else:
            self.vectorizer = None
            self.matrix = None
        return self.status()

    def ensure_current(self) -> None:
        current = _file_fingerprint(self.docs_dir)
        if current != self.fingerprint:
            self.rebuild()

    @staticmethod
    def detect_intent(question: str) -> str:
        text = question.casefold()
        if re.search(r"\b(money mule|money-mule|mule account)\b", text):
            return "money_mule"
        if re.search(r"\b(check|track|status|existing|acknowledg).{0,30}\b(complaint|report)\b", text):
            return "checking_complaint"
        if re.search(r"\b(credit card|debit card|card fraud|card stolen|card transaction)\b", text):
            return "card_fraud"
        if re.search(r"\b(phishing|fake kyc|kyc update|malicious link|suspicious link|smishing)\b", text):
            return "phishing"
        if re.search(r"\b(impersonat|social media|instagram|facebook)\b", text):
            return "social_media_impersonation"
        if re.search(r"\b(account(?: was)? hacked|hacked account)\b", text):
            return "otp_account_takeover"
        if re.search(r"\b(otp|one[- ]time password|account takeover|hacked account|account(?: was)? hacked|password stolen)\b", text):
            return "otp_account_takeover"
        if re.search(r"\b(online shopping|seller scam|payment scam|fake product|marketplace scam)\b", text):
            return "shopping_payment_scam"
        if re.search(r"\b(investment|trading|crypto|cryptocurrency|stock tip|ponzi)\b", text):
            return "investment_scam"
        if re.search(r"\b(job scam|task scam|work from home|employment scam|fake job|job offer)\b", text):
            return "job_task_scam"
        if re.search(r"\b(sextortion|blackmail|threat|intimate image)\b", text):
            return "blackmail_threat"
        if re.search(r"\b(cyberstalk\w*|cyber stalk|harass\w*|abusive messages?)\b", text):
            return "cyberstalking_harassment"
        if re.search(r"\b(identity theft|personal information|misuse of my id|aadhaar misuse)\b", text):
            return "identity_theft"
        financial_loss = re.search(
            r"(money\s+(?:was\s+)?(?:stolen|taken|deducted)|"
            r"someone\s+(?:stole|took)\s+(?:my\s+)?money|"
            r"(?:stolen|unauthori[sz]ed|fraudulent)\s+(?:money|transaction)|"
            r"bank\s+account\s+hacked|upi\s+scam|"
            r"(?:lost|debit|deduct|stole|stolen|fraud|scam|unauthori[sz]ed|recover|guarantee).{0,40}"
            r"(?:money|amount|account|bank|upi|transaction)|"
            r"(?:money|amount|account|bank|upi|transaction).{0,40}"
            r"(?:lost|debit|deduct|stole|stolen|fraud|scam|unauthori[sz]ed))",
            text,
        )
        financial_loss = financial_loss or (
            re.search(r"\b(upi|bank|account|transaction|payment)\b|₹|\brs\.?", text)
            and re.search(
                r"\b(lost|stole|stolen|stole|deducted|debit|fraud|scam|"
                r"unauthori[sz]ed|hacked|taken|recover|guarantee)\b", text,
            )
        )
        if financial_loss:
            return "recent_financial_fraud"
        if re.search(r"\b(export|preserve|save|screenshot|evidence|chat)\b", text):
            return "preserving_digital_evidence"
        if re.search(r"\b(report|complaint|complain|suspicious phone|number|cybercrime)\b", text):
            return "reporting_cybercrime"
        if re.search(r"\b(section|act|provision|offence|law|legal)\b", text):
            return "legal_provisions"
        if re.search(r"\b(avoid|prevent|protect|safe|safety|scam)\b", text):
            return "preventing_a_scam"
        if re.search(r"\b(unrelated|weather|recipe|capital of|physics)\b", text):
            return "out_of_scope"
        return "general"

    def retrieve(self, question: str, top_k: int = TOP_K,
                 threshold: float = RETRIEVAL_THRESHOLD) -> list[tuple[DocumentChunk, float]]:
        self.ensure_current()
        if not question.strip() or self.vectorizer is None or self.matrix is None:
            return []
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        intent = self.detect_intent(question)
        canonical_financial_source_present = (
            intent == "recent_financial_fraud"
            and any(item.source == FINANCIAL_FIRST_STEPS_SOURCE for item in self.chunks)
        )
        query_terms = set(re.findall(r"[a-z0-9₹]+", question.casefold()))
        ranked_scores: list[tuple[int, float]] = []
        for index, raw_score in enumerate(scores):
            item = self.chunks[index]
            text = item.text.casefold()
            boost = 0.0
            if intent == "recent_financial_fraud":
                if canonical_financial_source_present and item.trusted and (
                    item.source != FINANCIAL_FIRST_STEPS_SOURCE
                ):
                    continue
                evidence_requested = bool(re.search(
                    r"\b(chat|export|evidence|preserve|screenshot)\b",
                    question.casefold(),
                ))
                if not evidence_requested and re.search(
                    r"whatsapp|chat history|export whatsapp|exporting whatsapp",
                    text,
                ):
                    continue
                if item.trusted:
                    boost += 1.0
                    if item.source == FINANCIAL_FIRST_STEPS_SOURCE:
                        boost += 1.5
                    if item.section_heading and (
                        item.section_heading.startswith("Report immediately")
                        or item.section_heading.startswith("Immediate reporting")
                    ):
                        boost += 1.5
                    if re.search(r"\b(recover|guarantee)\b", question.casefold()) and (
                        item.section_heading and "recovery" in item.section_heading.casefold()
                    ):
                        boost += 1.5
                if any(term in text for term in ("1930", "upi", "bank", "transaction", "portal")):
                    boost += 0.25
                if "whatsapp" in text and not re.search(r"\b(chat|export|evidence|preserve)\b", question.casefold()):
                    boost -= 0.75
            elif intent == "preserving_digital_evidence":
                if any(term in text for term in ("whatsapp", "screenshot", "evidence", "preserve", "export")):
                    boost += 0.35
            elif intent == "money_mule":
                if any(term in text for term in ("money mule", "money-mule", "mule account")):
                    boost += 0.8
                else:
                    boost -= 0.25
            elif intent in {
                "card_fraud", "phishing", "otp_account_takeover",
                "shopping_payment_scam", "investment_scam", "job_task_scam",
                "social_media_impersonation", "blackmail_threat",
                "cyberstalking_harassment", "identity_theft",
            }:
                intent_terms = {
                    "card_fraud": ("card", "debit", "credit", "unauthorised"),
                    "phishing": ("phishing", "kyc", "link", "password"),
                    "otp_account_takeover": ("otp", "account", "password", "takeover"),
                    "shopping_payment_scam": ("shopping", "seller", "payment", "product"),
                    "investment_scam": ("investment", "trading", "crypto", "returns"),
                    "job_task_scam": ("job", "task", "work", "employment"),
                    "social_media_impersonation": ("social", "impersonat", "account"),
                    "blackmail_threat": ("blackmail", "threat", "sextortion"),
                    "cyberstalking_harassment": ("stalk", "harass", "message"),
                    "identity_theft": ("identity", "personal", "aadhaar"),
                }[intent]
                boost += min(0.35, sum(term in text for term in intent_terms) * 0.08)
            elif intent == "reporting_cybercrime" and any(term in text for term in ("report", "complaint", "1930", "portal")):
                boost += 0.2
            lexical_overlap = sum(1 for term in query_terms if term in text)
            boost += min(0.08, lexical_overlap * 0.01)
            ranked_scores.append((index, float(raw_score) + boost))
        ranked = sorted(ranked_scores, key=lambda pair: pair[1], reverse=True)
        results: list[tuple[DocumentChunk, float]] = []
        selected_fingerprints: set[str] = set()
        per_source: dict[str, int] = {}
        for index, _ in ranked:
            raw_score = float(scores[index])
            item = self.chunks[index]
            if raw_score < threshold and not (intent == "recent_financial_fraud" and item.trusted):
                continue
            fingerprint = " ".join(item.text.casefold().split())[:240]
            if fingerprint in selected_fingerprints:
                continue
            page_key = (
                f"{item.source}:page:{item.page}"
                if item.page is not None
                else f"{item.source}:chunk:{item.chunk_id}"
            )
            source_limit = 4 if (
                intent == "recent_financial_fraud"
                and item.source == FINANCIAL_FIRST_STEPS_SOURCE
            ) else 2
            if per_source.get(item.source, 0) >= source_limit or per_source.get(page_key, 0) >= 1:
                continue
            selected_fingerprints.add(fingerprint)
            results.append((item, raw_score))
            per_source[item.source] = per_source.get(item.source, 0) + 1
            per_source[page_key] = per_source.get(page_key, 0) + 1
            if len(results) >= top_k:
                break
        return results

    def context_for(self, results: list[tuple[DocumentChunk, float]]) -> str:
        parts: list[str] = []
        total = 0
        for item, _ in results:
            location = f"PDF page {item.page}" if item.page else f"text chunk {item.chunk_id}"
            trust = "TRUSTED CURATED GUIDANCE" if item.trusted else "UPLOADED DOCUMENT"
            section = f", section: {item.section_heading}" if item.section_heading else ""
            provenance = f", verified URL: {item.source_url}, verified date: {item.verified_date}" \
                if item.source_url else ""
            part = (f"Source: {item.source}, {location}{section}, {trust}{provenance}\n"
                    f"Passage (retrieval score is not confidence):\n{item.text}")
            if total + len(part) > MAX_CONTEXT_CHARS:
                break
            parts.append(part)
            total += len(part)
        return "\n\n---\n\n".join(parts)

    def status(self) -> dict[str, Any]:
        return {
            "chunk_count": len(self.chunks),
            "document_count": len(self.file_statuses),
            "indexed": self.vectorizer is not None,
            "last_indexed": self.last_indexed,
            "errors": self.errors,
            "documents": [
                {
                    "name": status.name, "type": status.file_type,
                    "status": status.status, "chunks": status.chunks,
                    "pages": status.pages, "message": status.message,
                }
                for status in self.file_statuses
            ],
        }
