from pathlib import Path

from rag import LegalDocumentIndex, split_into_chunks


def test_chunking_overlaps_and_is_bounded():
    chunks = split_into_chunks("alpha " * 600, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_txt_ingestion_and_retrieval(tmp_path: Path):
    (tmp_path / "contract.txt").write_text(
        "A contract is an agreement enforceable by law. Free consent is required.",
        encoding="utf-8",
    )
    index = LegalDocumentIndex(tmp_path)
    index.rebuild()
    results = index.retrieve("What is a contract?", threshold=0.01)
    assert results
    assert results[0][0].source == "contract.txt"
    assert results[0][0].page is None


def test_empty_document_is_reported(tmp_path: Path):
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    index = LegalDocumentIndex(tmp_path)
    status = index.rebuild()
    assert status["documents"][0]["status"] == "warning"
    assert status["chunk_count"] == 0
    assert index.retrieve("anything") == []


def test_corrupt_pdf_is_reported(tmp_path: Path):
    (tmp_path / "broken.pdf").write_bytes(b"not a PDF")
    index = LegalDocumentIndex(tmp_path)
    status = index.rebuild()
    assert status["documents"][0]["status"] == "error"
    assert status["errors"]


def test_unrelated_question_returns_no_results(tmp_path: Path):
    (tmp_path / "law.txt").write_text("Offer and acceptance form an agreement.", encoding="utf-8")
    index = LegalDocumentIndex(tmp_path)
    index.rebuild()
    assert index.retrieve("quantum physics black holes", threshold=0.08) == []


def _fraud_index(tmp_path: Path) -> LegalDocumentIndex:
    (tmp_path / "emergency_financial_fraud_guidance.txt").write_text(
        """Source URL: https://www.cybercrime.gov.in/
Verified date: 2026-10-11

[SECTION: Immediate reporting — official National Cyber Crime Reporting Portal]
Call 1930 promptly and report at https://www.cybercrime.gov.in/ after a UPI fraud.

[SECTION: Protect account information and avoid recovery scams]
Do not share OTPs or UPI PINs, and do not pay recovery scammers.
""",
        encoding="utf-8",
    )
    (tmp_path / "cybercrime_citizen_manual.txt").write_text(
        "[SECTION: WhatsApp chat export]\nExport your WhatsApp chat as evidence when you need to preserve messages.",
        encoding="utf-8",
    )
    index = LegalDocumentIndex(tmp_path)
    index.rebuild()
    return index


def test_recent_upi_fraud_prioritises_immediate_reporting(tmp_path: Path):
    index = _fraud_index(tmp_path)
    results = index.retrieve(
        "I lost ₹10,000 in a UPI scam five minutes ago. What should I do first?",
        threshold=0.01,
    )
    assert results
    assert results[0][0].trusted is True
    assert results[0][0].section_heading.startswith("Immediate reporting")
    assert all("WhatsApp" not in item.text for item, _ in results)


def test_whatsapp_export_intent_can_retrieve_whatsapp_guidance(tmp_path: Path):
    index = _fraud_index(tmp_path)
    results = index.retrieve("How do I export my WhatsApp chat as evidence?", threshold=0.01)
    assert results
    assert results[0][0].section_heading == "WhatsApp chat export"


def test_reporting_phone_number_has_reporting_intent(tmp_path: Path):
    index = _fraud_index(tmp_path)
    assert index.detect_intent("How do I report a suspicious phone number?") == "reporting_cybercrime"


def test_recovery_question_is_not_supported_as_a_guarantee(tmp_path: Path):
    index = _fraud_index(tmp_path)
    results = index.retrieve("Can you guarantee I will recover my money?", threshold=0.01)
    assert results
    assert any("Recovery" in item.text or "recovery" in item.text for item, _ in results)


def test_unrelated_question_ignores_fraud_guidance(tmp_path: Path):
    index = _fraud_index(tmp_path)
    assert index.retrieve("What is the capital of Mars?", threshold=0.08) == []


def test_plain_money_stolen_query_prioritises_first_steps(tmp_path: Path):
    index = _fraud_index(tmp_path)
    assert index.detect_intent("someone stole 1000 rs from my account") == "recent_financial_fraud"
    results = index.retrieve("someone stole 1000 rs from my account", threshold=0.01)
    assert results[0][0].trusted is True
    assert results[0][0].section_heading.startswith("Immediate reporting")


def test_recent_upi_loss_prioritises_first_steps(tmp_path: Path):
    index = _fraud_index(tmp_path)
    results = index.retrieve("I lost ₹10,000 in a UPI scam five minutes ago", threshold=0.01)
    assert results[0][0].trusted is True
    assert results[0][0].section_heading.startswith("Immediate reporting")


def test_whatsapp_evidence_query_retrieves_evidence_guidance(tmp_path: Path):
    index = _fraud_index(tmp_path)
    results = index.retrieve("How do I preserve WhatsApp evidence?", threshold=0.01)
    assert results
    assert "WhatsApp" in results[0][0].text


def test_money_mule_query_retrieves_money_mule_material(tmp_path: Path):
    (tmp_path / "rbi_booklet.txt").write_text(
        "A money mule is a person whose account receives and transfers money for criminals.",
        encoding="utf-8",
    )
    index = _fraud_index(tmp_path)
    results = index.retrieve("What is a money mule?", threshold=0.01)
    assert results
    assert results[0][0].source == "rbi_booklet.txt"
