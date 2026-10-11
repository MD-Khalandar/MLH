import json
from pathlib import Path

from rag import LegalDocumentIndex
from safety import analyze_scam_text


def test_intent_evaluation_dataset():
    cases = json.loads((Path(__file__).parent / "evaluation_queries.json").read_text())
    assert len(cases) >= 30
    assert sum(LegalDocumentIndex.detect_intent(question) == expected
               for question, expected in cases) == len(cases)


def test_scam_analysis_is_local_and_does_not_open_links():
    result = analyze_scam_text("URGENT: click this link and share your OTP to avoid account closure")
    assert result["local_only"] is True
    assert any(signal["strength"] == "strong" for signal in result["signals"])
    assert "Do not open links" in result["safe_steps"][0]


def test_scam_analysis_does_not_claim_certainty():
    result = analyze_scam_text("Hello, your statement is ready.")
    assert "does not prove the message is safe" in result["assessment"]
