from fastapi.testclient import TestClient

from app import app


def test_chat_validates_short_question():
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"question": "?"})
        assert response.status_code == 422


def test_health_has_index_and_local_status():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert "ollama" in body
        assert body["local_only"] is True


def test_chat_returns_claim_level_sources(monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "ollama_status", lambda: {
        "connected": True, "model": "qwen2.5:3b", "model_available": True,
        "models": [], "message": None,
    })
    monkeypatch.setattr(
        app_module.client,
        "chat",
        lambda **_: {"message": {"content": "Call 1930 promptly. Recovery is not guaranteed."}},
    )
    with TestClient(app) as client:
        response = client.post("/api/chat", json={
            "question": "I lost ₹10,000 in a UPI scam five minutes ago. What should I do first?"
        })
        assert response.status_code == 200
        body = response.json()
        assert body["intent"] == "recent_financial_fraud"
        assert body["sources"]
        assert "claim_supported" in body["sources"][0]
        assert body["sources"][0]["trusted_guidance"] is True


def test_scam_analysis_endpoint():
    with TestClient(app) as client:
        response = client.post("/api/analyze-scam", json={
            "text": "Your account will be blocked. Share OTP now."
        })
        assert response.status_code == 200
        assert response.json()["signals"]
