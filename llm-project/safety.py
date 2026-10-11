"""Local, non-invasive helpers for scam triage and incident notes."""

import re
from typing import Any


def analyze_scam_text(text: str) -> dict[str, Any]:
    """Identify explainable warning signals without opening URLs or contacting services."""
    value = text.strip()
    if not value:
        return {"signals": [], "safe_steps": [], "assessment": "No text was provided."}
    rules = [
        (r"\b(otp|upi pin|cvv|password|verification code)\b",
         "Requests for credentials or security codes are a strong warning signal."),
        (r"\b(urgent|immediately|within \d+ minutes|account will be blocked)\b",
         "Urgency or threats of account closure can pressure you into acting unsafely."),
        (r"\b(click|open|verify|update)\b.{0,40}\b(link|url|kyc|account)\b",
         "A request to open a link or update KYC/account details deserves independent verification."),
        (r"\b(refund|prize|lottery|cashback|investment return)\b",
         "Unexpected rewards, refunds, or guaranteed returns are common scam indicators."),
        (r"\b(remote access|anydesk|teamviewer|screen share)\b",
         "Requests for remote access or screen sharing are a strong warning signal."),
    ]
    signals = [
        {"strength": "strong" if index in (0, 4) else "caution", "explanation": explanation}
        for index, (pattern, explanation) in enumerate(rules)
        if re.search(pattern, value, re.I | re.S)
    ]
    return {
        "signals": signals,
        "assessment": (
            "The text contains warning signals. Do not open links or share credentials; "
            "verify through an official app, website, or published phone number."
            if signals else
            "No common warning signal was detected. This does not prove the message is safe."
        ),
        "safe_steps": [
            "Do not open links, download files, or reply with OTPs, UPI PINs, passwords, or CVVs.",
            "Verify independently using the organisation's official app or website.",
            "If money or account access is affected, contact the bank and report cyber financial fraud promptly.",
        ],
        "local_only": True,
    }
