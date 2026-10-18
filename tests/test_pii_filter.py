"""Offline tests for the PII protection layer (no API keys, no NER model).

Covers: India/US/global detection, checksum validators, masking -> placeholder
-> restore round-trip, custom rules (keyword + regex persisted), and
context-gated weak patterns.

Run: python -m pytest tests/test_pii_filter.py -q
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


# Verhoeff-valid sample Aadhaar numbers (same checksum algorithm UIDAI uses)
VALID_AADHAAR_SPACED = "2345 1234 5674"
VALID_AADHAAR_COMPACT = "234512345674"
INVALID_CARD = "1600 0000 0000 0001"  # fails Luhn
VALID_CARD = "4111 1111 1111 1111"


@pytest.fixture()
def pii(env):
    from services.pii_service import PIIService
    from services.sqlite_store import SQLiteStore
    import config

    workdir = env
    config.config.PII_LOG_MASKING = False  # keep test log handlers clean
    store = SQLiteStore(os.path.join(workdir, "pii.db"))
    service = PIIService(store=store)
    asyncio.run(store.initialize())
    asyncio.run(service.initialize())
    try:
        yield service
    finally:
        asyncio.run(store.close())


@pytest.fixture()
def env(monkeypatch):
    workdir = tempfile.mkdtemp(prefix="pii_test_")
    import config

    monkeypatch.setattr(config.config, "DB_PATH", os.path.join(workdir, "test.db"))
    monkeypatch.setattr(config.config, "PII_NER_ENABLED", False)
    monkeypatch.setattr(config.config, "PII_LOG_MASKING", False)
    return workdir


def _entities(service, text):
    return [f.entity for f in service.analyze(text)]


def test_detection_core_entities(pii):
    text = (
        "Contact ravi.sharma@gmail.com or +91 98765 43210, US line 212-555-0234. "
        f"Aadhaar {VALID_AADHAAR_SPACED}. PAN ABCDE1234F. IFSC HDFC0001234. "
        f"Card {VALID_CARD}. Server at 10.0.1.42."
    )
    entities = _entities(pii, text)
    assert "EMAIL" in entities
    assert "IN_PHONE" in entities
    assert "US_PHONE" in entities
    assert "IN_AADHAAR" in entities
    assert "IN_PAN" in entities
    assert "IN_IFSC" in entities
    assert "CREDIT_CARD" in entities
    assert "IP_ADDRESS" in entities


def test_checksum_validators(pii):
    # Luhn-valid card is caught, Luhn-invalid 16-digit number is not
    assert "CREDIT_CARD" in _entities(pii, f"pay with {VALID_CARD} now")
    assert "CREDIT_CARD" not in _entities(pii, f"ref {INVALID_CARD} nope")

    # Verhoeff-valid Aadhaar caught (with or without separators)
    assert "IN_AADHAAR" in _entities(pii, f"id {VALID_AADHAAR_COMPACT} ok")
    assert "IN_AADHAAR" in _entities(pii, f"id {VALID_AADHAAR_SPACED} ok")


def test_weak_patterns_need_context(pii):
    # bare 5-digit number is NOT a ZIP without zip/postal context
    assert "US_ZIP" not in _entities(pii, "order 94105 shipped")
    assert "US_ZIP" in _entities(pii, "zip code 94105 shipped")

    # bare 10-digit number is not a phone without phone-ish context
    assert "IN_PHONE_INFERRED" not in _entities(pii, "ticket 9876543210 batch")
    assert "IN_PHONE_INFERRED" in _entities(pii, "mobile 9876543210 active")

    # but strong-format phones (with +91) fire without context
    assert "IN_PHONE" in _entities(pii, "call +91 98765 43210 today")


def test_mask_restore_roundtrip(pii):
    original = (
        "Ravi asked: reach me at ravi.sharma@gmail.com or +91 98765 43210, "
        f"Aadhaar {VALID_AADHAAR_SPACED}"
    )
    masked = pii.mask_for_agent(original, namespace="test:1", register=True)
    assert masked.masked_text != original
    assert original not in masked.masked_text
    assert "ravi.sharma@gmail.com" not in masked.masked_text
    assert "[EMAIL_" in masked.masked_text
    assert "[PHONE_" in masked.masked_text
    assert "[AADHAAR_" in masked.masked_text

    # the agent's reply quotes placeholders; the user sees real values again
    agent_reply = (
        "The contact in [EMAIL_1] and phone [PHONE_1] belongs to document X; "
        "Aadhaar [AADHAAR_1] verified."
    )
    restored = pii.restore_for_user(agent_reply, "test:1")
    assert "ravi.sharma@gmail.com" in restored
    assert "+91 98765 43210" in restored
    assert VALID_AADHAAR_SPACED in restored
    assert "[EMAIL_" not in restored


def test_persisted_custom_rules(pii):
    async def flow():
        added = await pii.add_custom_rule(
            "Project Codename", "Project Falcon", "keyword", "PROJECT"
        )
        assert added["success"] is True

        text = "Launch details for Project Falcon were annotated, then ravi@gmail.com arrived."
        masked = pii.mask_for_agent(text, namespace="test:custom", register=False)
        assert "Project Falcon" not in masked.masked_text
        assert "[PROJECT_1]" in masked.masked_text
        assert "[EMAIL_1]" in masked.masked_text  # built-ins still work

        # regex rule
        regex_rule = await pii.add_custom_rule("Secret IDs", r"SEC-\d{4}", "regex", "SEC")
        assert regex_rule["success"]
        masked2 = pii.mask_for_agent("id SEC-4321 here", namespace="test:regex", register=False)
        assert "SEC-4321" not in masked2.masked_text
        assert "[SEC_1]" in masked2.masked_text

        # rules survive reload from the database
        from services.pii_service import PIIService
        fresh = PIIService(store=pii.store)
        await fresh.initialize()
        names = [r["name"] for r in fresh.list_custom_rules()]
        assert "Project Codename" in names
        assert "Secret IDs" in names

        removed = await pii.remove_custom_rule("Project Codename")
        assert removed["success"] is True
        removed_again = await pii.remove_custom_rule("Project Codename")
        assert removed_again["success"] is False

    asyncio.run(flow())


def test_disabled_types_respected(pii):
    text = "email someone@example.com"
    normal = pii.mask_for_agent(text, namespace="t:d1", register=False)
    assert "[EMAIL_" in normal.masked_text
    disabled = pii.mask_for_agent(
        text, namespace="t:d2", disabled_types=["EMAIL"], register=False
    )
    assert disabled.masked_text == text


def test_mapping_never_persisted(pii):
    """Mappings are memory-only: closing/recreating the service loses them."""
    text = "ping alice@corp.io"
    masked = pii.mask_for_agent(text, namespace="t:p1", register=True)
    assert masked.masked_count == 1
    from services.pii_service import PIIService
    fresh = PIIService(store=pii.store)
    asyncio.run(fresh.initialize())
    assert fresh.peek_mapping("t:p1") == {}
    restored = fresh.restore_for_user("alice is [EMAIL_1]", "t:p1")
    assert restored == "alice is [EMAIL_1]"  # placeholder survives; no mapping leak
