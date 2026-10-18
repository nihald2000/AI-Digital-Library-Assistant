"""PII pattern registry: India, USA, and global sensitive-data detectors.

Design basis (SCALING.md notes the citations):
* Mirrors Microsoft Presidio's recognizer taxonomy: deterministic regex for
  structured data, checksum validation where the issuing authority defines one
  (Aadhaar Verhoeff, credit card Luhn), and context keywords for high
  false-positive patterns that are meaningless without surrounding context
  (US ZIP, ABA routing, passport numbers).
* Entity ids follow Presidio naming so a later swap to Presidio NER is
  transparent: IN_AADHAAR, IN_PAN, IN_GSTIN, IN_IFSC, IN_PHONE, IN_PASSPORT,
  US_SSN, US_ITIN, US_PHONE, US_ZIP, US_ROUTING, US_DL, plus global EMAIL,
  IP_ADDRESS, CREDIT_CARD, IBAN, CRYPTO_WALLET, URL.

We make no claim of completeness or compliance - detection is best effort and
must be validated before use in regulated workflows (see PII tab disclaimer).

Name/location detection (unstructured) is available through an optional NER
engine (Presidio/spaCy when installed; see pii_service.PIINerEngine) because
regex alone cannot detect names reliably.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Pattern


def luhn_valid(digits: str) -> bool:
    """Luhn checksum for credit-card-like numbers."""
    digits = re.sub(r"\D", "", digits)
    if len(digits) < 13 or len(digits) > 19:
        return False
    total, parity = 0, 0
    for char in reversed(digits):
        d = ord(char) - 48
        if parity % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        parity += 1
    return total % 10 == 0


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 5, 8, 1, 4, 6, 0, 3],
    [7, 0, 4, 1, 3, 2, 9, 5, 8, 6],
    [3, 6, 5, 0, 9, 4, 8, 2, 1, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(digits: str) -> bool:
    """Verhoeff checksum - validates Aadhaar numbers and other gov IDs."""
    digits = re.sub(r"\D", "", digits)
    c = 0
    for i, char in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][ord(char) - 48]]
    return _VERHOEFF_INV[c] == 0


# Context keywords: these entities only fire when surrounding text mentions
# the keyword, because the bare pattern is too generic (5-digit numbers etc.).
_IN_PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_IN_GSTIN_PATTERN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b")
_IN_IFSC_PATTERN = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
_IN_AADHAAR_PATTERN = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
_IN_PHONE_STRONG = re.compile(r"(?:\+91|91[-\s]|0)[-\s]?[6-9]\d{4}[-\s]?\d{5}\b")
_IN_PHONE_WEAK = re.compile(r"(?<!\d)[6-9]\d{4}[-\s]?\d{5}(?!\d)")
_US_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+1[-\s.]?)?(?:\(\d{3}\)|\d{3})[-\s.]?\d{3}[-\s.]?\d{4}(?!\d)")
_US_SSN_PATTERN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_US_ITIN_PATTERN = re.compile(r"(?<!\d)9\d{2}[-\s](?:7\d|8[0-8]|9[024-579])[-\s]\d{4}(?!\d)")
_US_ZIP_PATTERN = re.compile(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)")
_US_ROUTING_PATTERN = re.compile(r"(?<!\d)\d{9}(?!\d)")
_US_DL_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Z][0-9]{7,12}(?![0-9])")
_IN_PASSPORT_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Z][0-9]{7}(?![0-9])")
_CREDIT_CARD_PATTERN = re.compile(r"(?<![\dA-Za-z])(?:\d[ -]?){13,19}\d(?![\dA-Za-z])")
_IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,26}\b")
_CRYPTO_PATTERN = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IP_PATTERN = re.compile(r"(?<![\d.])\b(?:\d{1,3}\.){3}\d{1,3}\b(?!\d)")
_URL_PATTERN = re.compile(r"\bhttps?://[^\s<>\"]+", re.IGNORECASE)


@dataclass
class PIIPattern:
    """One detector: regex + optional validator + optional context words."""

    entity: str                      # Presidio-style id, e.g. IN_AADHAAR
    label: str                       # human label for the UI
    category: str                    # contact | government_id | financial | digital
    pattern: Pattern[str]
    placeholder_prefix: str          # e.g. EMAIL -> [EMAIL_1]
    validator: Optional[Callable[[str], bool]] = None
    context_words: List[str] = field(default_factory=list)
    weak: bool = False               # weak patterns need context to fire

    def context_matches(self, around: str) -> bool:
        if not self.context_words:
            return True
        lowered = around.lower()
        return any(word in lowered for word in self.context_words)


_GOV_CONTEXT = ["aadhaar", "uid", "aadhar", "uidai"]
_ZIP_CONTEXT = ["zip", "postal", "pin code", "pincode"]
_ROUTING_CONTEXT = ["routing", "aba", "bank routing"]
_DL_CONTEXT = ["license", "licence", "dl no", "d.l."]
_PASSPORT_CONTEXT = ["passport", "ppt", "no."]


def _context_span(text: str, start: int, end: int, radius: int = 60) -> str:
    return text[max(0, start - radius): min(len(text), end + radius)]


# ---------------------------------------------------------------------------
# India entities
# ---------------------------------------------------------------------------
INDIA_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        # checksum-validated; fires without context (FP risk is accepted and
        # the type can be toggled off in the UI)
        entity="IN_AADHAAR", label="Aadhaar Number", category="government_id",
        pattern=_IN_AADHAAR_PATTERN, placeholder_prefix="AADHAAR",
        validator=verhoeff_valid, context_words=_GOV_CONTEXT, weak=False,
    ),
    PIIPattern(
        entity="IN_PAN", label="PAN Card", category="government_id",
        pattern=_IN_PAN_PATTERN, placeholder_prefix="PAN",
    ),
    PIIPattern(
        entity="IN_GSTIN", label="GSTIN", category="government_id",
        pattern=_IN_GSTIN_PATTERN, placeholder_prefix="GSTIN",
    ),
    PIIPattern(
        entity="IN_IFSC", label="IFSC Code", category="financial",
        pattern=_IN_IFSC_PATTERN, placeholder_prefix="IFSC",
    ),
    PIIPattern(
        entity="IN_PHONE", label="Indian Phone Number", category="contact",
        pattern=_IN_PHONE_STRONG, placeholder_prefix="PHONE",
    ),
    PIIPattern(
        entity="IN_PHONE_INFERRED", label="Indian Phone Number (inferred)",
        category="contact", pattern=_IN_PHONE_WEAK, placeholder_prefix="PHONE",
        context_words=["phone", "mobile", "contact", "call", "whatsapp", "tel"],
        weak=True,
    ),
    PIIPattern(
        entity="IN_PASSPORT", label="Indian Passport Number",
        category="government_id", pattern=_IN_PASSPORT_PATTERN,
        placeholder_prefix="PASSPORT", context_words=_PASSPORT_CONTEXT, weak=True,
    ),
]

# ---------------------------------------------------------------------------
# USA entities
# ---------------------------------------------------------------------------
USA_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        entity="US_SSN", label="Social Security Number (SSN)", category="government_id",
        pattern=_US_SSN_PATTERN, placeholder_prefix="SSN",
    ),
    PIIPattern(
        entity="US_ITIN", label="Individual Taxpayer ITIN", category="government_id",
        pattern=_US_ITIN_PATTERN, placeholder_prefix="ITIN",
    ),
    PIIPattern(
        entity="US_PHONE", label="US Phone Number", category="contact",
        pattern=_US_PHONE_PATTERN, placeholder_prefix="PHONE",
    ),
    PIIPattern(
        entity="US_ZIP", label="US ZIP / Postal Code", category="contact",
        pattern=_US_ZIP_PATTERN, placeholder_prefix="ZIP",
        context_words=_ZIP_CONTEXT, weak=True,
    ),
    PIIPattern(
        entity="US_ROUTING", label="ABA Routing Number", category="financial",
        pattern=_US_ROUTING_PATTERN, placeholder_prefix="ROUTING",
        context_words=_ROUTING_CONTEXT, weak=True,
    ),
    PIIPattern(
        entity="US_DL", label="Driver's License Number", category="government_id",
        pattern=_US_DL_PATTERN, placeholder_prefix="DL",
        context_words=_DL_CONTEXT, weak=True,
    ),
]

# ---------------------------------------------------------------------------
# Global entities
# ---------------------------------------------------------------------------
GLOBAL_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        entity="EMAIL", label="Email Address", category="contact",
        pattern=_EMAIL_PATTERN, placeholder_prefix="EMAIL",
    ),
    PIIPattern(
        entity="PHONE_GLOBAL", label="Phone Number (generic)", category="contact",
        # fallback leg for non IN/US phones: 10+ digits with separators
        pattern=re.compile(r"(?<!\d)\d{10,15}(?!\d)"),
        placeholder_prefix="PHONE", weak=True,
        context_words=["phone", "mobile", "contact", "tel", "call", "fax"],
    ),
    PIIPattern(
        entity="CREDIT_CARD", label="Credit/Debit Card Number", category="financial",
        pattern=_CREDIT_CARD_PATTERN, placeholder_prefix="CARD",
        validator=luhn_valid,
    ),
    PIIPattern(
        entity="IBAN", label="IBAN Code", category="financial",
        pattern=_IBAN_PATTERN, placeholder_prefix="IBAN",
    ),
    PIIPattern(
        entity="CRYPTO_WALLET", label="Crypto Wallet Address", category="financial",
        pattern=_CRYPTO_PATTERN, placeholder_prefix="WALLET",
    ),
    PIIPattern(
        entity="IP_ADDRESS", label="IP Address", category="digital",
        pattern=_IP_PATTERN, placeholder_prefix="IP",
    ),
    PIIPattern(
        entity="URL", label="URL", category="digital",
        pattern=_URL_PATTERN, placeholder_prefix="URL",
    ),
]

ALL_PATTERNS: Dict[str, PIIPattern] = {}
for _p in (*INDIA_PATTERNS, *USA_PATTERNS, *GLOBAL_PATTERNS):
    ALL_PATTERNS[_p.entity] = _p

PATTERN_BY_REGION = {
    "IN": [p.entity for p in INDIA_PATTERNS],
    "US": [p.entity for p in USA_PATTERNS],
    "GLOBAL": [p.entity for p in GLOBAL_PATTERNS],
}

# display order for the UI's findings table
CATEGORY_ORDER = ["contact", "government_id", "financial", "digital", "custom", "person"]


def get_patterns(enabled_entities: Optional[List[str]] = None) -> List[PIIPattern]:
    if enabled_entities is None:
        return list(ALL_PATTERNS.values())
    return [p for p in ALL_PATTERNS.values() if p.entity in set(enabled_entities)]
