"""PII protection service: detect, mask for the agent, restore for the user.

Guarantee model ("never pass raw PII to the agent"):
  * Every LLM boundary (generative_tool answer/summary paths) routes context
    through `mask_for_agent()`; the model only ever sees `[EMAIL_1]`,
    `[AADHAAR_2]`, ... placeholders.
  * When the model's reply (which quotes those placeholders) comes back,
    `restore_for_user()` swaps the placeholders back to the original values
    **in memory only**. The mapping lives in a short-TTL per-request cache,
    is never written to disk, database, or log files.
  * The PII tab shows the masked, placeholder form to the user - raw values
    never leave the boundary unless the user is looking at their own document.

Custom rules: users can add plain keywords or regexes, persisted in SQLite
(meta table, JSON list). Custom matches surface as `[CUSTOM_<tag>_N]`.

We make no claim of completeness or regulatory compliance - this is a
best-effort technical control; validate before relying on it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.pii_patterns import (
    ALL_PATTERNS,
    PIIPattern,
    get_patterns,
)
from services.sqlite_store import SQLiteStore, dumps_json, loads_json
import config

logger = logging.getLogger(__name__)


@dataclass
class PIIFinding:
    entity: str
    label: str
    category: str
    value: str
    start: int
    end: int
    placeholder: str

    def to_table_row(self) -> List[Any]:
        return [self.entity, self.label, self.category, self.placeholder, self.value]


@dataclass
class MaskedResult:
    masked_text: str
    findings: List[PIIFinding] = field(default_factory=list)
    mapping: Dict[str, str] = field(default_factory=dict)  # placeholder -> original

    @property
    def masked_count(self) -> int:
        return len(self.findings)


_COMPILABLE = re.compile


class PIIService:
    """Regex + optional NER PII engine with reversible placeholder masking."""

    META_KEY = "pii_custom_rules"

    def __init__(self, store: Optional[SQLiteStore] = None):
        self.config = config.config
        self.store = store or SQLiteStore(self.config.DB_PATH)
        self._custom_rules: List[Dict[str, Any]] = []
        self._custom_runtime_rules: List[Dict[str, Any]] = []  # non-persisted (UI session)
        # Reversible mapping cache: namespace (e.g. "session-db-token") ->
        # (placeholder -> original). TTL-bound; never persisted.
        self._mappings: Dict[str, Tuple[float, Dict[str, str]]] = {}
        self._mappings_lock = threading.Lock()
        self._mapping_ttl = self.config.PII_MAPPING_TTL_SECONDS
        self._ner = None
        self._ner_checked = False

    async def initialize(self) -> None:
        await self._load_custom_rules()
        if self.config.PII_LOG_MASKING:
            self.install_log_masking()
        if self.config.PII_NER_ENABLED:
            self._get_ner()  # warm the optional NER engine once

    # ------------------------------------------------------------ NER (opt)
    def _get_ner(self):
        """Optional Presidio NER for person names / orgs, lazily loaded."""
        if self._ner_checked:
            return self._ner
        self._ner_checked = True
        try:
            from presidio_analyzer import AnalyzerEngine  # optional dependency

            self._ner = AnalyzerEngine()
            logger.info("PII NER engine ready (presidio)")
        except Exception as exc:
            logger.info(
                "PII NER unavailable (%s) - regex-only mode; install "
                "`presidio-analyzer` to also detect person names/locations", exc,
            )
            self._ner = None
        return self._ner

    # -------------------------------------------------------- custom rules
    async def _load_custom_rules(self) -> None:
        try:
            row = await self.store.fetch_one(
                "SELECT value FROM meta WHERE key=? LIMIT 1", (self.META_KEY,)
            )
            loaded = loads_json(row["value"]) if row else []
            self._custom_rules = [
                rule for rule in (loaded or [])
                if isinstance(rule, dict) and rule.get("rule")
            ]
            logger.info("Loaded %s custom PII rules", len(self._custom_rules))
        except Exception as exc:
            logger.warning("Could not load custom PII rules: %s", exc)
            self._custom_rules = []

    async def add_custom_rule(self, name: str, rule: str, rule_type: str = "keyword",
                              tag: str = "KEYWORD") -> Dict[str, Any]:
        """Persist a custom filter rule. rule_type: keyword | regex."""
        rule = (rule or "").strip()
        if not rule:
            return {"success": False, "error": "Empty rule"}
        if rule_type == "regex":
            try:
                _COMPILABLE(rule)
            except re.error as exc:
                return {"success": False, "error": f"Invalid regex: {exc}"}
        entry = {
            "name": (name or rule[:32]).strip(),
            "rule": rule,
            "type": rule_type if rule_type in ("keyword", "regex") else "keyword",
            "tag": re.sub(r"[^A-Z0-9_]", "_", (tag or "KEYWORD").upper())[:24] or "KEYWORD",
        }
        exists = any(
            r["rule"] == entry["rule"] and r["type"] == entry["type"]
            for r in self._custom_rules
        )
        if not exists:
            self._custom_rules.append(entry)
        await self._save_custom_rules()
        return {"success": True, "rule": entry, "total": len(self._custom_rules)}

    async def remove_custom_rule(self, name: str) -> Dict[str, Any]:
        before = len(self._custom_rules)
        self._custom_rules = [r for r in self._custom_rules if r["name"] != name]
        removed = before - len(self._custom_rules)
        await self._save_custom_rules()
        return {"success": removed > 0, "removed": removed}

    def add_session_rule(self, name: str, rule: str, rule_type: str = "keyword",
                         tag: str = "KEYWORD") -> None:
        """Add a non-persisted runtime rule (UI 'apply now without saving')."""
        if not (rule or "").strip():
            return
        self._custom_runtime_rules.append({
            "name": name or rule[:32],
            "rule": rule.strip(),
            "type": rule_type if rule_type in ("keyword", "regex") else "keyword",
            "tag": re.sub(r"[^A-Z0-9_]", "_", (tag or "KEYWORD").upper())[:24],
        })

    async def _save_custom_rules(self) -> None:
        await self.store.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (self.META_KEY, dumps_json(self._custom_rules)),
        )

    def list_custom_rules(self) -> List[Dict[str, Any]]:
        return list(self._custom_rules) + list(self._custom_runtime_rules)

    # ------------------------------------------------------------ detection
    def _enabled_patterns(self, disabled: Optional[Sequence[str]] = None) -> List[PIIPattern]:
        patterns = get_patterns()
        if disabled:
            blocked = set(disabled)
            patterns = [p for p in patterns if p.entity not in blocked]
        if self.config.PII_IN_ENABLED is False:
            patterns = [p for p in patterns if not p.entity.startswith("IN_")]
        if self.config.PII_US_ENABLED is False:
            patterns = [p for p in patterns if not p.entity.startswith("US_")]
        return patterns

    @staticmethod
    def _pattern_priority(pattern: "PIIPattern") -> float:
        """Legacy tiering with one fix: checksum-validated strong on top.

        3.5 checksum-validated strong (Aadhaar, credit cards): beats every
               validator-less match overlapping the same span (e.g. the
               unverified-Aadhaar safety net).
        3.0  validator-less patterns (PAN, IFSC, phones, email, SSN, the
               unverified-Aadhaar net, and context-confirmed weak ones) tie
               here and are split by registry order (India before USA).
        1.0  validator+weak (validator lowers a heuristic's claim tier).
        """
        if pattern.validator:
            return 3.5 if not pattern.weak else 1.0
        return 3.0

    def analyze(
        self,
        text: str,
        disabled_types: Optional[Sequence[str]] = None,
        include_ner: Optional[bool] = None,
    ) -> List[PIIFinding]:
        """Detect PII spans. Returns non-overlapping findings (best wins)."""
        if not text:
            return []
        findings: List[Tuple[int, int, int, PIIFinding]] = []  # (priority, start, end, f)
        counter: Dict[str, int] = {}

        def _emit(match: re.Match, pattern: PIIPattern, value: str, priority: int) -> None:
            value = value.strip()
            if not value:
                return
            counter_key = pattern.placeholder_prefix
            counter[counter_key] = counter.get(counter_key, 0) + 1
            placeholder = f"[{counter_key}_{counter[counter_key]}]"
            findings.append((priority, match.start(), match.end(), PIIFinding(
                entity=pattern.entity,
                label=pattern.label,
                category=pattern.category,
                value=value,
                start=match.start(),
                end=match.end(),
                placeholder=placeholder,
            )))

        for index, pattern in enumerate(self._enabled_patterns(disabled_types)):
            for match_object in pattern.pattern.finditer(text):
                value = match_object.group(0)
                value = value.strip()
                if not value or value.isspace():
                    continue
                if pattern.validator and not pattern.validator(value):
                    continue
                if pattern.weak and not pattern.context_matches(_context_span(text, match_object.start(), match_object.end())):
                    continue
                # priority matches the comment: validated-strong > strong |
                # validated-weak > weak heuristic. A checksum-validated match
                # must outrank an unverified fallback spanning the same digits.
                priority = self._pattern_priority(pattern)
                _emit(match_object, pattern, value, priority - index * 0.01)

        # custom rules
        for rule in (*self._custom_rules, *self._custom_runtime_rules):
            try:
                compiled = (
                    _COMPILABLE(rule["rule"], re.IGNORECASE)
                    if rule["type"] == "regex"
                    else re.compile(r"(\b" + re.escape(rule["rule"]) + r"\b)", re.IGNORECASE)
                )
            except re.error:
                logger.warning("Skipping invalid custom PII rule: %s", rule.get("name"))
                continue
            for match_object in compiled.finditer(text):
                counter[rule["tag"]] = counter.get(rule["tag"], 0) + 1
                placeholder = f"[{rule['tag']}_{counter[rule['tag']]}]"
                label = f"Custom: {rule['name']}"
                findings.append((0.5, match_object.start(), match_object.end(), PIIFinding(
                    entity=rule["tag"],
                    label=label,
                    category="custom",
                    value=rule["rule"],
                    start=match_object.start(),
                    end=match_object.end(),
                    placeholder=placeholder,
                )))

        # optional NER pass (person names, locations, orgs)
        use_ner = self.config.PII_NER_ENABLED if include_ner is None else include_ner
        if use_ner:
            ner = self._get_ner()
            if ner is not None:
                try:
                    ner_results = ner.analyze(text, entities=["PERSON", "ORGANIZATION", "LOCATION"],
                                              language="en", return_decision_process=False)
                    for result in ner_results:
                        if result.score < self.config.PII_NER_MIN_SCORE:
                            continue
                        entity_key = "PERSON"
                        counter[entity_key] = counter.get(entity_key, 0) + 1
                        placeholder = f"[{entity_key}_{counter[entity_key]}]"
                        findings.append((2.5, result.start, result.end, PIIFinding(
                            entity="PERSON",
                            label="Person/Organization/Location (NER)",
                            category="person",
                            value=text[result.start:result.end],
                            start=result.start,
                            end=result.end,
                            placeholder=placeholder,
                        )))
                except Exception as exc:
                    logger.warning("PII NER pass failed: %s", exc)

        # resolve overlaps: highest priority wins within any overlap group
        findings.sort(key=lambda item: (-item[0], item[1], -item[2]))
        selected: List[PIIFinding] = []
        occupied: List[Tuple[int, int]] = []

        def overlaps(start: int, end: int) -> bool:
            return any(start < used_end and end > used_start for used_start, used_end in occupied)

        for priority, start, end, finding in findings:
            if overlaps(start, end):
                continue
            occupied.append((start, end))
            selected.append(finding)

        selected.sort(key=lambda f: f.start)
        return selected

    # ------------------------------------------------------------- masking
    def mask_for_agent(
        self,
        text: str,
        namespace: str,
        disabled_types: Optional[Sequence[str]] = None,
        register: bool = True,
    ) -> MaskedResult:
        """Rewrite PII to placeholders and remember originals for restore."""
        if not text or not self.config.PII_MASKING_ENABLED:
            return MaskedResult(masked_text=text)
        findings = self.analyze(text, disabled_types=disabled_types)
        if not findings:
            return MaskedResult(masked_text=text, findings=[])

        # reuse existing placeholder when the same value appeared before
        with self._mappings_lock:
            mapping = self._mappings.get(namespace, (0, {}))[1]
            reverse = {original: placeholder for placeholder, original in mapping.items()}
            mapping = dict(mapping)
            for finding in findings:
                if finding.value not in reverse:
                    reverse[finding.value] = finding.placeholder
                    mapping[finding.placeholder] = finding.value
            if register:
                self._mappings[namespace] = (time.time(), mapping)
            self._prune_mappings()

        masked_chunks: List[str] = []
        cursor = 0
        for finding in findings:
            placeholder = reverse.get(finding.value, finding.placeholder)
            masked_chunks.append(text[cursor:finding.start])
            masked_chunks.append(placeholder)
            cursor = finding.end
        masked_chunks.append(text[cursor:])
        masked_text = "".join(masked_chunks)

        # mask with the shared placeholders so identical values regenerate
        return MaskedResult(masked_text=masked_text, findings=findings, mapping=mapping)

    def restore_for_user(self, text: str, namespace: str) -> str:
        """Swap placeholders back to the original values (memory-only)."""
        if not text:
            return text
        with self._mappings_lock:
            entry = self._mappings.get(namespace)
            mapping = entry[1] if entry else {}
        for placeholder, original in mapping.items():
            if placeholder in text:
                text = text.replace(placeholder, original)
        return text

    def peek_mapping(self, namespace: str) -> Dict[str, str]:
        with self._mappings_lock:
            entry = self._mappings.get(namespace)
            return dict(entry[1]) if entry else {}

    def drop_namespace(self, namespace: str) -> None:
        with self._mappings_lock:
            self._mappings.pop(namespace, None)

    def _prune_mappings(self) -> None:
        now = time.time()
        expired = [key for key, (created, _) in self._mappings.items()
                   if now - created > self._mapping_ttl]
        for key in expired:
            self._mappings.pop(key, None)
        # hard cap so a runaway session cannot grow unbounded
        max_entries = self.config.PII_MAX_SESSIONS
        if len(self._mappings) > max_entries:
            for key, (created, _) in sorted(self._mappings.items(), key=lambda kv: kv[1][0])[
                    : len(self._mappings) - max_entries]:
                self._mappings.pop(key, None)

    # ------------------------------------------------------- log masking
    def install_log_masking(self) -> None:
        """Masks obvious PII in all emitted log records (queries included)."""
        service = self

        class _PIILogFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                try:
                    message = record.getMessage()
                    masked = service.mask_for_agent(message, namespace="__logs__", register=False)
                    if masked.findings:
                        record.msg = masked.masked_text
                        record.args = ()
                except Exception:
                    pass  # never break logging
                return True

        for handler in logging.getLogger().handlers:
            handler.addFilter(_PIILogFilter())


def _context_span(text: str, start: int, end: int, radius: int = 60) -> str:
    return text[max(0, start - radius): min(len(text), end + radius)]
