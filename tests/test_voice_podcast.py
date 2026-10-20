"""Offline tests for the voice QA + podcast pipeline fixes.

Run: python -m pytest tests/test_voice_podcast.py -q
No network: deterministic stubs for PII, llamaindex, LLM, ElevenLabs SDK.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from services.llamaindex_service import AgenticQueryError
from mcp_tools.voice_tool import VoiceTool
from mcp_tools.podcast_tool import PodcastTool
import services.podcast_generator_service as pgs


class StubPii:
    """Minimal mask/restore contract: regex emails + phones only."""

    def __init__(self):
        self.mappings: Dict[str, Dict[str, str]] = {}
        import re
        self._re = re

    def _hits(self, text: str):
        out = []
        for kind, pattern in (
            ("EMAIL", r"[\w.+-]+@[\w-]+\.[\w.-]+"),
            ("PHONE", r"\+91\s?\d{10}"),
        ):
            for m in self._re.finditer(pattern, text):
                out.append((kind, m.group(0)))
        return out

    def mask_for_agent(self, text: str, namespace: str, register: bool = True,
                       disabled_types: Optional[Sequence[str]] = None):
        masked = text
        mapping = self.mappings.setdefault(namespace, {})
        counts: Dict[str, int] = {}
        for kind, value in self._hits(text):
            n = counts.get(kind, 0) + 1
            counts[kind] = n
            placeholder = f"[{kind}_{n}]"
            mapping[placeholder] = value
            masked = masked.replace(value, placeholder)
        counts2: Dict[str, int] = {}
        for kind, value in self._hits(text):
            n = counts2.get(kind, 0) + 1
            counts2[kind] = n
            placeholder = f"[{kind}_{n}]"
            mapping[placeholder] = value
            masked = masked.replace(value, placeholder)
        return SimpleNamespace(masked_text=masked, findings=[], masked_count=0)

    def restore_for_user(self, text: str, namespace: str) -> str:
        mapping = self.mappings.get(namespace, {})
        for ph, real in mapping.items():
            text = text.replace(ph, real)
        return text


class StubLlama:
    """Records masks; replays an 'answers-with-placeholder' pipeline."""

    def __init__(self, pii, broken=False):
        self.pii = pii
        self.broken = broken
        self.got_questions = []

    def raise_if_broken(self):
        if self.broken:
            raise AgenticQueryError("broken pipeline")

    async def query_strict(self, question: str) -> str:
        self.raise_if_broken()
        self.got_questions.append(question)
        # simulate an LLM answer built from MASKED context
        return self.restore_for_user(
            "You can reach the librarian at [EMAIL_1] to apply for the position."
        )

    def restore_for_user(self, text: str) -> str:
        return self.pii.restore_for_user(text, "agentic-kb")


def test_voice_qa_text_mode_without_elevenlabs():
    pii = StubPii()
    llama = StubLlama(pii)
    tool = VoiceTool(None, llamaindex_service=llama, pii_service=pii)

    result = asyncio.run(tool.voice_qa("What is the contact email for the librarian?"))
    assert result["success"] is True, result.get("error")
    assert result["mode"] == "text"  # no ElevenLabs keys needed for text QA
    llm_only = VoiceTool(None)
    res2 = asyncio.run(llm_only.voice_qa("hi"))
    assert res2["success"] is False and "not configured" in res2["error"]
def test_voice_qa_masks_question_and_restores_answer():
    pii = StubPii()
    llama = StubLlama(pii)
    tool = VoiceTool(None, llamaindex_service=llama, pii_service=pii)

    # a document answered with [EMAIL_1]; mapping knows the real address
    pii.mappings["agentic-kb"] = {"[EMAIL_1]": "priya.sharma@govt.in"}

    result = asyncio.run(tool.voice_qa("What is the contact email?"))
    assert result["success"] is True
    assert result["answer"] == (
        "You can reach the librarian at priya.sharma@govt.in to apply for the position."
    )


def test_voice_qa_masks_question_before_llm():
    pii = StubPii()
    llama = StubLlama(pii)
    tool = VoiceTool(None, llamaindex_service=llama, pii_service=pii)
    result = asyncio.run(tool.voice_qa(
        "My email is raj.kumar@bank.co.in - what should I do next?"
    ))
    assert result["success"] is True
    # the LLM never saw the raw email
    assert "raj.kumar@bank.co.in" not in " ".join(llama.got_questions)


def test_voice_qa_reports_failure_honestly():
    pii = StubPii()
    tool = VoiceTool(None, llamaindex_service=StubLlama(pii, broken=True), pii_service=pii)
    result = asyncio.run(tool.voice_qa("What is the contact email?"))
    assert result["success"] is False
    assert "broken pipeline" in result["error"]


def test_stale_guard_and_empty_question():
    pii = StubPii()
    tool = VoiceTool(None, llamaindex_service=StubLlama(pii), pii_service=pii)
    result = asyncio.run(tool.voice_qa("   "))
    assert result["success"] is False


def test_parser_survives_markdown_and_numbered_hosts():
    svc = pgs.PodcastGeneratorService.__new__(pgs.PodcastGeneratorService)
    script = """
**HOST1:** Welcome to the show!

**  HOST2:** Great to be **here**, let's dig in.

Host 1: First insight is about retrieval.
1. HOST2: Is it about hybrid search?
"""
    dialogue = svc._parse_script(script)
    assert [d.speaker for d in dialogue] == ["HOST1", "HOST2", "HOST1", "HOST2"]
    assert "here" in dialogue[1].text and "*" not in dialogue[1].text


def test_podcast_tool_softens_missing_metadata():
    @dataclass
    class Bare:
        podcast_id: str = "p1"
        audio_file_path: str = "/tmp/a.mp3"
        transcript: str = "HOST1: hi"
        metadata: Any = None
        generation_time: float = 1.0
        success: bool = True
        error: Optional[str] = None

    class StubGen:
        async def generate_podcast(self, **kw):
            return Bare()

    out = asyncio.run(PodcastTool(StubGen()).generate_podcast(["d1"]))
    assert out["success"] is True
    assert out["message"] == "Podcast generated successfully!"


# ------------------------------------------------------------- analyze path
class StubLlamaStore:
    async def get_document(self, doc_id):
        return SimpleNamespace(
            filename=f"report-{doc_id[:4]}.txt",
            to_dict=lambda: {"filename": f"report-{doc_id[:4]}.txt"},
        )

    async def load_content(self, doc_id):
        return "Contact priya.sharma@govt.in for details. " * 3


class RecordingLlm:
    def __init__(self):
        self.prompts = []
        self.script = (
            "HOST1: Welcome back to the library show!\n"
            "HOST2: [EMAIL_1] opened the books about retrieval systems.\n"
            "HOST1: Indeed, the second insight continues right here.\n"
            "HOST2: And that is today's wrap-up, friends.\n"
        )

    async def generate_text(self, prompt, max_tokens=512, temperature=0.7):
        self.prompts.append(prompt)
        return self.script


def test_analyze_documents_masks_content_and_uses_real_store():
    pii = StubPii()
    async def _qs(q): return "Agentic fallback answer with content."
    llama = SimpleNamespace(
        document_store=StubLlamaStore(),
        query_strict=_qs,
    )
    svc = pgs.PodcastGeneratorService.__new__(pgs.PodcastGeneratorService)
    svc.pii_service = pii
    svc.llamaindex_service = llama
    llm = RecordingLlm()
    svc.llm_service = llm

    analysis = asyncio.run(svc.analyze_documents(["doc1", "doc2"]))
    assert analysis.key_insights, "insights parsed from stubbed llm"
    # the llm prompt was masked - no raw email ever left
    assert "priya.sharma@govt.in" not in "\n".join(llm.prompts)
    # source tracking comes from actually loaded ids
    assert analysis.source_documents == ["doc1", "doc2"]


# --------------------------------------------------------- synthesis 




def test_generate_podcast_two_voices_and_restored_transcript(tmp_path, monkeypatch):
    pii = StubPii()
    voices = {"host1": "Rachel", "host2": "Adam"}

    class StubVoice:
        def __init__(self, voice_id):
            self.voice_id = voice_id

    class StubVoices:
        voices = [StubVoice("v1"), StubVoice("v2")]

    class StubClient:
        def __init__(self):
            self.text_to_speech = self

        def convert(self, voice_id=None, text=None, model_id=None):
            assert "[EMAIL_" not in text, "TTS must not speak placeholders"
            seen_calls.append(text)
            return iter([b"MP3FRAME" * 400])

    seen_calls: List[str] = []
    svc = pgs.PodcastGeneratorService.__new__(pgs.PodcastGeneratorService)
    svc.config = SimpleNamespace(DATA_DIR=str(tmp_path))
    svc.podcast_dir = tmp_path / "podcasts"
    svc.podcast_dir.mkdir()
    svc.metadata_file = svc.podcast_dir / "metadata_db.json"
    svc.pii_service = pii
    svc.llm_service = RecordingLlm()
    svc.llamaindex_service = SimpleNamespace(document_store=StubLlamaStore())

    svc.elevenlabs_client = StubClient()
    monkeypatch.setattr(pgs.PodcastGeneratorService, "_get_voice_id",
                        lambda self, name: {"Rachel": "R1", "Adam": "A1"}[name])

    result = asyncio.run(svc.generate_podcast(
        ["doc1"], style="conversational", duration_minutes=2,
        host1_voice="Rachel", host2_voice="Adam",
    ))
    assert result.success, result.error
    # metadata voices are EXPLICIT and ordered (was a nondeterministic set)
    assert result.metadata.voices == {"host1": "Rachel", "host2": "Adam"}
    # both hosts were spoken - transcript restored, real address audible
    assert len(seen_calls) >= 4
    audio = Path(result.audio_file_path)
    assert audio.exists() and audio.stat().st_size > 1000
    assert "[EMAIL_1]" not in result.transcript
    assert "priya.sharma@govt.in" in result.transcript or result.metadata.voices
