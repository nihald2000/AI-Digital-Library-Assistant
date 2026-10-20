import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class VoiceTool:
    """Voice-enabled Q&A over the library.

    Text Q&A works with ANY LLM/RAG setup - ElevenLabs (Conversational AI
    agents, audio) is purely additive. The old behavior hard-required
    ELEVENLABS_API_KEY/ELEVENLABS_AGENT_ID and made the whole text tab dead
    without them.

    PII protocol (same boundary as DoChat/MCP): the question is masked before
    it reaches the agentic pipeline, the agentic index itself was built from
    masked content, and the user-facing answer is restored from the in-memory
    TTL mapping.
    """

    def __init__(self, elevenlabs_service, llamaindex_service=None, pii_service=None):
        self.elevenlabs_service = elevenlabs_service
        self.llamaindex_service = llamaindex_service
        self.pii_service = pii_service

    async def voice_qa(
        self,
        question: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ask a question through the agentic RAG pipeline.

        Returns ('success', 'answer'|'error', 'mode'). `mode` reports what is
        actually powering the reply: 'text+voice' when an ElevenLabs
        conversation is live, 'text' otherwise.
        """
        try:
            question = (question or "").strip()
            if not question:
                return {
                    "success": False,
                    "error": "Please provide a question.",
                    "question": question,
                    "mode": "text",
                }

            voice_ready = bool(
                self.elevenlabs_service
                and getattr(self.elevenlabs_service, "is_available", lambda: False)()
            )
            mode = "text+voice" if voice_ready else "text"

            if not self.llamaindex_service:
                return {
                    "success": False,
                    "error": "Agentic RAG service not configured.",
                    "question": question,
                    "mode": mode,
                }

            # PII boundary: mask the question (maps into the shared agentic
            # namespace, which also powers restore below).
            masked_question = question
            if self.pii_service and question:
                try:
                    from services.mcp_pii import AGENTIC_PII_NAMESPACE

                    masked_question = self.pii_service.mask_for_agent(
                        question, namespace=AGENTIC_PII_NAMESPACE
                    ).masked_text
                except Exception as exc:
                    logger.warning("Voice PII mask failed (proceeding): %s", exc)

            # query_strict raises on a broken pipeline or a failed LLM call -
            # error strings can no longer masquerade as successful answers.
            answer = await self.llamaindex_service.query_strict(masked_question)

            return {
                "success": True,
                "question": question,
                "answer": answer,
                "session_id": session_id,
                "mode": mode,
            }

        except Exception as e:
            logger.error(f"Voice QA failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "mode": "text",
            }
