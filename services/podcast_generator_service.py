import logging
import asyncio
import json
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import re

try:
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

import config
from services.llamaindex_service import LlamaIndexService
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

@dataclass
class DocumentAnalysis:
    """Analysis results from document(s)"""
    key_insights: List[str]  # 5-7 main points
    topics: List[str]
    complexity_level: str  # beginner, intermediate, advanced
    estimated_words: int
    source_documents: List[str]
    summary: str

@dataclass
class DialogueLine:
    """Single line of podcast dialogue"""
    speaker: str  # "HOST1" or "HOST2"
    text: str
    pause_after: float = 0.5  # seconds
    
@dataclass
class PodcastScript:
    """Complete podcast script"""
    dialogue: List[DialogueLine]
    total_duration_estimate: float
    word_count: int
    style: str
    
    def to_text(self) -> str:
        """Convert to readable transcript"""
        lines = []
        for line in self.dialogue:
            lines.append(f"{line.speaker}: {line.text}")
        return "\n\n".join(lines)

@dataclass
class PodcastMetadata:
    """Metadata for generated podcast"""
    podcast_id: str
    title: str
    description: str
    source_documents: List[str]
    style: str
    duration_seconds: float
    file_size_mb: float
    voices: Dict[str, str]
    generated_at: str
    generation_cost: Dict[str, float]
    key_topics: List[str]

@dataclass
class PodcastResult:
    """Complete podcast generation result"""
    podcast_id: str
    audio_file_path: str
    transcript: str
    metadata: PodcastMetadata
    generation_time: float
    success: bool
    error: Optional[str] = None


class PodcastGeneratorService:
    """
    Service for generating conversational podcasts from documents.
    Combines LlamaIndex for analysis and ElevenLabs for voice synthesis.
    """
    
    # Word count per minute for podcast pacing
    WORDS_PER_MINUTE = 150
    
    # Script generation prompts for different styles
    SCRIPT_PROMPTS = {
        "conversational": """You are an expert podcast script writer. Create an engaging 2-host podcast discussing insights from documents.

CONTEXT:
{analysis}

REQUIREMENTS:
- Duration: {duration_minutes} minutes (approximately {word_count} words)
- Style: Conversational, friendly, and accessible
- Format: Alternating dialogue between HOST1 and HOST2
- Include natural transitions, questions, and "aha!" moments
- Make complex topics easy to understand
- Add enthusiasm and genuine curiosity
- Balance speaking time between both hosts

DIALOGUE FORMAT (strictly follow):
HOST1: [What they say]
HOST2: [What they say]

STRUCTURE:
1. Opening Hook (30 seconds): Grab attention with an intriguing question or fact
2. Introduction (1 minute): Set context and preview what's coming
3. Main Discussion (70% of time): Deep dive into key insights
4. Wrap-up (1 minute): Summarize key takeaways and final thoughts

TONE: Friendly, enthusiastic, educational but not condescending

Generate the complete podcast script now:""",
        
        "educational": """You are creating an educational podcast script. Two hosts discuss document insights in a clear, instructive manner.

CONTEXT:
{analysis}

REQUIREMENTS:
- Duration: {duration_minutes} minutes (approximately {word_count} words)
- Style: Clear, methodical, educational
- HOST1 acts as the teacher/expert, HOST2 as the curious learner
- Include explanations of complex concepts
- Use examples and analogies
- Build knowledge progressively

DIALOGUE FORMAT:
HOST1: [Expert explanation]
HOST2: [Clarifying question or observation]

Generate the complete educational podcast script now:""",
        
        "technical": """You are writing a technical podcast for an informed audience. Discuss document insights with precision and depth.

CONTEXT:
{analysis}

REQUIREMENTS:
- Duration: {duration_minutes} minutes (approximately {word_count} words)
- Style: Professional, detailed, technically accurate
- HOST1 is the subject matter expert, HOST2 is an informed interviewer
- Use proper technical terminology
- Dive into implementation details
- Discuss implications and applications

DIALOGUE FORMAT:
HOST1: [Technical insight]
HOST2: [Probing question]

Generate the complete technical podcast script now:""",
        
        "casual": """You are creating a fun, casual podcast. Two friends discuss interesting ideas from documents.

CONTEXT:
{analysis}

REQUIREMENTS:
- Duration: {duration_minutes} minutes (approximately {word_count} words)
- Style: Relaxed, humorous, energetic
- Both hosts are enthusiastic and engaged
- Use casual language and occasional humor
- Make it entertaining while staying informative
- Quick pacing with energy

DIALOGUE FORMAT:
HOST1: [Casual commentary]
HOST2: [Enthusiastic response]

Generate the complete casual podcast script now:"""
    }
    
    def __init__(
        self,
        llamaindex_service: LlamaIndexService,
        llm_service: LLMService,
        elevenlabs_api_key: Optional[str] = None,
        pii_service: Any = None
    ):
        """
        Initialize podcast generator service

        Args:
            llamaindex_service: Service for document analysis (fallback path)
            llm_service: Service for script generation
            elevenlabs_api_key: ElevenLabs API key (uses config if not provided)
            pii_service: PII boundary - analysis inputs are masked, deliverables restored
        """
        self.config = config.config
        self.llamaindex_service = llamaindex_service
        self.llm_service = llm_service
        self.pii_service = pii_service

        # Initialize ElevenLabs client
        self.elevenlabs_client = None
        if ELEVENLABS_AVAILABLE:
            api_key = elevenlabs_api_key or self.config.ELEVENLABS_API_KEY
            if api_key:
                try:
                    self.elevenlabs_client = ElevenLabs(api_key=api_key)
                    logger.info("ElevenLabs client initialized for podcast generation")
                except Exception as e:
                    logger.error(f"Failed to initialize ElevenLabs client: {e}")

        # Create podcast storage directory (NO ./data hardcode: on HF Spaces
        # Persistent Storage the tree lives under /data; keep them identical)
        self.podcast_dir = Path(self.config.DATA_DIR) / "podcasts"
        self.podcast_dir.mkdir(parents=True, exist_ok=True)

        # Metadata database file
        self.metadata_file = self.podcast_dir / "metadata_db.json"
        self._ensure_metadata_db()
    
    def _ensure_metadata_db(self):
        """Ensure metadata database exists"""
        if not self.metadata_file.exists():
            self.metadata_file.write_text(json.dumps([], indent=2))
    
    async def generate_podcast(
        self,
        document_ids: List[str],
        style: str = "conversational",
        duration_minutes: int = 10,
        host1_voice: str = "Rachel",
        host2_voice: str = "Adam"
    ) -> PodcastResult:
        """
        Generate a complete podcast from documents
        
        Args:
            document_ids: List of document IDs to analyze
            style: Podcast style (conversational, educational, technical, casual)
            duration_minutes: Target duration in minutes
            host1_voice: Voice name for first host
            host2_voice: Voice name for second host
        
        Returns:
            PodcastResult with audio file path and metadata
        """
        start_time = datetime.now()
        podcast_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting podcast generation {podcast_id}")
            logger.info(f"Documents: {document_ids}, Style: {style}, Duration: {duration_minutes}min")
            
            # Step 1: Analyze documents
            logger.info("Step 1: Analyzing documents...")
            analysis = await self.analyze_documents(document_ids)
            
            # Step 2: Generate script
            logger.info("Step 2: Generating podcast script...")
            script = await self.generate_script(analysis, style, duration_minutes)
            
            # Step 3: Synthesize audio
            logger.info("Step 3: Synthesizing audio with voices...")
            audio_file_path = await self.synthesize_audio(
                podcast_id,
                script,
                host1_voice,
                host2_voice
            )
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()

            # Step 4: Create metadata
            # Voices kept as an ORDERED dict - the old `set` collapsed equal
            # names and re-ordered host1/host2 nondeterministically, so saved
            # metadata could swap the voices entirely.
            voices = {"host1": host1_voice, "host2": host2_voice}
            metadata = self._create_metadata(
                podcast_id,
                analysis,
                script,
                audio_file_path,
                voices,
                document_ids,
                style
            )
            
            # Save metadata
            self._save_metadata(metadata)
            
            # Save transcript (user-facing view: placeholders unmasked)
            transcript_text = self._restore_for_user(script.to_text())
            transcript_path = self.podcast_dir / f"{podcast_id}_transcript.txt"
            transcript_path.write_text(transcript_text, encoding="utf-8")

            logger.info(f"Podcast generated successfully: {podcast_id}")

            return PodcastResult(
                podcast_id=podcast_id,
                audio_file_path=str(audio_file_path),
                transcript=transcript_text,
                metadata=metadata,
                generation_time=generation_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Podcast generation failed: {str(e)}", exc_info=True)
            return PodcastResult(
                podcast_id=podcast_id,
                audio_file_path="",
                transcript="",
                metadata=None,
                generation_time=(datetime.now() - start_time).total_seconds(),
                success=False,
                error=str(e)
            )
    
    async def analyze_documents(self, document_ids: List[str]) -> DocumentAnalysis:
        """Analyze documents to extract key insights for the podcast.

        Reads the actual stored content directly (bounded) - the old path fed
        the agentic agent a list of *document IDs it has no way to filter on*,
        so the agent answered from whatever retrieval pulled and could go
        hallucinating against the wrong docs.
        """
        MAX_DOCS = 8
        MAX_CHARS_PER_DOC = 8_000
        MAX_TOTAL_CHARS = 40_000

        snippets: List[str] = []
        loaded_ids: List[str] = []
        total_chars = 0
        for doc_id in document_ids[:MAX_DOCS]:
            try:
                doc = await self._load_doc_info(doc_id)
                content = (doc or {}).get("content") or ""
                if not content:
                    continue
                take = min(len(content), MAX_CHARS_PER_DOC, max(0, MAX_TOTAL_CHARS - total_chars))
                if take <= 0:
                    break
                total_chars += take
                loaded_ids.append(doc_id)
                name = doc.get("filename") or doc_id[:8]
                snippets.append(f"=== DOCUMENT {name} ===\n{content[:take]}")
            except Exception as exc:
                logger.warning("Could not load document %s for podcast: %s", doc_id[:8], exc)

        if not snippets:
            # Fallback to the agentic agent when direct loads fail entirely -
            # and fail LOUDLY on a broken pipeline (no garbage podcasts).
            result = await self.llamaindex_service.query_strict(
                "Summarize the library's most substantial documents: 5-7 key "
                "insights, main themes, complexity level, and a brief summary."
            )
            return self._analysis_from_text(result, document_ids[:MAX_DOCS])

        combined = "\n\n".join(snippets)
        # PII boundary: analysis prompt content is masked; real values are
        # restored in the transcript/audio at delivery time.
        masked_combined = self._mask_for_pipeline(combined)
        analysis_text = await self.llm_service.generate_text(
            "You are preparing material for a podcast. From the following"
            " library documents, provide:\n"
            "1. The 5-7 most important insights (numbered list)\n"
            "2. Main themes and topics\n"
            "3. Overall complexity (beginner/intermediate/advanced)\n"
            "4. A brief summary suitable for podcast discussion\n\n"
            f"{masked_combined}",
            max_tokens=1200,
        )
        return self._analysis_from_text(analysis_text, loaded_ids or document_ids[:MAX_DOCS])

    async def _load_doc_info(self, doc_id: str) -> Dict[str, Any]:
        """Fetch {filename, content} for a document via the shared doc store."""
        store = self.llamaindex_service.document_store
        info: Dict[str, Any] = {}
        fetch = getattr(store, "get_document", None)
        if fetch:
            doc = await fetch(doc_id)
            info = doc.to_dict() if doc and hasattr(doc, "to_dict") else {}
            info.setdefault("filename", getattr(doc, "filename", None))
        loader = getattr(store, "load_content", None)
        if loader:
            info["content"] = await loader(doc_id) or ""
        return info

    def _analysis_from_text(self, text: str, doc_ids: List[str]) -> DocumentAnalysis:
        insights = self._extract_insights(text)
        topics = self._extract_topics(text)
        complexity = self._determine_complexity(text)
        return DocumentAnalysis(
            key_insights=insights[:7],  # Limit to 7
            topics=topics,
            complexity_level=complexity,
            estimated_words=len((text or "").split()),
            source_documents=doc_ids,
            summary=text or "",
        )

    def _mask_for_pipeline(self, text: str) -> str:
        """Mask sensitive values before they reach any LLM in this pipeline."""
        if not text or not self.pii_service:
            return text
        try:
            from services.mcp_pii import AGENTIC_PII_NAMESPACE

            return self.pii_service.mask_for_agent(text, namespace=AGENTIC_PII_NAMESPACE).masked_text
        except Exception as exc:
            logger.warning("Podcast PII masking failed (failing closed): %s", exc)
            return self._strip_likely_pii(text)

    def _restore_for_user(self, text: str) -> str:
        """Unmask placeholders in a user-facing deliverable (transcript/audio)."""
        if not text or not self.pii_service:
            return text
        try:
            from services.mcp_pii import AGENTIC_PII_NAMESPACE

            return self.pii_service.restore_for_user(text, AGENTIC_PII_NAMESPACE)
        except Exception:
            return text

    def _strip_likely_pii(self, text: str) -> str:
        """Emergency scrub when masking is impossible (never best-effort leak)."""
        patterns = [
            r"[\w.+-]+@[\w-]+\.[\w.-]+",          # emails
            r"(?:\+\d{1,3}[\s-]?)?\d{10}",         # 10-digit phones
            r"\b[A-Z]{5}\d{4}[A-Z]\b",             # PAN
            r"\b\d{4}\s?\d{4}\s?\d{4}\b",          # Aadhaar-shaped
            r"\b(?:\d[ -]*?){13,19}\b",            # card-shaped
        ]
        scrubbed = text
        for pattern in patterns:
            scrubbed = re.sub(pattern, "[REDACTED]", scrubbed)
        return scrubbed
    
    def _extract_insights(self, text: str) -> List[str]:
        """Extract key insights from analysis text"""
        insights = []
        #Simple extraction based on numbered lists or bullet points
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Match patterns like "1.", "2.", "-", "*", "•"
            if re.match(r'^\d+\.|\-|\*|•', line):
                insight = re.sub(r'^\d+\.|\-|\*|•', '', line).strip()
                if len(insight) > 20:  # Ensure it's substantial
                    insights.append(insight)
        
        # If no insights found, create from first few sentences
        if not insights:
            sentences = text.split('.')
            insights = [s.strip() + '.' for s in sentences[:7] if len(s.strip()) > 20]
        
        return insights
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from analysis"""
        # Simple keyword extraction - could be enhanced with NLP
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.lower().split()
        word_freq = {}
        
        for word in words:
            word = re.sub(r'[^\w\s]', '', word)
            if len(word) > 4 and word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top topics
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [topic[0].title() for topic in topics]
    
    def _determine_complexity(self, text: str) -> str:
        """Determine content complexity level"""
        text_lower = text.lower()
        
        # Simple heuristic based on keywords
        if any(word in text_lower for word in ['basic', 'introduction', 'beginner', 'simple']):
            return "beginner"
        elif any(word in text_lower for word in ['advanced', 'complex', 'sophisticated', 'expert']):
            return "advanced"
        else:
            return "intermediate"
    
    async def generate_script(
        self,
        analysis: DocumentAnalysis,
        style: str,
        duration_minutes: int
    ) -> PodcastScript:
        """
        Generate podcast script from analysis
        
        Args:
            analysis: Document analysis results
            style: Podcast style
            duration_minutes: Target duration
        
        Returns:
            Complete podcast script
        """
        # Calculate target word count
        target_words = duration_minutes * self.WORDS_PER_MINUTE
        
        # Prepare analysis context
        analysis_context = f"""
KEY INSIGHTS:
{chr(10).join(f"{i+1}. {insight}" for i, insight in enumerate(analysis.key_insights))}

TOPICS: {', '.join(analysis.topics)}
COMPLEXITY: {analysis.complexity_level}

SUMMARY:
{analysis.summary[:500]}...
"""
        
        # Get prompt template for style
        prompt_template = self.SCRIPT_PROMPTS.get(style, self.SCRIPT_PROMPTS["conversational"])
        
        # Fill in the template
        prompt = prompt_template.format(
            analysis=analysis_context,
            duration_minutes=duration_minutes,
            word_count=target_words
        )
        
        # Generate script using LLM
        script_text = await self.llm_service.generate_text(
            prompt,
            max_tokens=target_words * 2,  # Give room for generation
            temperature=0.8  # More creative
        )
        
        # Parse script into dialogue lines
        dialogue = self._parse_script(script_text)
        
        # Calculate actual word count and duration
        word_count = sum(len(line.text.split()) for line in dialogue)
        duration_estimate = word_count / self.WORDS_PER_MINUTE
        
        return PodcastScript(
            dialogue=dialogue,
            total_duration_estimate=duration_estimate * 60,  # Convert to seconds
            word_count=word_count,
            style=style
        )
    
    def _parse_script(self, script_text: str) -> List[DialogueLine]:
        """Parse generated script into dialogue lines.

        LLMs print the speaker in all sorts of shapes: `HOST1:`, `**HOST1:**`,
        `Host 1:`, `1. HOST1:` - if the parser missed those, the podcast had
        ZERO dialogue lines and the audio ended up empty. Tolerant regex +
        markdown emphasis stripping fix that.
        """
        dialogue: List[DialogueLine] = []
        header = re.compile(
            r"^(?:\d+[\).]\s*)?\**\s*HOST\s*[- ]?\s*([12])\*?\s*:\s*(.*)$",
            re.IGNORECASE,
        )
        current: Optional[DialogueLine] = None
        for raw in (script_text or "").split("\n"):
            line = raw.strip()
            if not line:
                continue
            match = header.match(line)
            if match:
                speaker = f"HOST{match.group(1)}"
                text = match.group(2).strip()
                text = re.sub(r"\*+", "", text).strip()  # markdown emphasis
                current = DialogueLine(speaker=speaker, text=text) if text else None
                if current:
                    dialogue.append(current)
            elif current:
                # continuation of the previous host's line
                text = re.sub(r"\*+", "", line).strip()
                if text:
                    current.text = (current.text + " " + text).strip()
        return dialogue
    
    def _get_voice_id(self, voice_name: str) -> Optional[str]:
        """
        Get voice ID from voice name.
        Falls back to first available voice if not found.

        Args:
            voice_name: Voice name (e.g., "Rachel", "Adam")

        Returns:
            Voice ID string, or None when no voices are reachable.
        """
        try:
            # Try to get voices and find by name
            voices = self.elevenlabs_client.voices.get_all()

            if not voices or not voices.voices:
                logger.error("No voices available from ElevenLabs")
                return None

            # First, try exact name match
            for voice in voices.voices:
                if voice.name.lower() == voice_name.lower():
                    logger.info(f"Found exact voice match for '{voice_name}': {voice.voice_id}")
                    return voice.voice_id

            # Try partial match
            for voice in voices.voices:
                if voice_name.lower() in voice.name.lower():
                    logger.info(f"Found partial voice match for '{voice_name}': {voice.name} ({voice.voice_id})")
                    return voice.voice_id

            # Use first available voice as fallback
            first_voice = voices.voices[0]
            logger.warning(f"Voice '{voice_name}' not found, using first available voice: {first_voice.name} ({first_voice.voice_id})")
            return first_voice.voice_id

        except Exception as e:
            logger.error(f"Could not fetch voices: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------ TTS (sync)
    def _tts_line_sync(self, voice_id: str, text: str) -> bytes:
        """Blocking ElevenLabs call for one dialogue line -> raw mp3 bytes."""
        audio_generator = self.elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
        )
        return b"".join(chunk for chunk in audio_generator if chunk)

    async def _voice_id_async(self, host_voice: str) -> Optional[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_voice_id, host_voice)

    async def synthesize_audio(
        self,
        podcast_id: str,
        script: PodcastScript,
        host1_voice: str,
        host2_voice: str
    ) -> Path:
        """Synthesize audio with each host in its OWN voice.

        The old one-shot version fed `HOST1:/HOST2: ...` labels into a single
        voice for the whole show (nobody could tell the hosts apart).
        Per-line synthesis with per-host voice id, plus mp3 concatenation of
        frame-aligned streams; every blocking network call runs in an
        executor so the shared event loop never stalls.
        """
        if not self.elevenlabs_client:
            raise RuntimeError("ElevenLabs client not initialized")

        audio_file = self.podcast_dir / f"{podcast_id}.mp3"
        loop = asyncio.get_running_loop()

        # Deliverables never speak placeholders aloud: unmask before TTS.
        lines = [
            (line.speaker, self._restore_for_user(line.text))
            for line in script.dialogue
        ]
        if not lines:
            raise RuntimeError("Podcast script has no dialogue lines to synthesize")

        voice_ids: Dict[str, Optional[str]] = {
            "HOST1": await self._voice_id_async(host1_voice),
            "HOST2": await self._voice_id_async(host2_voice),
        }
        if not voice_ids.get("HOST1"):
            raise RuntimeError("Could not resolve any ElevenLabs voice for HOST1")

        logger.info("Synthesizing %d lines (voices: host1=%s, host2=%s)",
                    len(lines), voice_ids["HOST1"][:8], voice_ids.get("HOST2", "?")[:8])

        segments: List[bytes] = []
        for i, (speaker, text) in enumerate(lines, 1):
            speaker_voice = voice_ids.get(speaker) or voice_ids["HOST1"]
            logger.info("TTS %s/%s %s (%d chars)", i, len(lines), speaker, len(text))
            segments.append(await loop.run_in_executor(
                None, lambda sv=speaker_voice, t=text: self._tts_line_sync(sv, t)
            ))

        # mp3 frames are self-contained; straight concatenation yields a
        # decodable stream (natural pause comes from the host turnaround).
        audio_file.write_bytes(b"".join(segments))

        if audio_file.exists() and audio_file.stat().st_size > 1000:
            logger.info(f"Audio synthesized successfully: {audio_file} ({audio_file.stat().st_size} bytes)")
            return audio_file
        raise RuntimeError(f"Generated audio file is too small or empty: {audio_file.stat().st_size} bytes")
    
    def _create_metadata(
        self,
        podcast_id: str,
        analysis: DocumentAnalysis,
        script: PodcastScript,
        audio_path: Path,
        voices: Dict[str, str],
        document_ids: List[str],
        style: str
    ) -> PodcastMetadata:
        """Create podcast metadata"""
        # Auto-generate title
        title = f"Podcast: {analysis.topics[0] if analysis.topics else 'Document Discussion'}"

        # Create description
        description = f"A {style} podcast discussing insights from {len(document_ids)} document(s)."

        # Calculate file size
        file_size_mb = audio_path.stat().st_size / (1024 * 1024) if audio_path and Path(str(audio_path)).exists() else 0

        # Estimate costs
        llm_cost = (script.word_count / 1000) * 0.01  # Rough estimate
        tts_cost = (script.word_count * 5 / 1000) * 0.30  # Rough estimate

        return PodcastMetadata(
            podcast_id=podcast_id,
            title=title,
            description=description,
            source_documents=document_ids,
            style=style,
            duration_seconds=script.total_duration_estimate,
            file_size_mb=file_size_mb,
            voices=dict(voices),
            generated_at=datetime.now().isoformat(),
            generation_cost={"llm_cost": llm_cost, "tts_cost": tts_cost, "total": llm_cost + tts_cost},
            key_topics=analysis.topics
        )
    
    def _save_metadata(self, metadata: PodcastMetadata):
        """Save metadata to database"""
        try:
            # Load existing metadata
            existing = json.loads(self.metadata_file.read_text())
            
            # Add new metadata
            existing.append(asdict(metadata))
            
            # Save back
            self.metadata_file.write_text(json.dumps(existing, indent=2))
            
            logger.info(f"Metadata saved for podcast: {metadata.podcast_id}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def list_podcasts(self, limit: int = 10) -> List[PodcastMetadata]:
        """List generated podcasts"""
        try:
            data = json.loads(self.metadata_file.read_text())
            podcasts = [PodcastMetadata(**item) for item in data[-limit:]]
            return list(reversed(podcasts))  # Most recent first
        except Exception as e:
            logger.error(f"Failed to list podcasts: {e}")
            return []
    
    def get_podcast(self, podcast_id: str) -> Optional[PodcastMetadata]:
        """Get specific podcast metadata"""
        try:
            data = json.loads(self.metadata_file.read_text())
            for item in data:
                if item.get('podcast_id') == podcast_id:
                    return PodcastMetadata(**item)
            return None
        except Exception as e:
            logger.error(f"Failed to get podcast: {e}")
            return None

