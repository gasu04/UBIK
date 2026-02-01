"""
UBIK Ingestion System - Transcript Processor

Specialized processor for meeting/therapy session transcripts with
speaker attribution and rich metadata extraction.

Supported Formats:
    - YAML front matter with transcript body
    - Companion .meta.yaml files
    - Various speaker formats (Name:, [Name], Otter.ai, etc.)

Features:
    - Speaker mapping and attribution
    - Turn-based chunking with overlap
    - Meeting metadata extraction
    - Neo4j graph operation generation
    - Action item and decision tracking

Usage:
    from ingest.transcript_processor import TranscriptProcessor

    processor = TranscriptProcessor()
    result = await processor.process(ingest_item, transcript_path)

    # Access meeting metadata
    metadata = result.extracted_metadata["meeting_metadata"]
    neo4j_ops = result.extracted_metadata["neo4j_operations"]

Version: 0.1.0
"""

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .chunkers import Chunk
from .models import (
    ContentType,
    IngestItem,
    MemoryCandidate,
    MemoryType,
    ProcessedContent,
)
from .processors import BaseProcessor, ProcessorConfig

__all__ = [
    'MeetingMetadata',
    'ActionItem',
    'TranscriptTurn',
    'TranscriptChunk',
    'SpeakerTurnParser',
    'TranscriptChunker',
    'TranscriptProcessor',
    'parse_front_matter',
    'find_companion_metadata',
    'transcript_to_memory_candidates',
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ActionItem:
    """
    An action item from a meeting.

    Attributes:
        who: Person responsible
        what: Action to take
        due: Due date or timeframe
        status: Current status (pending, completed, blocked)
    """
    who: str
    what: str
    due: Optional[str] = None
    status: str = "pending"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionItem":
        """Create ActionItem from dictionary."""
        return cls(
            who=data.get("who", ""),
            what=data.get("what", ""),
            due=data.get("due"),
            status=data.get("status", "pending"),
        )


@dataclass
class MeetingMetadata:
    """
    Rich metadata for meeting/therapy transcripts.

    Attributes:
        meeting_date: When the meeting occurred
        meeting_type: Type of meeting (family_meeting, therapy, etc.)
        participants: List of participant names
        speakers: Mapping of raw speaker labels to actual names
        location: Where the meeting took place
        duration_minutes: Length of meeting
        recorder: Who recorded/transcribed
        meeting_title: Title or subject
        agenda_topics: Planned discussion topics
        actual_topics: Topics actually discussed
        decisions_made: Key decisions from meeting
        action_items: Action items assigned
        emotional_tone: Overall emotional atmosphere
        relationship_context: Relationship dynamics present
        transcript_quality: Quality of transcription
        contains_sensitive: Whether content is sensitive
        language: Primary language of transcript

    Example:
        metadata = MeetingMetadata.from_dict(yaml_data)
        chroma_meta = metadata.to_chromadb_metadata()
        neo4j_ops = metadata.to_neo4j_operations()
    """
    meeting_date: Optional[datetime] = None
    meeting_type: str = "conversation"
    participants: List[str] = field(default_factory=list)
    speakers: Dict[str, str] = field(default_factory=dict)
    location: Optional[str] = None
    duration_minutes: Optional[int] = None
    recorder: str = "unknown"
    meeting_title: Optional[str] = None
    agenda_topics: List[str] = field(default_factory=list)
    actual_topics: List[str] = field(default_factory=list)
    decisions_made: List[str] = field(default_factory=list)
    action_items: List[ActionItem] = field(default_factory=list)
    emotional_tone: str = "neutral"
    relationship_context: str = "general"
    transcript_quality: str = "good"
    contains_sensitive: bool = False
    language: str = "en"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeetingMetadata":
        """
        Create MeetingMetadata from dictionary (e.g., YAML front matter).

        Handles various date formats and nested structures.
        """
        # Parse date (YAML may parse as date, datetime, or string)
        meeting_date = None
        if "meeting_date" in data:
            date_val = data["meeting_date"]
            if isinstance(date_val, datetime):
                meeting_date = date_val
            elif isinstance(date_val, date):
                # Convert date to datetime
                meeting_date = datetime.combine(date_val, datetime.min.time())
            elif isinstance(date_val, str):
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]:
                    try:
                        meeting_date = datetime.strptime(date_val, fmt)
                        break
                    except ValueError:
                        continue

        # Parse action items
        action_items = []
        for item in data.get("action_items", []):
            if isinstance(item, dict):
                action_items.append(ActionItem.from_dict(item))
            elif isinstance(item, str):
                action_items.append(ActionItem(who="", what=item))

        return cls(
            meeting_date=meeting_date,
            meeting_type=data.get("meeting_type", "conversation"),
            participants=data.get("participants", []),
            speakers=data.get("speakers", data.get("speaker_mapping", {})),
            location=data.get("location"),
            duration_minutes=data.get("duration_minutes", data.get("duration")),
            recorder=data.get("recorder", data.get("transcribed_by", "unknown")),
            meeting_title=data.get("meeting_title", data.get("title")),
            agenda_topics=data.get("agenda_topics", data.get("agenda", [])),
            actual_topics=data.get("actual_topics", data.get("topics", [])),
            decisions_made=data.get("decisions_made", data.get("decisions", [])),
            action_items=action_items,
            emotional_tone=data.get("emotional_tone", "neutral"),
            relationship_context=data.get("relationship_context", "general"),
            transcript_quality=data.get("transcript_quality", "good"),
            contains_sensitive=data.get("contains_sensitive", False),
            language=data.get("language", "en"),
        )

    def to_chromadb_metadata(self) -> Dict[str, Any]:
        """
        Convert to ChromaDB-compatible metadata.

        ChromaDB supports: str, int, float, bool
        Lists are converted to comma-separated strings.
        """
        meta = {
            "meeting_type": self.meeting_type,
            "emotional_tone": self.emotional_tone,
            "relationship_context": self.relationship_context,
            "transcript_quality": self.transcript_quality,
            "contains_sensitive": self.contains_sensitive,
            "language": self.language,
            "recorder": self.recorder,
        }

        if self.meeting_date:
            meta["meeting_date"] = self.meeting_date.isoformat()

        if self.location:
            meta["location"] = self.location

        if self.duration_minutes:
            meta["duration_minutes"] = self.duration_minutes

        if self.meeting_title:
            meta["meeting_title"] = self.meeting_title

        if self.participants:
            meta["participants"] = ",".join(self.participants)

        if self.actual_topics:
            meta["topics"] = ",".join(self.actual_topics)

        if self.decisions_made:
            meta["decisions_count"] = len(self.decisions_made)

        if self.action_items:
            meta["action_items_count"] = len(self.action_items)

        return meta

    def to_neo4j_operations(self) -> List[Dict[str, Any]]:
        """
        Generate Neo4j graph operations for this meeting.

        Creates nodes for:
        - Meeting event
        - Participants (Person nodes)
        - Topics discussed
        - Decisions made

        Creates relationships:
        - PARTICIPATED_IN
        - DISCUSSED
        - DECIDED
        - ASSIGNED_TO (for action items)
        """
        operations = []
        meeting_id = f"meeting_{self.meeting_date.isoformat() if self.meeting_date else 'unknown'}_{hash(self.meeting_title or '')}"

        # Create Meeting node
        operations.append({
            "operation": "merge_node",
            "label": "Meeting",
            "properties": {
                "id": meeting_id,
                "type": self.meeting_type,
                "date": self.meeting_date.isoformat() if self.meeting_date else None,
                "title": self.meeting_title,
                "location": self.location,
                "duration_minutes": self.duration_minutes,
                "emotional_tone": self.emotional_tone,
            },
        })

        # Create Person nodes and PARTICIPATED_IN relationships
        for participant in self.participants:
            person_id = f"person_{participant.lower().replace(' ', '_')}"

            operations.append({
                "operation": "merge_node",
                "label": "Person",
                "properties": {
                    "id": person_id,
                    "name": participant,
                },
            })

            operations.append({
                "operation": "merge_relationship",
                "from_label": "Person",
                "from_id": person_id,
                "to_label": "Meeting",
                "to_id": meeting_id,
                "rel_type": "PARTICIPATED_IN",
                "properties": {
                    "date": self.meeting_date.isoformat() if self.meeting_date else None,
                },
            })

        # Create Topic nodes and DISCUSSED relationships
        for topic in self.actual_topics:
            topic_id = f"topic_{topic.lower().replace(' ', '_')[:50]}"

            operations.append({
                "operation": "merge_node",
                "label": "Topic",
                "properties": {
                    "id": topic_id,
                    "name": topic,
                },
            })

            operations.append({
                "operation": "merge_relationship",
                "from_label": "Meeting",
                "from_id": meeting_id,
                "to_label": "Topic",
                "to_id": topic_id,
                "rel_type": "DISCUSSED",
                "properties": {},
            })

        # Create Decision nodes
        for i, decision in enumerate(self.decisions_made):
            decision_id = f"decision_{meeting_id}_{i}"

            operations.append({
                "operation": "merge_node",
                "label": "Decision",
                "properties": {
                    "id": decision_id,
                    "content": decision,
                    "date": self.meeting_date.isoformat() if self.meeting_date else None,
                },
            })

            operations.append({
                "operation": "merge_relationship",
                "from_label": "Meeting",
                "from_id": meeting_id,
                "to_label": "Decision",
                "to_id": decision_id,
                "rel_type": "DECIDED",
                "properties": {},
            })

        # Create ActionItem nodes and relationships
        for i, action in enumerate(self.action_items):
            action_id = f"action_{meeting_id}_{i}"

            operations.append({
                "operation": "merge_node",
                "label": "ActionItem",
                "properties": {
                    "id": action_id,
                    "what": action.what,
                    "due": action.due,
                    "status": action.status,
                },
            })

            operations.append({
                "operation": "merge_relationship",
                "from_label": "Meeting",
                "from_id": meeting_id,
                "to_label": "ActionItem",
                "to_id": action_id,
                "rel_type": "CREATED_ACTION",
                "properties": {},
            })

            # Link to assignee if specified
            if action.who:
                person_id = f"person_{action.who.lower().replace(' ', '_')}"
                operations.append({
                    "operation": "merge_relationship",
                    "from_label": "ActionItem",
                    "from_id": action_id,
                    "to_label": "Person",
                    "to_id": person_id,
                    "rel_type": "ASSIGNED_TO",
                    "properties": {},
                })

        return operations


@dataclass
class TranscriptTurn:
    """
    A single speaker turn in a transcript.

    Attributes:
        speaker_raw: Original speaker label (e.g., "Speaker 1")
        speaker_name: Resolved speaker name (e.g., "Gines")
        text: What was said
        start_time: Start timestamp (if available)
        end_time: End timestamp (if available)
    """
    speaker_raw: str
    speaker_name: str
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds if timestamps available."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


@dataclass
class TranscriptChunk(Chunk):
    """
    Extended chunk with transcript-specific metadata.

    Inherits from Chunk and adds speaker attribution.

    Attributes:
        speakers_in_chunk: Set of speaker names in this chunk
        turn_count: Number of speaker turns
        topic: Detected topic of this segment
        meeting_metadata: Reference to full meeting metadata
    """
    speakers_in_chunk: List[str] = field(default_factory=list)
    turn_count: int = 0
    topic: Optional[str] = None
    meeting_metadata: Optional[MeetingMetadata] = None


# =============================================================================
# Helper Functions
# =============================================================================

def parse_front_matter(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Parse YAML front matter from transcript content.

    Front matter is delimited by --- markers:

        ---
        meeting_date: 2024-01-15
        participants: [Alice, Bob]
        ---

        Transcript content here...

    Args:
        content: Full transcript content with potential front matter

    Returns:
        Tuple of (metadata_dict, body_text)
        If no front matter, returns ({}, content)
    """
    content = content.strip()

    # Check for front matter delimiter
    if not content.startswith("---"):
        return {}, content

    # Find closing delimiter
    lines = content.split("\n")
    end_idx = -1

    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        return {}, content

    # Parse YAML
    front_matter_text = "\n".join(lines[1:end_idx])
    body_text = "\n".join(lines[end_idx + 1:]).strip()

    try:
        metadata = yaml.safe_load(front_matter_text) or {}
    except yaml.YAMLError:
        metadata = {}

    return metadata, body_text


def find_companion_metadata(transcript_path: Path) -> Optional[Path]:
    """
    Find companion metadata file for a transcript.

    Looks for files with same base name and .meta.yaml or .yaml extension.

    Search order:
        1. {name}.meta.yaml
        2. {name}_metadata.yaml
        3. {name}.yaml (if transcript is not .yaml)

    Args:
        transcript_path: Path to transcript file

    Returns:
        Path to metadata file if found, None otherwise
    """
    stem = transcript_path.stem
    parent = transcript_path.parent

    candidates = [
        parent / f"{stem}.meta.yaml",
        parent / f"{stem}_metadata.yaml",
        parent / f"{stem}.meta.yml",
        parent / f"{stem}_metadata.yml",
    ]

    # Only check .yaml if transcript isn't already yaml
    if transcript_path.suffix.lower() not in [".yaml", ".yml"]:
        candidates.append(parent / f"{stem}.yaml")
        candidates.append(parent / f"{stem}.yml")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


# =============================================================================
# Speaker Turn Parser
# =============================================================================

class SpeakerTurnParser:
    """
    Parses speaker turns from various transcript formats.

    Supported Formats:
        - "Name: text"
        - "**Name**: text" (Markdown bold)
        - "[Name] text"
        - "NAME: text" (all caps)
        - Otter.ai: "Speaker 1  0:00\\ntext"
        - Timestamps: "00:00:00 Name: text"

    Attributes:
        speaker_mapping: Dict mapping raw labels to actual names

    Example:
        parser = SpeakerTurnParser({"Speaker 1": "Gines", "Speaker 2": "Adrian"})
        turns = parser.parse(transcript_text)
    """

    # Pattern definitions
    PATTERNS = [
        # Otter.ai format: "Speaker 1  0:00\ntext" or "Name  0:00\ntext"
        (
            "otter",
            re.compile(
                r'^([A-Za-z][A-Za-z0-9 ]+?)\s{2,}(\d+:\d{2})\s*\n(.+?)(?=\n[A-Za-z][A-Za-z0-9 ]+?\s{2,}\d+:\d{2}|\Z)',
                re.MULTILINE | re.DOTALL
            ),
        ),
        # Markdown bold: "**Name**: text"
        (
            "markdown",
            re.compile(
                r'^\*\*([^*]+)\*\*:\s*(.+?)(?=\n\*\*[^*]+\*\*:|\Z)',
                re.MULTILINE | re.DOTALL
            ),
        ),
        # Bracketed: "[Name] text"
        (
            "bracketed",
            re.compile(
                r'^\[([^\]]+)\]\s*(.+?)(?=\n\[[^\]]+\]|\Z)',
                re.MULTILINE | re.DOTALL
            ),
        ),
        # Timestamped: "00:00:00 Name: text" or "(00:00) Name: text"
        (
            "timestamped",
            re.compile(
                r'^[\(\[]?(\d{1,2}:\d{2}(?::\d{2})?)[\)\]]?\s+([A-Za-z][A-Za-z ]+?):\s*(.+?)(?=\n[\(\[]?\d{1,2}:\d{2}|\Z)',
                re.MULTILINE | re.DOTALL
            ),
        ),
        # Standard colon: "Name: text" (must be at line start)
        (
            "colon",
            re.compile(
                r'^([A-Z][A-Za-z ]+?):\s*(.+?)(?=\n[A-Z][A-Za-z ]+?:|\Z)',
                re.MULTILINE | re.DOTALL
            ),
        ),
        # All caps: "NAME: text"
        (
            "caps",
            re.compile(
                r'^([A-Z][A-Z ]+):\s*(.+?)(?=\n[A-Z][A-Z ]+:|\Z)',
                re.MULTILINE | re.DOTALL
            ),
        ),
    ]

    def __init__(self, speaker_mapping: Optional[Dict[str, str]] = None):
        """Initialize parser with optional speaker mapping."""
        self.speaker_mapping = speaker_mapping or {}

    def parse(self, text: str) -> List[TranscriptTurn]:
        """
        Parse transcript text into speaker turns.

        Tries each pattern format until one produces results.

        Args:
            text: Raw transcript text

        Returns:
            List of TranscriptTurn objects in order
        """
        text = text.strip()

        for format_name, pattern in self.PATTERNS:
            turns = self._parse_with_pattern(text, pattern, format_name)
            if turns:
                return turns

        # Fallback: treat entire text as single turn
        return [TranscriptTurn(
            speaker_raw="Unknown",
            speaker_name="Unknown",
            text=text,
        )]

    def _parse_with_pattern(
        self,
        text: str,
        pattern: re.Pattern,
        format_name: str
    ) -> List[TranscriptTurn]:
        """Parse using a specific pattern."""
        matches = list(pattern.finditer(text))

        if not matches:
            return []

        turns = []

        for match in matches:
            groups = match.groups()

            if format_name == "otter":
                speaker_raw = groups[0].strip()
                timestamp_str = groups[1]
                turn_text = groups[2].strip()
                start_time = self._parse_timestamp(timestamp_str)
            elif format_name == "timestamped":
                timestamp_str = groups[0]
                speaker_raw = groups[1].strip()
                turn_text = groups[2].strip()
                start_time = self._parse_timestamp(timestamp_str)
            else:
                speaker_raw = groups[0].strip()
                turn_text = groups[1].strip()
                start_time = None

            # Apply speaker mapping
            speaker_name = self.speaker_mapping.get(speaker_raw, speaker_raw)

            turns.append(TranscriptTurn(
                speaker_raw=speaker_raw,
                speaker_name=speaker_name,
                text=turn_text,
                start_time=start_time,
            ))

        # Calculate end times based on next turn's start
        for i in range(len(turns) - 1):
            if turns[i].start_time is not None and turns[i + 1].start_time is not None:
                turns[i].end_time = turns[i + 1].start_time

        return turns

    def _parse_timestamp(self, ts: str) -> Optional[float]:
        """
        Parse timestamp string to seconds.

        Supports:
            - "0:00" -> 0.0
            - "1:30" -> 90.0
            - "01:30:00" -> 5400.0
        """
        parts = ts.split(":")
        try:
            if len(parts) == 2:
                minutes, seconds = int(parts[0]), int(parts[1])
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
                return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            pass
        return None


# =============================================================================
# Transcript Chunker
# =============================================================================

class TranscriptChunker:
    """
    Chunks transcripts by speaker turns with overlap.

    Groups consecutive turns together while respecting size limits.
    Maintains speaker context with turn overlap.

    Attributes:
        turns_per_chunk: Target number of turns per chunk
        max_chunk_chars: Maximum characters per chunk
        overlap_turns: Number of turns to overlap between chunks

    Example:
        chunker = TranscriptChunker(turns_per_chunk=4, overlap_turns=1)
        chunks = chunker.chunk(turns, metadata)
    """

    def __init__(
        self,
        turns_per_chunk: int = 4,
        max_chunk_chars: int = 1500,
        overlap_turns: int = 1
    ):
        """Initialize chunker with configuration."""
        self.turns_per_chunk = turns_per_chunk
        self.max_chunk_chars = max_chunk_chars
        self.overlap_turns = overlap_turns

    def chunk(
        self,
        turns: List[TranscriptTurn],
        metadata: Optional[MeetingMetadata] = None
    ) -> List[TranscriptChunk]:
        """
        Chunk transcript turns into TranscriptChunks.

        Args:
            turns: List of parsed transcript turns
            metadata: Optional meeting metadata

        Returns:
            List of TranscriptChunk objects
        """
        if not turns:
            return []

        chunks = []
        current_turns: List[TranscriptTurn] = []
        current_chars = 0
        char_pos = 0
        chunk_start = 0

        for turn in turns:
            turn_text = f"{turn.speaker_name}: {turn.text}"
            turn_chars = len(turn_text) + 2  # +2 for newlines

            # Check if adding this turn exceeds limits
            would_exceed_turns = len(current_turns) >= self.turns_per_chunk
            would_exceed_chars = current_chars + turn_chars > self.max_chunk_chars

            if current_turns and (would_exceed_turns or would_exceed_chars):
                # Create chunk from current turns
                chunk = self._create_chunk(
                    current_turns, len(chunks), chunk_start, char_pos, metadata
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_count = min(self.overlap_turns, len(current_turns))
                current_turns = current_turns[-overlap_count:] if overlap_count > 0 else []
                current_chars = sum(
                    len(f"{t.speaker_name}: {t.text}") + 2
                    for t in current_turns
                )
                chunk_start = char_pos - current_chars

            current_turns.append(turn)
            current_chars += turn_chars
            char_pos += turn_chars

        # Final chunk
        if current_turns:
            chunk = self._create_chunk(
                current_turns, len(chunks), chunk_start, char_pos, metadata
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        turns: List[TranscriptTurn],
        index: int,
        start_char: int,
        end_char: int,
        metadata: Optional[MeetingMetadata]
    ) -> TranscriptChunk:
        """Create a TranscriptChunk from turns."""
        # Build chunk text
        lines = [f"{t.speaker_name}: {t.text}" for t in turns]
        text = "\n\n".join(lines)

        # Collect speakers
        speakers = list(set(t.speaker_name for t in turns))

        return TranscriptChunk(
            text=text,
            index=index,
            start_char=start_char,
            end_char=end_char,
            chunk_type="transcript",
            speakers_in_chunk=speakers,
            turn_count=len(turns),
            topic=None,  # Could be enhanced with topic detection
            meeting_metadata=metadata,
        )


# =============================================================================
# Transcript Processor
# =============================================================================

class TranscriptProcessor(BaseProcessor):
    """
    Processor for meeting/therapy transcript files.

    Handles .transcript files with optional YAML front matter or
    companion metadata files.

    Supported: .transcript

    Features:
        - YAML front matter parsing
        - Companion .meta.yaml file support
        - Multiple transcript format parsing
        - Speaker mapping and attribution
        - Neo4j graph operation generation
    """

    SUPPORTED_EXTENSIONS = [".transcript"]
    CONTENT_TYPE = ContentType.TEXT

    def __init__(self, config: Optional[ProcessorConfig] = None):
        super().__init__(config)
        self.parser = SpeakerTurnParser()
        self.chunker = TranscriptChunker()

    async def process(
        self,
        item: IngestItem,
        file_path: Path
    ) -> ProcessedContent:
        """
        Process a transcript file.

        1. Read content
        2. Parse front matter or load companion metadata
        3. Parse speaker turns
        4. Chunk by turns
        5. Generate Neo4j operations
        """
        start_time = time.time()

        # Read content
        content = await self._run_in_executor(
            lambda: file_path.read_text(encoding="utf-8")
        )

        # Parse front matter
        front_matter, body = parse_front_matter(content)

        # Check for companion metadata
        if not front_matter:
            companion = find_companion_metadata(file_path)
            if companion:
                companion_text = await self._run_in_executor(
                    lambda: companion.read_text(encoding="utf-8")
                )
                try:
                    front_matter = yaml.safe_load(companion_text) or {}
                except yaml.YAMLError:
                    front_matter = {}

        # Create metadata
        metadata = MeetingMetadata.from_dict(front_matter)

        # Update parser with speaker mapping
        self.parser.speaker_mapping = metadata.speakers

        # Parse turns
        turns = await self._run_in_executor(self.parser.parse, body)

        # Update participants from speakers if not set
        if not metadata.participants and turns:
            metadata.participants = list(set(t.speaker_name for t in turns))

        # Chunk turns
        chunks = await self._run_in_executor(
            self.chunker.chunk, turns, metadata
        )

        # Generate Neo4j operations
        neo4j_ops = metadata.to_neo4j_operations()

        # Build full text from chunks
        full_text = "\n\n---\n\n".join(c.text for c in chunks)

        # Detect title
        title = metadata.meeting_title
        if not title and metadata.meeting_type and metadata.meeting_date:
            title = f"{metadata.meeting_type.replace('_', ' ').title()} - {metadata.meeting_date.strftime('%Y-%m-%d')}"

        return self._create_result(
            item=item,
            text=full_text,
            start_time=start_time,
            title=title,
            language=metadata.language,
            extracted_metadata={
                "meeting_metadata": metadata,
                "chromadb_metadata": metadata.to_chromadb_metadata(),
                "neo4j_operations": neo4j_ops,
                "turns_count": len(turns),
                "chunks_count": len(chunks),
                "speakers": list(set(t.speaker_name for t in turns)),
                "transcript_chunks": chunks,
            },
        )


# =============================================================================
# Memory Candidate Conversion
# =============================================================================

def transcript_to_memory_candidates(
    processed: ProcessedContent
) -> List[MemoryCandidate]:
    """
    Convert processed transcript to memory candidates.

    Creates one MemoryCandidate per transcript chunk with appropriate
    classification based on meeting type and content.

    Args:
        processed: ProcessedContent from TranscriptProcessor

    Returns:
        List of MemoryCandidate objects
    """
    candidates = []
    metadata = processed.extracted_metadata.get("meeting_metadata")
    chunks = processed.extracted_metadata.get("transcript_chunks", [])

    if not chunks:
        return candidates

    # Determine base memory type from meeting type
    meeting_type = metadata.meeting_type if metadata else "conversation"
    base_memory_type = MemoryType.EPISODIC  # Conversations are events

    # Therapy sessions may contain semantic insights
    contains_insights = meeting_type in ["therapy", "coaching"]

    for chunk in chunks:
        if not isinstance(chunk, TranscriptChunk):
            continue

        # Determine themes from topics and content
        themes = []
        if metadata and metadata.actual_topics:
            themes.extend(metadata.actual_topics[:3])

        # Add relationship context
        if metadata and metadata.relationship_context != "general":
            themes.append(metadata.relationship_context)

        # Determine emotional valence
        emotional_valence = "neutral"
        if metadata:
            emotional_valence = _map_tone_to_valence(metadata.emotional_tone)

        # Calculate importance
        importance = 0.6  # Base importance for conversations
        if meeting_type == "therapy":
            importance = 0.8
        elif meeting_type == "family_meeting":
            importance = 0.75

        if metadata and metadata.decisions_made:
            importance = min(importance + 0.1, 1.0)

        candidate = MemoryCandidate(
            content=chunk.text,
            memory_type=base_memory_type,
            confidence=0.85,
            category=meeting_type,
            themes=themes or ["conversation"],
            event_type="conversation",
            participants=chunk.speakers_in_chunk,
            emotional_valence=emotional_valence,
            knowledge_type=None,
            stability="stable",
            source_file=processed.source_item.original_filename,
            source_chunk_index=chunk.index,
            timestamp=metadata.meeting_date if metadata else None,
            importance=importance,
        )

        candidates.append(candidate)

    return candidates


def _map_tone_to_valence(tone: str) -> str:
    """Map emotional tone to valence category."""
    positive_tones = ["happy", "joyful", "hopeful", "grateful", "loving", "warm"]
    negative_tones = ["sad", "angry", "frustrated", "anxious", "tense", "difficult"]
    reflective_tones = ["thoughtful", "contemplative", "serious", "deep"]

    tone_lower = tone.lower()

    if any(t in tone_lower for t in positive_tones):
        return "positive"
    elif any(t in tone_lower for t in negative_tones):
        return "negative"
    elif any(t in tone_lower for t in reflective_tones):
        return "reflective"
    else:
        return "neutral"
