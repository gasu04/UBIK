#!/usr/bin/env python3
"""
UBIK Interactive Ingestion Mode

Human-in-the-loop review for content ingestion with:
- Classification verification and editing
- DPO training data collection (ideal responses)
- Session state persistence for resume capability

This tool is especially valuable for:
- Collecting ideal responses for DPO training
- Verifying classification accuracy on new content types
- Capturing additional metadata during review
- Building high-quality curated memory datasets

Usage:
    python interactive_ingest.py ~/path/to/content
    python interactive_ingest.py ~/path/to/file.txt --output-dir ~/ubik/data/reviewed
    python interactive_ingest.py ~/documents --resume session_20241215_143022

Example Session:
    ┌─────────────────────────────────────────────────────────────┐
    │ Memory 3 of 12                                              │
    ├─────────────────────────────────────────────────────────────┤
    │ Content:                                                    │
    │   "My father taught me to fish at Lake Michigan when I      │
    │   was ten years old. Those summer mornings, the mist        │
    │   rising off the water..."                                  │
    │                                                             │
    │ Classification:                                             │
    │   Type: EPISODIC (confidence: 0.92)                         │
    │   Category: family                                          │
    │   Themes: childhood, father, outdoors                       │
    │   Participants: father                                      │
    │   Emotional Valence: positive                               │
    │   Importance: 0.85                                          │
    ├─────────────────────────────────────────────────────────────┤
    │ [A]ccept  [E]dit  [S]kip  [I]deal response  [Q]uit          │
    └─────────────────────────────────────────────────────────────┘

Version: 1.0.0
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ingest.models import MemoryCandidate, MemoryType
from ingest.pipeline import IngestPipeline, PipelineConfig
from ingest.chunkers import ChunkConfig

__all__ = [
    'InteractiveIngestSession',
    'SessionState',
    'IdealResponse',
]

logger = logging.getLogger(__name__)


# =============================================================================
# Terminal Colors and Formatting
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls):
        """Disable colors for non-TTY output."""
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.RED = ''
        cls.BOLD = ''
        cls.DIM = ''
        cls.UNDERLINE = ''
        cls.RESET = ''


def colorize(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.RESET}"


def print_box(lines: List[str], width: int = 65) -> None:
    """Print text in a box."""
    border = "─" * (width - 2)
    print(f"┌{border}┐")
    for line in lines:
        # Pad or truncate to fit width
        visible_len = len(line.replace(Colors.BOLD, '').replace(Colors.RESET, '')
                         .replace(Colors.GREEN, '').replace(Colors.YELLOW, '')
                         .replace(Colors.RED, '').replace(Colors.CYAN, '')
                         .replace(Colors.BLUE, '').replace(Colors.DIM, ''))
        padding = width - 4 - visible_len
        if padding < 0:
            # Truncate
            line = line[:width - 7] + "..."
            padding = 0
        print(f"│ {line}{' ' * padding} │")
    print(f"└{border}┘")


def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IdealResponse:
    """
    An ideal response paired with a memory candidate.

    Used for DPO (Direct Preference Optimization) training.

    Attributes:
        memory_content: The original memory content
        memory_type: Classification of the memory
        ideal_response: Human-written ideal response
        context: Additional context about the response
        timestamp: When this was collected
        source_file: Original source file
    """
    memory_content: str
    memory_type: str
    ideal_response: str
    context: str
    timestamp: str
    source_file: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SkippedMemory:
    """
    A memory that was skipped during review.

    Attributes:
        content: The memory content
        reason: Why it was skipped
        original_classification: What the system classified it as
        source_file: Original source file
    """
    content: str
    reason: str
    original_classification: str
    source_file: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SessionState:
    """
    Persistent state for resumable sessions.

    Attributes:
        session_id: Unique session identifier
        source_path: Path being processed
        current_file_index: Current file in directory
        current_memory_index: Current memory in file
        processed_files: Files already completed
        total_accepted: Count of accepted memories
        total_edited: Count of edited memories
        total_skipped: Count of skipped memories
        total_ideal: Count of ideal responses collected
        started_at: Session start timestamp
        last_activity: Last activity timestamp
    """
    session_id: str
    source_path: str
    current_file_index: int = 0
    current_memory_index: int = 0
    processed_files: List[str] = field(default_factory=list)
    total_accepted: int = 0
    total_edited: int = 0
    total_skipped: int = 0
    total_ideal: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Create from dictionary."""
        return cls(**data)

    def save(self, output_dir: Path) -> None:
        """Save session state to file."""
        self.last_activity = datetime.now().isoformat()
        state_file = output_dir / f"session_{self.session_id}.json"
        with open(state_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, session_file: Path) -> "SessionState":
        """Load session state from file."""
        with open(session_file, 'r') as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Interactive Ingest Session
# =============================================================================

class InteractiveIngestSession:
    """
    Interactive session for human-in-the-loop content ingestion.

    Provides a terminal interface for reviewing, editing, and
    collecting ideal responses for memory candidates.

    Attributes:
        pipeline: The ingestion pipeline
        output_dir: Directory for output files
        state: Current session state

    Example:
        async with InteractiveIngestSession(pipeline, output_dir) as session:
            await session.process_file_interactive(Path("memoir.txt"))
    """

    def __init__(
        self,
        pipeline: IngestPipeline,
        output_dir: Path,
        resume_session: Optional[str] = None,
    ):
        """
        Initialize interactive session.

        Args:
            pipeline: Configured IngestPipeline instance
            output_dir: Directory for output files
            resume_session: Optional session ID to resume
        """
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or resume session state
        if resume_session:
            session_file = self.output_dir / f"session_{resume_session}.json"
            if session_file.exists():
                self.state = SessionState.load(session_file)
                print(colorize(f"Resuming session: {resume_session}", Colors.GREEN))
            else:
                raise FileNotFoundError(f"Session file not found: {session_file}")
        else:
            self.state = SessionState(
                session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                source_path=""
            )

        # Output file paths
        self.processed_file = self.output_dir / "processed_memories.json"
        self.skipped_file = self.output_dir / "skipped_memories.json"
        self.ideal_file = self.output_dir / "ideal_responses.json"

        # Load existing data if resuming
        self.processed_memories: List[Dict[str, Any]] = self._load_json(self.processed_file)
        self.skipped_memories: List[Dict[str, Any]] = self._load_json(self.skipped_file)
        self.ideal_responses: List[Dict[str, Any]] = self._load_json(self.ideal_file)

        # Check if terminal supports colors
        if not sys.stdout.isatty():
            Colors.disable()

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSON file or return empty list."""
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _save_json(self, path: Path, data: List[Dict[str, Any]]) -> None:
        """Save data to JSON file."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _save_all(self) -> None:
        """Save all output files and session state."""
        self._save_json(self.processed_file, self.processed_memories)
        self._save_json(self.skipped_file, self.skipped_memories)
        self._save_json(self.ideal_file, self.ideal_responses)
        self.state.save(self.output_dir)

    async def __aenter__(self) -> "InteractiveIngestSession":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context and save state."""
        self._save_all()
        print(colorize("\nSession saved.", Colors.GREEN))

    # =========================================================================
    # Main Processing Methods
    # =========================================================================

    async def process_file_interactive(
        self,
        file_path: Path,
        start_index: int = 0
    ) -> Tuple[int, int, int, int]:
        """
        Process a single file with interactive review.

        Args:
            file_path: Path to the file to process
            start_index: Memory index to start from (for resume)

        Returns:
            Tuple of (accepted, edited, skipped, ideal_count)
        """
        file_path = Path(file_path).expanduser().resolve()
        self.state.source_path = str(file_path)

        print(colorize(f"\nProcessing: {file_path.name}", Colors.BOLD))
        print(colorize("=" * 50, Colors.DIM))

        # Process file through pipeline (dry run to get candidates)
        result = await self.pipeline.ingest_file(file_path)

        if not result.success:
            print(colorize(f"Error: {result.error}", Colors.RED))
            return (0, 0, 0, 0)

        candidates = result.memory_candidates
        total = len(candidates)

        if total == 0:
            print(colorize("No memory candidates generated.", Colors.YELLOW))
            return (0, 0, 0, 0)

        print(colorize(f"Generated {total} memory candidates.\n", Colors.CYAN))

        accepted = 0
        edited = 0
        skipped = 0
        ideal_count = 0

        for i, candidate in enumerate(candidates[start_index:], start=start_index):
            self.state.current_memory_index = i

            # Display candidate
            action = self._interactive_review(candidate, i + 1, total, file_path.name)

            if action == 'accept':
                self._accept_candidate(candidate)
                accepted += 1
                self.state.total_accepted += 1

            elif action == 'edit':
                edited_candidate = self._edit_candidate(candidate)
                self._accept_candidate(edited_candidate)
                edited += 1
                self.state.total_edited += 1

            elif action == 'skip':
                reason = input("  Reason for skipping (optional): ").strip()
                self._skip_candidate(candidate, reason or "User skipped")
                skipped += 1
                self.state.total_skipped += 1

            elif action == 'ideal':
                ideal = self._collect_ideal_response(candidate)
                if ideal:
                    self._accept_candidate(candidate)
                    accepted += 1
                    ideal_count += 1
                    self.state.total_accepted += 1
                    self.state.total_ideal += 1

            elif action == 'accept_all':
                # Accept all remaining
                for remaining in candidates[i:]:
                    self._accept_candidate(remaining)
                    accepted += 1
                    self.state.total_accepted += 1
                print(colorize(f"\nAccepted {len(candidates) - i} remaining memories.", Colors.GREEN))
                break

            elif action == 'quit':
                print(colorize("\nSession paused. Use --resume to continue.", Colors.YELLOW))
                break

            # Auto-save every 5 memories
            if (i + 1) % 5 == 0:
                self._save_all()

        return (accepted, edited, skipped, ideal_count)

    async def process_directory_interactive(
        self,
        directory: Path,
        extensions: Optional[List[str]] = None
    ) -> None:
        """
        Process all files in a directory interactively.

        Args:
            directory: Directory to process
            extensions: Optional list of file extensions to filter
        """
        directory = Path(directory).expanduser().resolve()

        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Collect files
        files = sorted([
            f for f in directory.rglob("*")
            if f.is_file() and not f.name.startswith('.')
            and (not extensions or f.suffix.lower() in extensions)
            and self.pipeline.processor_registry.can_process(f.suffix.lower())
        ])

        total_files = len(files)
        if total_files == 0:
            print(colorize("No processable files found.", Colors.YELLOW))
            return

        print(colorize(f"\nFound {total_files} files to process.\n", Colors.CYAN))

        start_file = self.state.current_file_index

        for i, file_path in enumerate(files[start_file:], start=start_file):
            self.state.current_file_index = i

            # Skip already processed files
            if str(file_path) in self.state.processed_files:
                continue

            print(colorize(f"\n[File {i + 1}/{total_files}] ", Colors.BOLD), end="")

            accepted, edited, skipped, ideal = await self.process_file_interactive(file_path)

            self.state.processed_files.append(str(file_path))
            self._save_all()

            # Summary for this file
            print(colorize(
                f"\n  Summary: {accepted} accepted, {edited} edited, "
                f"{skipped} skipped, {ideal} ideal responses",
                Colors.DIM
            ))

            # Prompt to continue
            if i < total_files - 1:
                choice = input("\nContinue to next file? [Y/n/q]: ").strip().lower()
                if choice == 'n':
                    print(colorize("Pausing. Use --resume to continue.", Colors.YELLOW))
                    break
                elif choice == 'q':
                    print(colorize("Quitting session.", Colors.YELLOW))
                    break

        self._print_final_summary()

    # =========================================================================
    # Display Methods
    # =========================================================================

    def _display_candidate(self, candidate: MemoryCandidate) -> List[str]:
        """
        Format candidate for display.

        Args:
            candidate: MemoryCandidate to display

        Returns:
            List of formatted lines
        """
        lines = []

        # Content preview
        lines.append(colorize("Content:", Colors.BOLD))
        content_preview = candidate.content[:300]
        if len(candidate.content) > 300:
            content_preview += "..."

        # Wrap content
        wrapped = textwrap.wrap(content_preview, width=58)
        for line in wrapped[:6]:  # Max 6 lines
            lines.append(f"  {colorize(line, Colors.DIM)}")
        if len(wrapped) > 6:
            lines.append(f"  {colorize('...', Colors.DIM)}")

        lines.append("")

        # Classification
        lines.append(colorize("Classification:", Colors.BOLD))

        type_color = Colors.GREEN if candidate.memory_type == MemoryType.EPISODIC else Colors.BLUE
        lines.append(f"  Type: {colorize(candidate.memory_type.value.upper(), type_color)} "
                    f"(confidence: {candidate.confidence:.2f})")

        lines.append(f"  Category: {colorize(candidate.category, Colors.CYAN)}")

        if candidate.themes:
            themes_str = ", ".join(candidate.themes[:5])
            if len(candidate.themes) > 5:
                themes_str += f" (+{len(candidate.themes) - 5} more)"
            lines.append(f"  Themes: {themes_str}")

        if candidate.participants:
            participants_str = ", ".join(candidate.participants[:5])
            lines.append(f"  Participants: {participants_str}")

        lines.append(f"  Emotional Valence: {candidate.emotional_valence}")

        importance_color = Colors.GREEN if candidate.importance > 0.7 else (
            Colors.YELLOW if candidate.importance > 0.4 else Colors.DIM
        )
        lines.append(f"  Importance: {colorize(f'{candidate.importance:.2f}', importance_color)}")

        if candidate.timestamp:
            lines.append(f"  Timestamp: {candidate.timestamp.isoformat()}")

        return lines

    def _interactive_review(
        self,
        candidate: MemoryCandidate,
        current: int,
        total: int,
        source_file: str
    ) -> str:
        """
        Display candidate and get user action.

        Args:
            candidate: The candidate to review
            current: Current memory index
            total: Total memory count
            source_file: Source file name

        Returns:
            Action string: 'accept', 'edit', 'skip', 'ideal', 'accept_all', 'quit'
        """
        # Header
        header = f"Memory {current} of {total}"
        if source_file:
            header += f" ({source_file})"

        lines = [colorize(header, Colors.BOLD)]
        lines.append("─" * 60)
        lines.extend(self._display_candidate(candidate))
        lines.append("─" * 60)

        # Actions
        actions = [
            f"{colorize('[A]', Colors.GREEN)}ccept",
            f"{colorize('[E]', Colors.YELLOW)}dit",
            f"{colorize('[S]', Colors.RED)}kip",
            f"{colorize('[I]', Colors.CYAN)}deal response",
            f"{colorize('[B]', Colors.DIM)}atch accept all",
            f"{colorize('[Q]', Colors.DIM)}uit"
        ]
        lines.append("  ".join(actions))

        print_box(lines)

        while True:
            choice = input("\nAction: ").strip().lower()

            if choice in ('a', 'accept'):
                return 'accept'
            elif choice in ('e', 'edit'):
                return 'edit'
            elif choice in ('s', 'skip'):
                return 'skip'
            elif choice in ('i', 'ideal'):
                return 'ideal'
            elif choice in ('b', 'batch', 'all'):
                confirm = input("Accept all remaining? [y/N]: ").strip().lower()
                if confirm == 'y':
                    return 'accept_all'
            elif choice in ('q', 'quit'):
                confirm = input("Quit and save session? [y/N]: ").strip().lower()
                if confirm == 'y':
                    return 'quit'
            else:
                print(colorize("Invalid choice. Use A/E/S/I/B/Q.", Colors.RED))

    # =========================================================================
    # Edit Methods
    # =========================================================================

    def _edit_candidate(self, candidate: MemoryCandidate) -> MemoryCandidate:
        """
        Edit a memory candidate interactively.

        Args:
            candidate: Original candidate

        Returns:
            Modified MemoryCandidate
        """
        print(colorize("\nEditing memory candidate:", Colors.BOLD))
        print(colorize("(Press Enter to keep current value)\n", Colors.DIM))

        # Memory type
        current_type = candidate.memory_type.value
        print(f"Memory Type [{current_type}]: ", end="")
        new_type = input().strip().lower()
        if new_type and new_type in ('episodic', 'semantic', 'skip'):
            candidate.memory_type = MemoryType(new_type)

        # Category
        print(f"Category [{candidate.category}]: ", end="")
        new_category = input().strip()
        if new_category:
            candidate.category = new_category

        # Themes
        current_themes = ", ".join(candidate.themes)
        print(f"Themes [{current_themes}]: ", end="")
        new_themes = input().strip()
        if new_themes:
            candidate.themes = [t.strip() for t in new_themes.split(",")]

        # Participants
        current_participants = ", ".join(candidate.participants)
        print(f"Participants [{current_participants}]: ", end="")
        new_participants = input().strip()
        if new_participants:
            candidate.participants = [p.strip() for p in new_participants.split(",")]

        # Emotional valence
        print(f"Emotional Valence [{candidate.emotional_valence}]: ", end="")
        new_valence = input().strip()
        if new_valence:
            candidate.emotional_valence = new_valence

        # Importance
        print(f"Importance [{candidate.importance}]: ", end="")
        new_importance = input().strip()
        if new_importance:
            try:
                candidate.importance = float(new_importance)
            except ValueError:
                print(colorize("Invalid number, keeping original.", Colors.YELLOW))

        # Event type (for episodic)
        if candidate.memory_type == MemoryType.EPISODIC:
            current_event = candidate.event_type or "general"
            print(f"Event Type [{current_event}]: ", end="")
            new_event = input().strip()
            if new_event:
                candidate.event_type = new_event

        # Knowledge type (for semantic)
        if candidate.memory_type == MemoryType.SEMANTIC:
            current_knowledge = candidate.knowledge_type or "belief"
            print(f"Knowledge Type [{current_knowledge}]: ", end="")
            new_knowledge = input().strip()
            if new_knowledge:
                candidate.knowledge_type = new_knowledge

        print(colorize("\nCandidate updated.", Colors.GREEN))
        return candidate

    def _collect_ideal_response(self, candidate: MemoryCandidate) -> Optional[IdealResponse]:
        """
        Collect an ideal response for DPO training.

        Args:
            candidate: The memory candidate

        Returns:
            IdealResponse or None if cancelled
        """
        print(colorize("\nCollecting ideal response for DPO training:", Colors.BOLD))
        print(colorize("Write how an ideal UBIK persona should respond to queries", Colors.DIM))
        print(colorize("about this memory. This helps train the response model.\n", Colors.DIM))

        print(colorize("Memory content:", Colors.CYAN))
        print(textwrap.indent(candidate.content[:500], "  "))
        if len(candidate.content) > 500:
            print("  ...")
        print()

        print(colorize("Enter ideal response (empty line to finish, 'cancel' to abort):", Colors.YELLOW))

        lines = []
        while True:
            line = input()
            if line.lower() == 'cancel':
                print(colorize("Cancelled.", Colors.RED))
                return None
            if line == '':
                break
            lines.append(line)

        if not lines:
            print(colorize("No response entered.", Colors.YELLOW))
            return None

        ideal_text = "\n".join(lines)

        # Context
        print(colorize("\nOptional context (e.g., 'when asked about fishing memories'):", Colors.DIM))
        context = input("Context: ").strip()

        ideal = IdealResponse(
            memory_content=candidate.content,
            memory_type=candidate.memory_type.value,
            ideal_response=ideal_text,
            context=context or "general query",
            timestamp=datetime.now().isoformat(),
            source_file=candidate.source_file
        )

        self.ideal_responses.append(ideal.to_dict())
        print(colorize("Ideal response saved!", Colors.GREEN))

        return ideal

    # =========================================================================
    # Storage Methods
    # =========================================================================

    def _accept_candidate(self, candidate: MemoryCandidate) -> None:
        """Add accepted candidate to processed memories."""
        memory_dict = {
            "content": candidate.content,
            "memory_type": candidate.memory_type.value,
            "confidence": candidate.confidence,
            "category": candidate.category,
            "themes": candidate.themes,
            "participants": candidate.participants,
            "emotional_valence": candidate.emotional_valence,
            "importance": candidate.importance,
            "source_file": candidate.source_file,
            "timestamp": candidate.timestamp.isoformat() if candidate.timestamp else None,
            "event_type": candidate.event_type,
            "knowledge_type": candidate.knowledge_type,
            "stability": candidate.stability,
            "reviewed_at": datetime.now().isoformat(),
            "session_id": self.state.session_id,
        }
        self.processed_memories.append(memory_dict)

    def _skip_candidate(self, candidate: MemoryCandidate, reason: str) -> None:
        """Add skipped candidate to skipped memories."""
        skipped = SkippedMemory(
            content=candidate.content[:500],
            reason=reason,
            original_classification=candidate.memory_type.value,
            source_file=candidate.source_file
        )
        self.skipped_memories.append(skipped.to_dict())

    # =========================================================================
    # Summary Methods
    # =========================================================================

    def _print_final_summary(self) -> None:
        """Print final session summary."""
        print(colorize("\n" + "=" * 50, Colors.BOLD))
        print(colorize("SESSION SUMMARY", Colors.BOLD))
        print(colorize("=" * 50, Colors.BOLD))

        print(f"Session ID: {self.state.session_id}")
        print(f"Started: {self.state.started_at}")
        print(f"Last Activity: {self.state.last_activity}")
        print()

        print(colorize("Results:", Colors.CYAN))
        print(f"  Accepted: {colorize(str(self.state.total_accepted), Colors.GREEN)}")
        print(f"  Edited: {colorize(str(self.state.total_edited), Colors.YELLOW)}")
        print(f"  Skipped: {colorize(str(self.state.total_skipped), Colors.RED)}")
        print(f"  Ideal Responses: {colorize(str(self.state.total_ideal), Colors.CYAN)}")
        print()

        print(colorize("Output Files:", Colors.CYAN))
        print(f"  {self.processed_file}")
        print(f"  {self.skipped_file}")
        print(f"  {self.ideal_file}")
        print()

        if self.state.total_ideal > 0:
            print(colorize(
                f"Collected {self.state.total_ideal} ideal responses for DPO training!",
                Colors.GREEN
            ))


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Main entry point for interactive ingestion."""
    parser = argparse.ArgumentParser(
        description="Interactive UBIK content ingestion with human review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python interactive_ingest.py ~/documents/memoir.txt
              python interactive_ingest.py ~/letters/ --output-dir ~/ubik/reviewed
              python interactive_ingest.py ~/content --resume session_20241215_143022

            Actions during review:
              [A]ccept    - Accept the classification as-is
              [E]dit      - Modify the classification
              [S]kip      - Skip this memory
              [I]deal     - Write an ideal response for DPO training
              [B]atch     - Accept all remaining memories
              [Q]uit      - Save and exit (can resume later)
        """)
    )

    parser.add_argument(
        "source",
        type=str,
        help="File or directory to process"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reviewed",
        help="Output directory for processed data (default: ./reviewed)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Session ID to resume"
    )

    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model for audio (default: base)"
    )

    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        help="File extensions to process (e.g., .txt .md .pdf)"
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    # Disable colors if requested
    if args.no_color:
        Colors.disable()

    # Resolve paths
    source_path = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not source_path.exists():
        print(colorize(f"Error: Source not found: {source_path}", Colors.RED))
        sys.exit(1)

    # Create pipeline (dry-run mode for interactive review)
    config = PipelineConfig(
        storage_mode=False,  # Don't store during interactive review
        whisper_model=args.whisper_model,
        chunk_config=ChunkConfig(
            min_chunk_size=100,
            target_chunk_size=500,
            max_chunk_size=1500,
            overlap_size=50,
        ),
    )

    pipeline = IngestPipeline(config=config)

    # Print banner
    print(colorize("\n" + "=" * 60, Colors.BOLD))
    print(colorize("  UBIK Interactive Ingestion", Colors.BOLD))
    print(colorize("  Human-in-the-loop review for digital legacy content", Colors.DIM))
    print(colorize("=" * 60 + "\n", Colors.BOLD))

    # Run interactive session
    async with InteractiveIngestSession(
        pipeline=pipeline,
        output_dir=output_dir,
        resume_session=args.resume
    ) as session:
        if source_path.is_file():
            await session.process_file_interactive(source_path)
        else:
            extensions = args.extensions
            if extensions:
                extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
            await session.process_directory_interactive(source_path, extensions)


if __name__ == "__main__":
    asyncio.run(main())
