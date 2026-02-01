"""
UBIK Ingestion System - Smart Content Chunker

Intelligent text chunking that respects content boundaries.
Splits content for embedding while preserving semantic coherence.

Chunking Strategy (hierarchical):
    1. If content fits in max_chunk_size → single chunk
    2. Split by section headers (# Heading)
    3. Split by paragraphs (double newline)
    4. Split by sentences (. ? !)
    5. Hard split at max_chunk_size (last resort)

Features:
    - Respects paragraph boundaries
    - Preserves list item groupings
    - Maintains quote integrity
    - Applies overlap for context continuity
    - Merges undersized chunks

Usage:
    from ingest.chunkers import SmartChunker, ChunkConfig

    chunker = SmartChunker(ChunkConfig(target_chunk_size=500))
    chunks = chunker.chunk(processed_content)

Version: 0.1.0
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .models import ProcessedContent

__all__ = [
    'ChunkConfig',
    'Chunk',
    'SmartChunker',
]


# Regex patterns for content boundaries
SECTION_PATTERN = re.compile(r'\n(#{1,3}\s+.+)\n')
PARAGRAPH_PATTERN = re.compile(r'\n\s*\n')
SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
LIST_ITEM_PATTERN = re.compile(r'^\s*[-*•]\s+', re.MULTILINE)
NUMBERED_LIST_PATTERN = re.compile(r'^\s*\d+[.)]\s+', re.MULTILINE)
QUOTE_PATTERN = re.compile(r'^>\s+', re.MULTILINE)
BLOCKQUOTE_PATTERN = re.compile(r'(^>\s+.+$\n?)+', re.MULTILINE)


@dataclass
class ChunkConfig:
    """
    Configuration for content chunking.

    Attributes:
        min_chunk_size: Minimum characters per chunk (avoid tiny fragments)
        target_chunk_size: Ideal chunk size for embeddings
        max_chunk_size: Maximum characters before forced split
        overlap_size: Characters to overlap between chunks for context
        respect_paragraphs: Keep paragraphs together when possible
        respect_sentences: Avoid splitting mid-sentence
        preserve_lists: Keep list items grouped together
        preserve_quotes: Keep block quotes intact

    Example:
        config = ChunkConfig(
            target_chunk_size=500,
            max_chunk_size=1000,
            overlap_size=100
        )
    """
    min_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_MIN_SIZE", "100"))
    )
    target_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_TARGET_SIZE", "500"))
    )
    max_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_MAX_SIZE", "1500"))
    )
    overlap_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP_SIZE", "50"))
    )
    respect_paragraphs: bool = True
    respect_sentences: bool = True
    preserve_lists: bool = True
    preserve_quotes: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        if self.min_chunk_size >= self.target_chunk_size:
            raise ValueError("min_chunk_size must be less than target_chunk_size")
        if self.target_chunk_size >= self.max_chunk_size:
            raise ValueError("target_chunk_size must be less than max_chunk_size")
        if self.overlap_size >= self.min_chunk_size:
            raise ValueError("overlap_size must be less than min_chunk_size")


@dataclass
class Chunk:
    """
    A single content chunk with position metadata.

    Attributes:
        text: The chunk text content
        index: Position in the chunk sequence (0-indexed)
        start_char: Starting character position in source
        end_char: Ending character position in source
        chunk_type: How this chunk was created
            - "single": Entire content fit in one chunk
            - "section": Split by section header
            - "paragraph": Split by paragraph boundary
            - "sentence": Split by sentence boundary
            - "hard": Forced split at max size
            - "merged": Result of merging small chunks

    Example:
        chunk = Chunk(
            text="This is the chunk content...",
            index=0,
            start_char=0,
            end_char=500,
            chunk_type="paragraph"
        )
        print(f"Chunk {chunk.index}: {chunk.word_count} words")
    """
    text: str
    index: int
    start_char: int
    end_char: int
    chunk_type: str

    @property
    def char_count(self) -> int:
        """Number of characters in chunk."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Number of words in chunk."""
        return len(self.text.split())

    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"Chunk(index={self.index}, chars={self.char_count}, "
            f"type={self.chunk_type}, text={preview!r})"
        )


class SmartChunker:
    """
    Intelligent content chunker that respects semantic boundaries.

    Uses a hierarchical strategy to split content:
    1. Sections (headers)
    2. Paragraphs
    3. Sentences
    4. Hard split (last resort)

    Attributes:
        config: Chunking configuration

    Example:
        chunker = SmartChunker()
        chunks = chunker.chunk(processed_content)

        for chunk in chunks:
            print(f"[{chunk.chunk_type}] {chunk.word_count} words")
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        """Initialize chunker with configuration."""
        self.config = config or ChunkConfig()

    def chunk(self, content: ProcessedContent) -> List[Chunk]:
        """
        Chunk processed content into semantic units.

        Args:
            content: ProcessedContent with extracted text

        Returns:
            List of Chunk objects in order

        Raises:
            ValueError: If content text is empty
        """
        text = content.text.strip()

        if not text:
            raise ValueError("Cannot chunk empty content")

        # Single chunk if content is small enough
        if len(text) <= self.config.max_chunk_size:
            return [Chunk(
                text=text,
                index=0,
                start_char=0,
                end_char=len(text),
                chunk_type="single"
            )]

        # Hierarchical splitting
        raw_chunks = self._split_hierarchically(text)

        # Merge small chunks
        merged_chunks = self._merge_small_chunks(raw_chunks)

        # Build final chunk objects with proper indices
        final_chunks = []
        for i, (chunk_text, chunk_type, start, end) in enumerate(merged_chunks):
            final_chunks.append(Chunk(
                text=chunk_text,
                index=i,
                start_char=start,
                end_char=end,
                chunk_type=chunk_type
            ))

        return final_chunks

    def _split_hierarchically(
        self,
        text: str
    ) -> List[Tuple[str, str, int, int]]:
        """
        Split text using hierarchical strategy.

        Returns list of (text, chunk_type, start_char, end_char) tuples.
        """
        # Try section-based splitting first
        sections = self._split_by_sections(text)

        if len(sections) > 1:
            # Process each section
            all_chunks = []
            for section_text, start, end in sections:
                section_chunks = self._split_section(section_text, start)
                all_chunks.extend(section_chunks)
            return all_chunks

        # No sections found, split the whole text
        return self._split_section(text, 0)

    def _split_by_sections(
        self,
        text: str
    ) -> List[Tuple[str, int, int]]:
        """
        Split text by section headers (Markdown headings).

        Returns list of (section_text, start_char, end_char) tuples.
        """
        # Find all section headers
        matches = list(SECTION_PATTERN.finditer(text))

        if not matches:
            return [(text, 0, len(text))]

        sections = []

        # Content before first header
        if matches[0].start() > 0:
            pre_content = text[:matches[0].start()].strip()
            if pre_content:
                sections.append((pre_content, 0, matches[0].start()))

        # Each section from header to next header
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append((section_text, start, end))

        return sections

    def _split_section(
        self,
        text: str,
        base_offset: int
    ) -> List[Tuple[str, str, int, int]]:
        """
        Split a section into chunks by paragraphs, then sentences.

        Args:
            text: Section text to split
            base_offset: Character offset in original document

        Returns:
            List of (chunk_text, chunk_type, start, end) tuples
        """
        # If section fits, return as-is
        if len(text) <= self.config.max_chunk_size:
            return [(text, "section", base_offset, base_offset + len(text))]

        chunks = []
        current_chunk = ""
        current_start = base_offset
        chunk_type = "paragraph"

        # Split by paragraphs
        paragraphs = PARAGRAPH_PATTERN.split(text)
        char_pos = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Handle special content (lists, quotes)
            if self.config.preserve_lists and self._is_list(para):
                para_chunks = self._split_list(para, base_offset + char_pos)
            elif self.config.preserve_quotes and self._is_quote(para):
                para_chunks = [(para, "quote", base_offset + char_pos,
                               base_offset + char_pos + len(para))]
            else:
                para_chunks = [(para, "paragraph", base_offset + char_pos,
                               base_offset + char_pos + len(para))]

            for para_text, ptype, pstart, pend in para_chunks:
                # Check if adding this paragraph exceeds max size
                potential_size = len(current_chunk) + len(para_text) + 2

                if potential_size <= self.config.target_chunk_size:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + para_text
                    else:
                        current_chunk = para_text
                        current_start = pstart
                    chunk_type = ptype

                elif potential_size <= self.config.max_chunk_size:
                    # Slightly over target but under max, still add
                    if current_chunk:
                        current_chunk += "\n\n" + para_text
                    else:
                        current_chunk = para_text
                        current_start = pstart
                    chunk_type = ptype

                else:
                    # Would exceed max, flush current and start new
                    if current_chunk:
                        chunks.append((
                            current_chunk,
                            chunk_type,
                            current_start,
                            current_start + len(current_chunk)
                        ))

                    # Check if paragraph itself is too large
                    if len(para_text) > self.config.max_chunk_size:
                        # Split paragraph by sentences
                        sub_chunks = self._split_by_sentences(
                            para_text, pstart
                        )
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                        current_start = pend
                    else:
                        current_chunk = para_text
                        current_start = pstart
                        chunk_type = ptype

            # Track position (account for paragraph separators)
            char_pos += len(para) + 2

        # Flush remaining
        if current_chunk:
            chunks.append((
                current_chunk,
                chunk_type,
                current_start,
                current_start + len(current_chunk)
            ))

        return chunks

    def _split_by_sentences(
        self,
        text: str,
        base_offset: int
    ) -> List[Tuple[str, str, int, int]]:
        """
        Split text by sentence boundaries.

        Used when paragraphs are too large.
        """
        if len(text) <= self.config.max_chunk_size:
            return [(text, "sentence", base_offset, base_offset + len(text))]

        if not self.config.respect_sentences:
            return self._hard_split(text, base_offset)

        sentences = SENTENCE_PATTERN.split(text)
        chunks = []
        current_chunk = ""
        current_start = base_offset
        char_pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            potential_size = len(current_chunk) + len(sentence) + 1

            if potential_size <= self.config.target_chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    current_start = base_offset + char_pos

            elif potential_size <= self.config.max_chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    current_start = base_offset + char_pos

            else:
                # Flush current chunk
                if current_chunk:
                    chunks.append((
                        current_chunk,
                        "sentence",
                        current_start,
                        current_start + len(current_chunk)
                    ))

                # Check if sentence itself is too large
                if len(sentence) > self.config.max_chunk_size:
                    sub_chunks = self._hard_split(
                        sentence, base_offset + char_pos
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                    current_start = base_offset + char_pos + len(sentence)
                else:
                    current_chunk = sentence
                    current_start = base_offset + char_pos

            char_pos += len(sentence) + 1

        # Flush remaining
        if current_chunk:
            chunks.append((
                current_chunk,
                "sentence",
                current_start,
                current_start + len(current_chunk)
            ))

        return chunks

    def _split_list(
        self,
        text: str,
        base_offset: int
    ) -> List[Tuple[str, str, int, int]]:
        """
        Split a list block, keeping related items together.

        Attempts to keep list items grouped up to max_chunk_size.
        """
        if len(text) <= self.config.max_chunk_size:
            return [(text, "list", base_offset, base_offset + len(text))]

        # Split by list item markers
        lines = text.split('\n')
        chunks = []
        current_items = []
        current_start = base_offset
        current_size = 0
        char_pos = 0

        for line in lines:
            line_len = len(line) + 1  # +1 for newline

            if current_size + line_len <= self.config.max_chunk_size:
                current_items.append(line)
                current_size += line_len
            else:
                # Flush current items
                if current_items:
                    chunk_text = '\n'.join(current_items)
                    chunks.append((
                        chunk_text,
                        "list",
                        current_start,
                        current_start + len(chunk_text)
                    ))
                current_items = [line]
                current_start = base_offset + char_pos
                current_size = line_len

            char_pos += line_len

        # Flush remaining
        if current_items:
            chunk_text = '\n'.join(current_items)
            chunks.append((
                chunk_text,
                "list",
                current_start,
                current_start + len(chunk_text)
            ))

        return chunks

    def _hard_split(
        self,
        text: str,
        base_offset: int
    ) -> List[Tuple[str, str, int, int]]:
        """
        Force split text at max_chunk_size with overlap.

        Last resort when no semantic boundaries found.
        Attempts to split at word boundaries.
        """
        chunks = []
        remaining = text
        char_pos = 0

        while remaining:
            if len(remaining) <= self.config.max_chunk_size:
                chunks.append((
                    remaining,
                    "hard",
                    base_offset + char_pos,
                    base_offset + char_pos + len(remaining)
                ))
                break

            # Find a word boundary near max_chunk_size
            split_point = self.config.max_chunk_size

            # Look backwards for a space
            while split_point > self.config.min_chunk_size:
                if remaining[split_point] in ' \n\t':
                    break
                split_point -= 1

            # If no good boundary found, force split
            if split_point <= self.config.min_chunk_size:
                split_point = self.config.max_chunk_size

            chunk_text = remaining[:split_point].strip()
            chunks.append((
                chunk_text,
                "hard",
                base_offset + char_pos,
                base_offset + char_pos + len(chunk_text)
            ))

            # Apply overlap for next chunk
            overlap_start = max(0, split_point - self.config.overlap_size)
            # Find word boundary for overlap
            while overlap_start < split_point and remaining[overlap_start] not in ' \n\t':
                overlap_start += 1

            remaining = remaining[overlap_start:].strip()
            char_pos += overlap_start

        return chunks

    def _merge_small_chunks(
        self,
        chunks: List[Tuple[str, str, int, int]]
    ) -> List[Tuple[str, str, int, int]]:
        """
        Merge consecutive small chunks that are under min_chunk_size.

        Preserves chunk boundaries when merged size would exceed target.
        """
        if not chunks:
            return chunks

        merged = []
        current_text = ""
        current_type = ""
        current_start = 0
        current_end = 0

        for text, chunk_type, start, end in chunks:
            if not current_text:
                current_text = text
                current_type = chunk_type
                current_start = start
                current_end = end
                continue

            # Check if current chunk is small and can be merged
            if len(current_text) < self.config.min_chunk_size:
                potential_size = len(current_text) + len(text) + 2

                if potential_size <= self.config.target_chunk_size:
                    # Merge chunks
                    current_text = current_text + "\n\n" + text
                    current_type = "merged"
                    current_end = end
                    continue

            # Can't merge, flush current
            merged.append((current_text, current_type, current_start, current_end))
            current_text = text
            current_type = chunk_type
            current_start = start
            current_end = end

        # Flush final chunk
        if current_text:
            merged.append((current_text, current_type, current_start, current_end))

        # Handle case where last chunk is too small
        if len(merged) >= 2:
            last_text, last_type, last_start, last_end = merged[-1]
            if len(last_text) < self.config.min_chunk_size:
                prev_text, prev_type, prev_start, prev_end = merged[-2]
                potential = len(prev_text) + len(last_text) + 2

                if potential <= self.config.max_chunk_size:
                    merged_text = prev_text + "\n\n" + last_text
                    merged[-2] = (merged_text, "merged", prev_start, last_end)
                    merged.pop()

        return merged

    def _is_list(self, text: str) -> bool:
        """Check if text is a list block."""
        lines = text.strip().split('\n')
        if not lines:
            return False

        list_lines = 0
        for line in lines:
            if LIST_ITEM_PATTERN.match(line) or NUMBERED_LIST_PATTERN.match(line):
                list_lines += 1

        # Consider it a list if majority of lines are list items
        return list_lines > len(lines) / 2

    def _is_quote(self, text: str) -> bool:
        """Check if text is a block quote."""
        return bool(BLOCKQUOTE_PATTERN.match(text.strip()))
