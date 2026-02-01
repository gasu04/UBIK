"""
UBIK Ingestion System - Format Processors

Content extraction processors for various file formats.
Each processor handles specific file types and extracts text content.

Supported Formats:
    - Text: .txt, .md, .markdown
    - Documents: .pdf, .docx, .doc
    - Audio: .mp3, .wav, .m4a, .ogg, .flac
    - Structured: .json

Architecture:
    BaseProcessor (abstract)
    ├── TextProcessor
    ├── PDFProcessor
    ├── DOCXProcessor
    ├── AudioProcessor
    └── JSONProcessor

Usage:
    from ingest.processors import ProcessorRegistry

    registry = ProcessorRegistry()
    result = await registry.process(ingest_item, file_path)

Version: 0.1.0
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .models import ContentType, IngestItem, ProcessedContent

__all__ = [
    'BaseProcessor',
    'TextProcessor',
    'PDFProcessor',
    'DOCXProcessor',
    'AudioProcessor',
    'JSONProcessor',
    'ProcessorRegistry',
    'ProcessorConfig',
]

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """
    Configuration for processors.

    Attributes:
        whisper_model: Whisper model size (tiny, base, small, medium, large)
        whisper_device: Device for Whisper (cuda, cpu, auto)
        ocr_enabled: Whether to use OCR fallback for PDFs
        ocr_language: Tesseract language code
        max_workers: Thread pool size for blocking operations
        encoding_fallbacks: Encodings to try for text files
    """
    whisper_model: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL", "base")
    )
    whisper_device: str = field(
        default_factory=lambda: os.getenv("WHISPER_DEVICE", "auto")
    )
    ocr_enabled: bool = field(
        default_factory=lambda: os.getenv("OCR_ENABLED", "true").lower() == "true"
    )
    ocr_language: str = field(
        default_factory=lambda: os.getenv("OCR_LANGUAGE", "eng")
    )
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("PROCESSOR_MAX_WORKERS", "4"))
    )
    encoding_fallbacks: List[str] = field(
        default_factory=lambda: ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    )


class BaseProcessor(ABC):
    """
    Abstract base class for format processors.

    Subclasses must implement:
        - SUPPORTED_EXTENSIONS: File extensions this processor handles
        - CONTENT_TYPE: The ContentType enum value
        - process(): Async method to extract content

    Attributes:
        config: Processor configuration
        _executor: Thread pool for blocking operations

    Example:
        class MyProcessor(BaseProcessor):
            SUPPORTED_EXTENSIONS = [".xyz"]
            CONTENT_TYPE = ContentType.TEXT

            async def process(self, item, file_path):
                # Extract content
                return self._create_result(item, text, start_time)
    """

    SUPPORTED_EXTENSIONS: List[str] = []
    CONTENT_TYPE: ContentType = ContentType.UNKNOWN

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize processor with configuration."""
        self.config = config or ProcessorConfig()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

    @abstractmethod
    async def process(
        self,
        item: IngestItem,
        file_path: Path
    ) -> ProcessedContent:
        """
        Process a file and extract content.

        Args:
            item: IngestItem with source metadata
            file_path: Path to the file to process

        Returns:
            ProcessedContent with extracted text and metadata

        Raises:
            ProcessingError: If processing fails
        """
        pass

    def _create_result(
        self,
        item: IngestItem,
        text: str,
        start_time: float,
        title: Optional[str] = None,
        page_count: Optional[int] = None,
        audio_duration: Optional[float] = None,
        transcription_confidence: Optional[float] = None,
        language: str = "en",
        extracted_metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedContent:
        """
        Create a ProcessedContent result.

        Args:
            item: Source IngestItem
            text: Extracted text content
            start_time: Processing start timestamp
            title: Detected title
            page_count: Number of pages (documents)
            audio_duration: Duration in seconds (audio)
            transcription_confidence: Confidence score (audio)
            language: Detected language code
            extracted_metadata: Additional metadata

        Returns:
            ProcessedContent with computed fields
        """
        processing_time = (time.time() - start_time) * 1000
        word_count = len(text.split()) if text else 0

        return ProcessedContent(
            source_item=item,
            text=text,
            title=title,
            processor_used=self.__class__.__name__,
            processing_time_ms=processing_time,
            page_count=page_count,
            word_count=word_count,
            language=language,
            audio_duration_seconds=audio_duration,
            transcription_confidence=transcription_confidence,
            extracted_metadata=extracted_metadata or {},
        )

    async def _run_in_executor(self, func, *args):
        """Run a blocking function in the thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, func, *args)


class TextProcessor(BaseProcessor):
    """
    Processor for plain text and Markdown files.

    Handles encoding detection and title extraction.

    Supported: .txt, .md, .markdown

    Features:
        - Multiple encoding fallbacks
        - Markdown heading detection for title
        - First line title extraction
    """

    SUPPORTED_EXTENSIONS = [".txt", ".md", ".markdown"]
    CONTENT_TYPE = ContentType.TEXT

    async def process(
        self,
        item: IngestItem,
        file_path: Path
    ) -> ProcessedContent:
        """
        Process a text file.

        Tries multiple encodings until successful.
        Extracts title from Markdown heading or first line.
        """
        start_time = time.time()

        text = await self._run_in_executor(self._read_with_fallback, file_path)
        title = self._extract_title(text, file_path)
        language = self._detect_language(text)

        return self._create_result(
            item=item,
            text=text,
            start_time=start_time,
            title=title,
            language=language,
            extracted_metadata={
                "encoding_used": getattr(self, "_last_encoding", "utf-8"),
                "is_markdown": file_path.suffix.lower() in [".md", ".markdown"],
            },
        )

    def _read_with_fallback(self, file_path: Path) -> str:
        """
        Read file with encoding fallback chain.

        Tries each encoding in config.encoding_fallbacks until one works.
        """
        last_error = None

        for encoding in self.config.encoding_fallbacks:
            try:
                text = file_path.read_text(encoding=encoding)
                self._last_encoding = encoding
                logger.debug(f"Read {file_path} with encoding {encoding}")
                return text
            except UnicodeDecodeError as e:
                last_error = e
                continue

        raise ValueError(
            f"Could not decode {file_path} with any encoding: {last_error}"
        )

    def _extract_title(self, text: str, file_path: Path) -> str:
        """
        Extract title from content or filename.

        Priority:
        1. Markdown # heading
        2. First non-empty line
        3. Filename without extension
        """
        if not text.strip():
            return file_path.stem

        lines = text.strip().split("\n")

        # Check for Markdown heading
        for line in lines[:5]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()

        # Use first non-empty line
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # Truncate if too long
                return line[:100] + "..." if len(line) > 100 else line

        return file_path.stem

    def _detect_language(self, text: str) -> str:
        """Simple language detection heuristic."""
        # Could integrate langdetect here, for now default to English
        return "en"


class PDFProcessor(BaseProcessor):
    """
    Processor for PDF documents.

    Uses pdfplumber for text extraction with OCR fallback.

    Supported: .pdf

    Features:
        - Native text extraction via pdfplumber
        - OCR fallback using pytesseract + pdf2image
        - Page count tracking
        - Metadata extraction
    """

    SUPPORTED_EXTENSIONS = [".pdf"]
    CONTENT_TYPE = ContentType.DOCUMENT

    async def process(
        self,
        item: IngestItem,
        file_path: Path
    ) -> ProcessedContent:
        """
        Process a PDF file.

        Attempts native text extraction first, falls back to OCR
        if insufficient text is found.
        """
        start_time = time.time()

        result = await self._run_in_executor(self._extract_pdf, file_path)

        return self._create_result(
            item=item,
            text=result["text"],
            start_time=start_time,
            title=result.get("title"),
            page_count=result.get("page_count"),
            extracted_metadata={
                "extraction_method": result.get("method", "pdfplumber"),
                "pdf_metadata": result.get("metadata", {}),
                "ocr_used": result.get("ocr_used", False),
            },
        )

    def _extract_pdf(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from PDF using pdfplumber.

        Falls back to OCR if native extraction yields insufficient text.
        """
        import pdfplumber

        result = {
            "text": "",
            "page_count": 0,
            "method": "pdfplumber",
            "ocr_used": False,
            "metadata": {},
        }

        try:
            with pdfplumber.open(file_path) as pdf:
                result["page_count"] = len(pdf.pages)
                result["metadata"] = pdf.metadata or {}

                # Extract title from metadata
                if pdf.metadata:
                    result["title"] = pdf.metadata.get("Title") or pdf.metadata.get("title")

                # Extract text from all pages
                pages_text = []
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    pages_text.append(page_text)

                result["text"] = "\n\n".join(pages_text)

        except Exception as e:
            logger.error(f"pdfplumber failed for {file_path}: {e}")
            result["text"] = ""

        # Check if we need OCR fallback
        word_count = len(result["text"].split())
        if word_count < 50 and self.config.ocr_enabled:
            logger.info(f"Insufficient text ({word_count} words), attempting OCR")
            ocr_text = self._extract_with_ocr(file_path)
            if ocr_text and len(ocr_text.split()) > word_count:
                result["text"] = ocr_text
                result["method"] = "ocr"
                result["ocr_used"] = True

        return result

    def _extract_with_ocr(self, file_path: Path) -> str:
        """
        Extract text using OCR (pytesseract + pdf2image).

        Returns empty string if OCR dependencies unavailable.
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError:
            logger.warning("OCR dependencies not available (pdf2image, pytesseract)")
            return ""

        try:
            images = convert_from_path(file_path)
            ocr_texts = []

            for i, image in enumerate(images):
                text = pytesseract.image_to_string(
                    image,
                    lang=self.config.ocr_language
                )
                ocr_texts.append(text)
                logger.debug(f"OCR page {i + 1}: {len(text)} chars")

            return "\n\n".join(ocr_texts)

        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}")
            return ""


class DOCXProcessor(BaseProcessor):
    """
    Processor for Microsoft Word documents.

    Supports .docx natively, .doc via LibreOffice conversion.

    Supported: .docx, .doc

    Features:
        - pandoc conversion (preferred)
        - python-docx fallback for .docx
        - LibreOffice conversion for .doc
    """

    SUPPORTED_EXTENSIONS = [".docx", ".doc"]
    CONTENT_TYPE = ContentType.DOCUMENT

    def __init__(self, config: Optional[ProcessorConfig] = None):
        super().__init__(config)
        self._pandoc_available: Optional[bool] = None
        self._libreoffice_available: Optional[bool] = None

    async def process(
        self,
        item: IngestItem,
        file_path: Path
    ) -> ProcessedContent:
        """
        Process a Word document.

        Strategy:
        1. For .doc files, convert to .docx first via LibreOffice
        2. Try pandoc for best formatting preservation
        3. Fall back to python-docx for .docx files
        """
        start_time = time.time()

        # Handle .doc conversion
        actual_path = file_path
        temp_docx = None

        if file_path.suffix.lower() == ".doc":
            temp_docx = await self._convert_doc_to_docx(file_path)
            if temp_docx:
                actual_path = temp_docx
            else:
                raise ValueError(
                    f"Cannot process .doc file without LibreOffice: {file_path}"
                )

        try:
            result = await self._run_in_executor(self._extract_docx, actual_path)
        finally:
            # Clean up temp file
            if temp_docx and temp_docx.exists():
                temp_docx.unlink()

        return self._create_result(
            item=item,
            text=result["text"],
            start_time=start_time,
            title=result.get("title"),
            extracted_metadata={
                "extraction_method": result.get("method", "unknown"),
                "converted_from_doc": temp_docx is not None,
            },
        )

    def _check_pandoc(self) -> bool:
        """Check if pandoc is available."""
        if self._pandoc_available is None:
            self._pandoc_available = shutil.which("pandoc") is not None
        return self._pandoc_available

    def _check_libreoffice(self) -> bool:
        """Check if LibreOffice is available."""
        if self._libreoffice_available is None:
            self._libreoffice_available = (
                shutil.which("libreoffice") is not None or
                shutil.which("soffice") is not None
            )
        return self._libreoffice_available

    async def _convert_doc_to_docx(self, doc_path: Path) -> Optional[Path]:
        """
        Convert .doc to .docx using LibreOffice.

        Returns path to temporary .docx file, or None if conversion fails.
        """
        if not self._check_libreoffice():
            logger.warning("LibreOffice not available for .doc conversion")
            return None

        try:
            temp_dir = tempfile.mkdtemp()
            cmd = [
                "libreoffice",
                "--headless",
                "--convert-to", "docx",
                "--outdir", temp_dir,
                str(doc_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()

            # Find converted file
            docx_path = Path(temp_dir) / f"{doc_path.stem}.docx"
            if docx_path.exists():
                return docx_path

            logger.error(f"LibreOffice conversion did not produce output: {doc_path}")
            return None

        except Exception as e:
            logger.error(f"LibreOffice conversion failed: {e}")
            return None

    def _extract_docx(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from .docx file.

        Tries pandoc first, then python-docx.
        """
        result = {"text": "", "method": "unknown"}

        # Try pandoc first
        if self._check_pandoc():
            try:
                proc = subprocess.run(
                    ["pandoc", "-f", "docx", "-t", "plain", str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if proc.returncode == 0:
                    result["text"] = proc.stdout
                    result["method"] = "pandoc"
                    return result
            except Exception as e:
                logger.warning(f"pandoc failed, trying python-docx: {e}")

        # Fall back to python-docx
        try:
            from docx import Document

            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs]
            result["text"] = "\n\n".join(paragraphs)
            result["method"] = "python-docx"

            # Try to get title from core properties
            if doc.core_properties.title:
                result["title"] = doc.core_properties.title

        except Exception as e:
            logger.error(f"python-docx failed: {e}")
            raise ValueError(f"Could not extract text from {file_path}: {e}")

        return result


class AudioProcessor(BaseProcessor):
    """
    Processor for audio files using OpenAI Whisper.

    Lazy-loads Whisper model on first use to save memory.

    Supported: .mp3, .wav, .m4a, .ogg, .flac

    Features:
        - Configurable model size (tiny to large)
        - GPU acceleration when available
        - Language detection
        - Confidence scores per segment
        - Non-blocking async transcription
    """

    SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]
    CONTENT_TYPE = ContentType.AUDIO

    def __init__(self, config: Optional[ProcessorConfig] = None):
        super().__init__(config)
        self._model = None
        self._model_lock = asyncio.Lock()

    async def _get_model(self):
        """
        Lazy-load Whisper model.

        Uses lock to prevent multiple simultaneous loads.
        """
        if self._model is None:
            async with self._model_lock:
                if self._model is None:
                    self._model = await self._run_in_executor(self._load_model)
        return self._model

    def _load_model(self):
        """Load Whisper model (blocking, run in executor)."""
        import whisper

        model_name = self.config.whisper_model
        device = self.config.whisper_device

        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading Whisper model '{model_name}' on {device}")
        model = whisper.load_model(model_name, device=device)
        logger.info(f"Whisper model loaded successfully")

        return model

    async def process(
        self,
        item: IngestItem,
        file_path: Path
    ) -> ProcessedContent:
        """
        Process an audio file with Whisper transcription.

        Runs transcription in thread pool to avoid blocking async loop.
        """
        start_time = time.time()

        model = await self._get_model()
        result = await self._run_in_executor(
            self._transcribe,
            model,
            file_path
        )

        return self._create_result(
            item=item,
            text=result["text"],
            start_time=start_time,
            language=result.get("language", "en"),
            audio_duration=result.get("duration"),
            transcription_confidence=result.get("confidence"),
            extracted_metadata={
                "segments": result.get("segments", []),
                "model_used": self.config.whisper_model,
                "detected_language": result.get("language"),
            },
        )

    def _transcribe(self, model, file_path: Path) -> Dict[str, Any]:
        """
        Run Whisper transcription (blocking, run in executor).

        Returns dict with text, language, duration, confidence, segments.
        """
        import whisper

        logger.info(f"Transcribing {file_path}")

        try:
            result = model.transcribe(
                str(file_path),
                verbose=False,
                task="transcribe"
            )
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise ValueError(f"Failed to transcribe {file_path}: {e}")

        # Calculate average confidence from segments
        segments = result.get("segments", [])
        if segments:
            # Whisper doesn't provide confidence directly, estimate from no_speech_prob
            confidences = [
                1.0 - seg.get("no_speech_prob", 0.0)
                for seg in segments
            ]
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_confidence = None

        # Calculate duration from segments
        duration = None
        if segments:
            duration = segments[-1].get("end", 0.0)

        # Simplify segments for storage
        simplified_segments = [
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", "").strip(),
            }
            for seg in segments
        ]

        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "en"),
            "duration": duration,
            "confidence": avg_confidence,
            "segments": simplified_segments,
        }


class JSONProcessor(BaseProcessor):
    """
    Processor for JSON files.

    Converts JSON structure to readable text while preserving
    structure in metadata.

    Supported: .json

    Features:
        - Pretty-printed text representation
        - Full structure preserved in metadata
        - Nested object flattening
    """

    SUPPORTED_EXTENSIONS = [".json"]
    CONTENT_TYPE = ContentType.STRUCTURED

    async def process(
        self,
        item: IngestItem,
        file_path: Path
    ) -> ProcessedContent:
        """
        Process a JSON file.

        Converts to readable text and stores original structure in metadata.
        """
        start_time = time.time()

        result = await self._run_in_executor(self._parse_json, file_path)

        return self._create_result(
            item=item,
            text=result["text"],
            start_time=start_time,
            title=result.get("title"),
            extracted_metadata={
                "json_structure": result.get("structure"),
                "root_type": result.get("root_type"),
                "key_count": result.get("key_count"),
            },
        )

    def _parse_json(self, file_path: Path) -> Dict[str, Any]:
        """Parse JSON and convert to text."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Determine structure type
        if isinstance(data, dict):
            root_type = "object"
            key_count = len(data)
        elif isinstance(data, list):
            root_type = "array"
            key_count = len(data)
        else:
            root_type = type(data).__name__
            key_count = 1

        # Convert to readable text
        text = self._json_to_text(data)

        # Try to extract title
        title = None
        if isinstance(data, dict):
            title = data.get("title") or data.get("name") or data.get("id")
            if title and not isinstance(title, str):
                title = str(title)

        return {
            "text": text,
            "structure": data,
            "root_type": root_type,
            "key_count": key_count,
            "title": title,
        }

    def _json_to_text(self, data: Any, indent: int = 0) -> str:
        """
        Convert JSON data to readable text format.

        Recursively processes nested structures.
        """
        prefix = "  " * indent
        lines = []

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._json_to_text(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")

        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{i}]:")
                    lines.append(self._json_to_text(item, indent + 1))
                else:
                    lines.append(f"{prefix}- {item}")

        else:
            lines.append(f"{prefix}{data}")

        return "\n".join(lines)


class ProcessorRegistry:
    """
    Registry for format processors.

    Maps file extensions to appropriate processors and provides
    unified processing interface.

    Attributes:
        config: Shared processor configuration
        processors: Mapping of extensions to processor instances

    Example:
        registry = ProcessorRegistry()

        if registry.can_process(".pdf"):
            result = await registry.process(item, path)
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize registry with all processors."""
        self.config = config or ProcessorConfig()
        self._processors: Dict[str, BaseProcessor] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all default processors."""
        processor_classes: List[Type[BaseProcessor]] = [
            TextProcessor,
            PDFProcessor,
            DOCXProcessor,
            AudioProcessor,
            JSONProcessor,
        ]

        for processor_class in processor_classes:
            instance = processor_class(self.config)
            for ext in processor_class.SUPPORTED_EXTENSIONS:
                self._processors[ext.lower()] = instance
                logger.debug(f"Registered {processor_class.__name__} for {ext}")

    def register(self, extension: str, processor: BaseProcessor) -> None:
        """
        Register a custom processor for an extension.

        Args:
            extension: File extension (with leading dot)
            processor: Processor instance to handle this extension
        """
        self._processors[extension.lower()] = processor

    def get_processor(self, extension: str) -> Optional[BaseProcessor]:
        """
        Get processor for a file extension.

        Args:
            extension: File extension (with or without leading dot)

        Returns:
            Processor instance or None if not supported
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return self._processors.get(ext)

    def can_process(self, extension: str) -> bool:
        """
        Check if an extension is supported.

        Args:
            extension: File extension to check

        Returns:
            True if a processor is registered for this extension
        """
        return self.get_processor(extension) is not None

    @property
    def supported_extensions(self) -> List[str]:
        """Get list of all supported extensions."""
        return sorted(self._processors.keys())

    async def process(
        self,
        item: IngestItem,
        file_path: Path
    ) -> ProcessedContent:
        """
        Process a file using the appropriate processor.

        Args:
            item: IngestItem with source metadata
            file_path: Path to the file to process

        Returns:
            ProcessedContent with extracted content

        Raises:
            ValueError: If no processor is registered for the file type
        """
        processor = self.get_processor(item.file_extension)

        if processor is None:
            raise ValueError(
                f"No processor registered for extension: {item.file_extension}"
            )

        logger.info(
            f"Processing {file_path.name} with {processor.__class__.__name__}"
        )

        return await processor.process(item, file_path)
