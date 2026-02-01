# UBIK Content Ingestion System

A comprehensive content ingestion pipeline for the UBIK digital legacy project. Processes personal documents, transcripts, and recordings into classified memory candidates for storage in the Hippocampal Node.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        UBIK Ingestion Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Sources    │    │  Processors  │    │   Chunkers   │    │Classifiers│ │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤    ├───────────┤ │
│  │ Local Files  │───▶│ TextProcessor│───▶│ SmartChunker │───▶│ Content   │ │
│  │ Google Drive │    │ PDFProcessor │    │ Transcript   │    │ Classifier│ │
│  │              │    │ AudioProcessor│   │ Chunker      │    │           │ │
│  │              │    │ JSONProcessor│    │              │    │           │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └─────┬─────┘ │
│                                                                     │       │
│                                                                     ▼       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Memory Candidates                                │  │
│  │  ┌─────────────────────────┐    ┌─────────────────────────┐          │  │
│  │  │   EPISODIC Memories     │    │   SEMANTIC Memories      │          │  │
│  │  │   - Events/experiences  │    │   - Beliefs/values       │          │  │
│  │  │   - Personal stories    │    │   - Knowledge/facts      │          │  │
│  │  │   - Temporal context    │    │   - Preferences          │          │  │
│  │  └─────────────────────────┘    └─────────────────────────┘          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                          │                                  │
│                                          ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Hippocampal Node (MCP)                             │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │  │
│  │  │    ChromaDB     │  │     Neo4j       │  │   Identity      │       │  │
│  │  │  Vector Store   │  │  Knowledge Graph│  │   Graph         │       │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- Hippocampal Node MCP server running (for storage)
- Optional: Whisper for audio transcription

### Quick Install

```bash
cd ~/ubik/somatic/ubik_ingest

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from ingest import IngestPipeline; print('Package ready')"

# Run tests
python -m pytest tests/ -v
```

### Dependencies

```
pdfplumber>=0.9.0      # PDF extraction
python-docx>=0.8.11    # DOCX processing
pyyaml>=6.0            # YAML front matter
httpx>=0.24.0          # HTTP client
openai-whisper>=20231117  # Audio transcription (optional)
pytesseract>=0.3.10    # OCR fallback (optional)
```

## Supported Formats

| Format | Extension(s) | Processor | Notes |
|--------|-------------|-----------|-------|
| Plain Text | `.txt`, `.md`, `.rst` | TextProcessor | UTF-8 with fallback encodings |
| PDF | `.pdf` | PDFProcessor | Text extraction + OCR fallback |
| Word | `.docx`, `.doc` | DOCXProcessor | Full paragraph/heading support |
| Transcripts | `.transcript` | TranscriptProcessor | YAML front matter + speaker turns |
| Audio | `.mp3`, `.wav`, `.m4a`, `.flac` | AudioProcessor | Whisper transcription |
| JSON | `.json` | JSONProcessor | Flattens structure to text |

## Usage

### Command Line Interface

```bash
# Process a single file (dry run)
python -m ingest.cli file ~/documents/memoir.pdf --dry-run

# Process a directory
python -m ingest.cli local ~/documents/letters/ --dry-run --verbose

# Process with storage to Hippocampal Node
python -m ingest.cli local ~/documents/ --mcp-host 192.168.1.100 --mcp-port 3000

# Filter by file type
python -m ingest.cli local ~/documents/ --extensions .txt .md --dry-run

# Use larger Whisper model for audio
python -m ingest.cli file ~/recordings/interview.mp3 --whisper-model medium
```

### Python API

```python
import asyncio
from pathlib import Path
from ingest import IngestPipeline, PipelineConfig
from mcp_client import HippocampalClient

async def main():
    # Dry run (no storage)
    pipeline = IngestPipeline(storage_mode=False)
    result = await pipeline.ingest_file(Path("~/documents/letter.txt"))

    print(f"Generated {result.episodic_count} episodic memories")
    print(f"Generated {result.semantic_count} semantic memories")

    for candidate in result.memory_candidates:
        print(f"  [{candidate.memory_type.value}] {candidate.content[:50]}...")

    # With MCP storage
    async with HippocampalClient() as mcp:
        async with IngestPipeline(mcp_client=mcp) as pipeline:
            batch = await pipeline.ingest_directory(Path("~/documents/"))
            print(batch.summary())

asyncio.run(main())
```

### Interactive Review Mode

For human-in-the-loop review and DPO training data collection:

```bash
# Review and classify content interactively
python interactive_ingest.py ~/documents/memoir.txt

# With custom output directory
python interactive_ingest.py ~/documents/ --output-dir ~/ubik/reviewed

# Resume a previous session
python interactive_ingest.py ~/documents/ --resume session_20241215_143022
```

Interactive commands:
- `[A]ccept` - Accept classification as-is
- `[E]dit` - Modify classification fields
- `[S]kip` - Skip with optional reason
- `[I]deal` - Write ideal response for DPO training
- `[B]atch` - Accept all remaining
- `[Q]uit` - Save and exit (resumable)

## Meeting Transcript Format

Transcripts use YAML front matter for metadata followed by speaker turns.

### Example Format

```yaml
---
# Session Metadata
meeting_date: 2024-12-15
meeting_type: family_meeting
meeting_title: "Holiday Planning"

# Participants
participants:
  - Gines
  - Elena
  - Adrian
  - Sofia

# Speaker Mapping (for Otter.ai format)
speakers:
  Speaker 1: Gines
  Speaker 2: Elena
  Speaker 3: Adrian
  Speaker 4: Sofia

# Context
location: Living room
duration_minutes: 45
recorder: Adrian
language: en

# Outcomes
decisions_made:
  - Travel to Lake House December 23-27
  - Host New Year's Eve dinner at home

action_items:
  - who: Elena
    what: Book Lake House
    due: December 18
    status: pending

# Flags
emotional_tone: warm, optimistic
contains_sensitive: false
---

Speaker 1  0:00
Alright everyone, thanks for making time for this.

Speaker 2  0:12
Of course. Should we start with the holiday plans?
```

### Supported Turn Formats

```
# Otter.ai format
Speaker 1  0:00
Text here

# Standard colon format
Name: Text here

# Markdown format
**Name**: Text here

# Bracketed format
[Speaker] Text here

# Timestamped format
[00:00:00] Name: Text here
```

## Configuration

### Pipeline Configuration

```python
from ingest import IngestPipeline, PipelineConfig
from ingest.chunkers import ChunkConfig

config = PipelineConfig(
    storage_mode=True,              # Enable MCP storage
    whisper_model="base",           # tiny, base, small, medium, large
    parallel_files=4,               # Concurrent file processing
    continue_on_storage_error=True, # Don't abort on storage failures
    execute_neo4j_ops=True,         # Execute transcript graph operations

    chunk_config=ChunkConfig(
        min_chunk_size=100,         # Minimum characters per chunk
        target_chunk_size=500,      # Target chunk size
        max_chunk_size=1500,        # Maximum before hard split
        overlap_size=50,            # Overlap for context
        respect_paragraphs=True,    # Prefer paragraph boundaries
        respect_sentences=True,     # Prefer sentence boundaries
    ),
)

pipeline = IngestPipeline(config=config)
```

### Environment Variables

```bash
# MCP Server Configuration
HIPPOCAMPAL_HOST=localhost
HIPPOCAMPAL_PORT=3000
MCP_TIMEOUT=30.0

# Processing Options
WHISPER_MODEL=base
OCR_ENABLED=true

# Logging
LOG_LEVEL=INFO
```

## Recommended Directory Structure

Organize source materials by content type for easier processing:

```
~/ubik/data/source_materials/
├── 1_ethical_constitution/     # Values, beliefs, principles
│   ├── core_values.md
│   └── life_philosophy.md
├── 2_letters/                  # Personal correspondence
│   ├── letter_to_grandchildren.md
│   └── family_letters/
├── 3_therapy/                  # Therapy session transcripts
│   ├── session_2024_01.transcript
│   └── session_2024_02.transcript
├── 4_journal/                  # Personal journal entries
│   ├── 2024_entries/
│   └── reflections.md
├── 5_family_meetings/          # Family meeting transcripts
│   ├── holiday_2024.transcript
│   └── weekly_meetings/
├── 6_recordings/               # Audio recordings
│   ├── stories/
│   └── interviews/
└── 7_documents/                # Documents (PDF, DOCX)
    ├── career_history.pdf
    └── family_history.docx
```

## Output Files

### Processed Memories

When using interactive mode or batch processing with output:

```json
// processed_memories.json
[
  {
    "content": "My father taught me to fish...",
    "memory_type": "episodic",
    "confidence": 0.92,
    "category": "family",
    "themes": ["childhood", "father", "outdoors"],
    "participants": ["father"],
    "emotional_valence": "positive",
    "importance": 0.85,
    "source_file": "memoir.txt",
    "reviewed_at": "2024-12-15T14:30:22"
  }
]
```

### DPO Training Data

```json
// ideal_responses.json
[
  {
    "memory_content": "My father taught me to fish at Lake Michigan...",
    "memory_type": "episodic",
    "ideal_response": "I remember those mornings at Lake Michigan so vividly...",
    "context": "when asked about childhood memories with father",
    "timestamp": "2024-12-15T14:35:00",
    "source_file": "memoir.txt"
  }
]
```

## Troubleshooting

### Common Issues

**PDF extraction fails**
```bash
# Install system dependencies
sudo apt install poppler-utils tesseract-ocr

# Verify installation
python -c "import pdfplumber; print('pdfplumber OK')"
```

**Audio transcription slow**
```bash
# Use a smaller Whisper model
python -m ingest.cli file recording.mp3 --whisper-model tiny

# Or disable audio processing
python -m ingest.cli local ~/docs/ --extensions .txt .md .pdf
```

**MCP connection refused**
```bash
# Check Hippocampal Node is running
curl http://localhost:3000/health

# Verify MCP configuration
python -c "
from mcp_client import HippocampalClient
import asyncio
async def test():
    async with HippocampalClient() as client:
        print('Connected:', client.is_connected)
asyncio.run(test())
"
```

**Memory classification seems wrong**
```bash
# Use interactive mode to review and correct
python interactive_ingest.py ~/documents/file.txt --output-dir ~/reviewed

# Check classification confidence in verbose mode
python -m ingest.cli file document.txt --dry-run --verbose
```

### Debug Mode

```bash
# Enable debug logging
python -m ingest.cli local ~/documents/ --dry-run --debug

# Or set environment variable
LOG_LEVEL=DEBUG python -m ingest.cli local ~/documents/
```

## Integration with UBIK Components

### Hippocampal Node (MCP)

The ingestion pipeline stores memories via the HippocampalClient:

```python
# Episodic memories
await client.store_episodic(
    content="The memory content...",
    memory_type="conversation",
    timestamp="2024-12-15T14:30:00Z",
    emotional_valence="positive",
    importance=0.8,
    participants="gines,elena",
    themes="family,holiday",
    source_file="meeting.transcript"
)

# Semantic memories
await client.store_semantic(
    content="I believe that family is the foundation...",
    knowledge_type="belief",
    category="values",
    confidence=0.95,
    stability="stable",
    source="constitution.md"
)
```

### Neo4j Graph Operations

Transcript processing generates graph operations for relationship tracking:

```python
# Automatically generated from transcript metadata
operations = [
    {
        "operation": "merge_relationship",
        "from_id": "Person:Gines",
        "rel_type": "ATTENDED",
        "to_id": "Event:FamilyMeeting_20241215",
        "properties": {"role": "facilitator"}
    }
]
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_mcp_integration.py -v

# Run with coverage
python -m pytest tests/ --cov=ingest --cov-report=html
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12 | Initial release |

## License

Part of the UBIK Digital Legacy Project. For personal use.
