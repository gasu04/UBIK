# Ubik Hippocampal Node

Memory storage and retrieval system for the Ubik Digital Legacy Project.

## Overview

The Hippocampal Node is responsible for storing, indexing, and retrieving memories across three complementary systems:

| System | Storage | Purpose |
|--------|---------|---------|
| **Episodic Memory** | ChromaDB | Time-bound personal experiences, conversations, events |
| **Semantic Memory** | ChromaDB | Conceptual knowledge, beliefs, values, preferences |
| **Identity Graph** | Neo4j | Parfitian identity relationships, values, traits |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hippocampal Node                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   Somatic    │◄───────►│  MCP Server  │                 │
│  │    Node      │  MCP    │  (FastMCP)   │                 │
│  └──────────────┘         └──────┬───────┘                 │
│                                  │                          │
│                    ┌─────────────┼─────────────┐           │
│                    ▼             ▼             ▼           │
│             ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│             │ Episodic │  │ Semantic │  │ Identity │      │
│             │ ChromaDB │  │ ChromaDB │  │  Neo4j   │      │
│             └──────────┘  └──────────┘  └──────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Neo4j 5.x
- ChromaDB 1.x

### Installation

```bash
# Clone the repository
cd /path/to/ubik/hippocampal

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

### Start Services

```bash
# Start Neo4j and ChromaDB via Docker
docker-compose up -d

# Initialize Neo4j schema
python init_neo4j_schema.py

# Set up ChromaDB collections
python setup_chromadb.py

# Start MCP server
./run_mcp.sh -d
```

### Verify Installation

```bash
# Check service status
./run_mcp.sh status

# Run health check
python health_check.py

# View logs
./run_mcp.sh logs
```

## MCP Tools

The Hippocampal Node exposes these tools via the Model Context Protocol:

### Memory Operations

| Tool | Description |
|------|-------------|
| `store_episodic` | Store a new episodic memory |
| `query_episodic` | Search episodic memories semantically |
| `store_semantic` | Store semantic knowledge (respects freeze state) |
| `query_semantic` | Search semantic knowledge |

### Identity Graph Operations

| Tool | Description |
|------|-------------|
| `get_identity_context` | Traverse identity graph from a concept |
| `update_identity_graph` | Add/update identity relationships |
| `get_memory_stats` | Get counts for all memory stores |

### Administrative

| Tool | Description |
|------|-------------|
| `freeze_voice` | Lock semantic memory (training complete) |
| `unfreeze_voice` | Unlock semantic memory for updates |

## Configuration

All configuration is done via environment variables. See `.env.example` for details.

### Required Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | (required) |
| `CHROMADB_HOST` | ChromaDB hostname | `localhost` |
| `CHROMADB_PORT` | ChromaDB port | `8001` |
| `CHROMADB_TOKEN` | ChromaDB auth token | (required) |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_HOST` | MCP server bind address | `0.0.0.0` |
| `MCP_PORT` | MCP server port | `8080` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Project Structure

```
hippocampal/
├── mcp_server.py          # Main MCP server with all tools
├── exceptions.py          # Custom exception hierarchy
├── init_neo4j_schema.py   # Neo4j schema initialization
├── setup_chromadb.py      # ChromaDB collection setup
├── health_check.py        # Component health checker
├── run_mcp.sh             # Server management script
├── requirements.txt       # Python dependencies
├── .env.example           # Configuration template
├── scripts/
│   └── ingest_memories.py # Memory ingestion utility
├── data/
│   └── samples/           # Sample memory data
├── tests/
│   ├── conftest.py        # Pytest configuration
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
└── logs/                  # Log files (gitignored)
```

## Memory Data Formats

### Episodic Memory

```json
{
  "content": "Memory content text",
  "memory_type": "letter|therapy_session|family_meeting|conversation|event",
  "timestamp": "2024-01-15T10:30:00Z",
  "emotional_valence": "positive|negative|neutral|reflective|mixed",
  "importance": 0.0-1.0,
  "participants": "person1,person2",
  "themes": "theme1,theme2"
}
```

### Semantic Knowledge

```json
{
  "content": "Knowledge statement",
  "knowledge_type": "belief|value|preference|fact|opinion",
  "category": "family|relationships|philosophy|communication|career|health",
  "confidence": 0.0-1.0,
  "stability": "core|stable|evolving",
  "source": "reflection|therapy|reading|life_experience"
}
```

## Frozen Voice Architecture

The Hippocampal Node implements the "Frozen Voice, Living Mind" pattern:

- **Frozen Voice**: After voice training is complete, semantic memories are locked to preserve the authentic voice
- **Living Mind**: Episodic memories can always be added, allowing the system to continue learning

```python
# Lock semantic memory after training
await freeze_voice()

# Attempts to store semantic memory will fail
await store_semantic(...)  # Raises FrozenVoiceError

# Episodic memory always works
await store_episodic(...)  # Success
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/ -m integration
```

### Code Style

This project follows the coding standards defined in `CLAUDE.md`:

- Type hints on all functions
- Google-style docstrings
- Custom exceptions for error handling
- Structured logging with context

## Troubleshooting

### Server Won't Start

```bash
# Check for existing processes
./run_mcp.sh status

# Kill any orphan processes and restart
./run_mcp.sh restart
```

### Database Connection Errors

```bash
# Verify services are running
docker-compose ps

# Check Neo4j logs
docker-compose logs neo4j

# Check ChromaDB logs
docker-compose logs chromadb
```

### Memory Queries Return Empty

```bash
# Check collection stats
python scripts/ingest_memories.py --stats

# Ingest sample data
python scripts/ingest_memories.py --all data/samples/
```

## License

Part of the Ubik Digital Legacy Project. Private repository.

## Related

- [Ubik Project Documentation](../UBIK_PROJECT_DOCUMENTATION.md)
- [Coding Standards](../CLAUDE.md)
