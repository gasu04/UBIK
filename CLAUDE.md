# Claude.md - Global Coding Standards

**Version:** 2.0.0  
**Derived From:** UBIK Digital Legacy Project  
**Purpose:** Persistent coding rules for all Claude-code sessions

---

## Core Philosophy

> **"Build systems that survive their creators."**

Every piece of code should embody:
- **Resilience** - Graceful degradation over catastrophic failure
- **Clarity** - Code that explains itself
- **Portability** - Run anywhere, depend on nothing specific
- **Maintainability** - Future developers are users too

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Configuration Management](#2-configuration-management)
3. [Resilience Patterns](#3-resilience-patterns)
4. [Runtime Optimization](#4-runtime-optimization)
5. [Documentation Standards](#5-documentation-standards)
6. [Code Organization](#6-code-organization)
7. [Error Handling](#7-error-handling)
8. [Testing Requirements](#8-testing-requirements)
9. [Cross-Platform Considerations](#9-cross-platform-considerations)
10. [Security Practices](#10-security-practices)

---

## 1. Project Structure

### 1.1 Standard Directory Layout

```
project_root/
├── config/                    # All configuration files
│   ├── .env.example          # Template (ALWAYS include)
│   ├── .env                  # Actual config (gitignored)
│   └── settings.py           # Configuration loader
├── src/                      # Source code (or use descriptive names)
│   ├── __init__.py          # Package exports
│   ├── core/                 # Core business logic
│   ├── services/             # External service integrations
│   └── utils/                # Shared utilities
├── tests/
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── fixtures/             # Test data
├── scripts/                  # Operational scripts
│   ├── setup.sh             # Environment setup
│   ├── health_check.py      # Health monitoring
│   └── run_service.sh       # Service launcher
├── docs/                     # Documentation
│   ├── architecture.md      # System design
│   ├── api.md               # API reference
│   └── deployment.md        # Deployment guide
├── logs/                     # Log output (gitignored)
├── data/                     # Data storage (gitignored)
├── docker-compose.yml        # Container orchestration
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Modern Python config
├── README.md                 # Project overview
└── Claude.md                 # AI assistant rules
```

### 1.2 Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Files | snake_case | `mcp_client.py` |
| Classes | PascalCase | `HippocampalClient` |
| Functions | snake_case | `get_rag_context()` |
| Constants | SCREAMING_SNAKE | `MAX_RETRIES = 3` |
| Private | Leading underscore | `_internal_state` |
| Config | SCREAMING_SNAKE in .env | `HIPPOCAMPAL_HOST` |

### 1.3 Package Exports

**Always define explicit exports in `__init__.py`:**

```python
"""
Package Name - Brief Description

Detailed description of what this package provides.
"""

from .module_a import ClassA, function_a
from .module_b import ClassB, ClassBConfig

__all__ = [
    'ClassA',
    'function_a',
    'ClassB',
    'ClassBConfig',
]

__version__ = '1.0.0'
```

---

## 2. Configuration Management

### 2.1 Environment-Based Configuration

**Always use dataclasses with factory defaults:**

```python
from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class ServiceConfig:
    """Configuration for external service."""
    host: str = field(default_factory=lambda: os.getenv("SERVICE_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("SERVICE_PORT", "8080")))
    timeout: float = field(default_factory=lambda: float(os.getenv("SERVICE_TIMEOUT", "30.0")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    
    @property
    def base_url(self) -> str:
        """Computed property for full URL."""
        return f"http://{self.host}:{self.port}"
    
    def validate(self) -> bool:
        """Validate configuration values."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port}")
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive: {self.timeout}")
        return True
```

### 2.2 Configuration Hierarchy

Load configuration in this order (later overrides earlier):

1. **Hardcoded defaults** - In dataclass field defaults
2. **Config files** - `.env` or `config.yaml`
3. **Environment variables** - `os.getenv()`
4. **Command line arguments** - For runtime overrides

### 2.3 Required .env.example

**Every project MUST have a `.env.example`:**

```bash
# =============================================================================
# Project Name - Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# Service Connection
# -----------------------------------------------------------------------------
SERVICE_HOST=localhost
SERVICE_PORT=8080
SERVICE_TIMEOUT=30.0

# -----------------------------------------------------------------------------
# Performance Tuning
# -----------------------------------------------------------------------------
MAX_RETRIES=3
RETRY_BASE_DELAY=1.0
CIRCUIT_BREAKER_THRESHOLD=5

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL=INFO
LOG_DIR=./logs
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

---

## 3. Resilience Patterns

### 3.1 Circuit Breaker Pattern

**Implement for all external service calls:**

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import time

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery

@dataclass
class CircuitBreaker:
    """Prevent cascade failures to external services."""
    threshold: int = 5          # Failures before opening
    timeout: float = 60.0       # Seconds before half-open
    
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    
    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and \
               time.time() - self._last_failure_time > self.timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state
    
    def record_success(self):
        self._failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
    
    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.threshold:
            self._state = CircuitState.OPEN
    
    def allow_request(self) -> bool:
        return self.state != CircuitState.OPEN
```

### 3.2 Retry with Exponential Backoff

**Standard retry pattern for transient failures:**

```python
import asyncio
from typing import TypeVar, Callable, Any

T = TypeVar('T')

async def retry_with_backoff(
    func: Callable[..., T],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    **kwargs
) -> T:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        
    Returns:
        Function result
        
    Raises:
        Last exception after all retries exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
    
    raise last_exception
```

### 3.3 Health Check Pattern

**Every service MUST expose health status:**

```python
from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime

@dataclass
class HealthStatus:
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    components: Dict[str, Dict[str, Any]]
    
    @classmethod
    def check_all(cls, checks: Dict[str, Callable]) -> "HealthStatus":
        """Run all health checks and aggregate results."""
        components = {}
        
        for name, check_func in checks.items():
            try:
                result = check_func()
                components[name] = {"status": "healthy", **result}
            except Exception as e:
                components[name] = {"status": "unhealthy", "error": str(e)}
        
        all_healthy = all(c.get("status") == "healthy" for c in components.values())
        
        return cls(
            status="healthy" if all_healthy else "degraded",
            timestamp=datetime.now().isoformat(),
            components=components
        )
```

---

## 4. Runtime Optimization

### 4.1 Lazy Loading

**Never initialize expensive resources until needed:**

```python
class ServiceClient:
    """Client with lazy-loaded connections."""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self._connection: Optional[Connection] = None  # Lazy
        self._cache: Optional[Cache] = None            # Lazy
    
    @property
    def connection(self) -> Connection:
        """Lazy-load database connection."""
        if self._connection is None:
            self._connection = Connection(self.config.connection_string)
            logger.info("Database connection initialized")
        return self._connection
    
    @property
    def cache(self) -> Cache:
        """Lazy-load cache client."""
        if self._cache is None:
            self._cache = Cache(self.config.cache_url)
        return self._cache
```

### 4.2 Connection Pooling

**Always pool connections for external services:**

```python
import httpx

# HTTP Connection Pool
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(30.0),
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=30.0
    )
)

# Database Connection Pool (example with asyncpg)
pool = await asyncpg.create_pool(
    dsn=config.database_url,
    min_size=5,
    max_size=20,
    command_timeout=30
)
```

### 4.3 Context Managers for Resources

**Always use context managers for resource lifecycle:**

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def service_session():
    """Managed session for service client."""
    client = ServiceClient()
    try:
        await client.connect()
        yield client
    finally:
        await client.disconnect()

# Usage
async with service_session() as client:
    result = await client.query("...")
```

### 4.4 Caching Strategy

```python
from functools import lru_cache
from datetime import datetime, timedelta

# Simple in-memory cache for expensive computations
@lru_cache(maxsize=128)
def compute_expensive_result(input_hash: str) -> Result:
    """Cached computation."""
    return expensive_operation(input_hash)

# Time-based cache
class TTLCache:
    """Simple TTL cache for configuration values."""
    
    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, tuple] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._ttl:
                return value
            del self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self._cache[key] = (value, datetime.now())
```

---

## 5. Documentation Standards

### 5.1 Module Docstrings

**Every module MUST have a header docstring:**

```python
#!/usr/bin/env python3
"""
Module Name - Brief Description

Extended description explaining:
- What this module does
- How it fits into the larger system
- Key classes/functions it provides
- Important design decisions

Usage:
    from module import MainClass
    
    client = MainClass(config)
    result = await client.operation()

Dependencies:
    - external_package>=1.0.0
    - another_package>=2.0.0

Author: Project Team
Version: 1.0.0
"""
```

### 5.2 Function/Method Docstrings

**Use Google-style docstrings:**

```python
async def process_query(
    query: str,
    context: Optional[str] = None,
    max_results: int = 10
) -> List[Result]:
    """
    Process a semantic query against the knowledge base.
    
    Performs vector similarity search and returns ranked results
    with relevance scores.
    
    Args:
        query: The search query text
        context: Optional context to narrow search scope
        max_results: Maximum number of results to return (1-100)
        
    Returns:
        List of Result objects ordered by relevance score,
        each containing:
            - id: Unique result identifier
            - content: Matched content text
            - score: Relevance score (0.0-1.0)
            - metadata: Additional result metadata
            
    Raises:
        ConnectionError: If database is unreachable
        ValidationError: If query is empty or max_results invalid
        
    Example:
        >>> results = await process_query("family values", max_results=5)
        >>> for r in results:
        ...     print(f"{r.score:.2f}: {r.content[:50]}")
        0.95: Family has always been at the center of my life...
        
    Note:
        Results are cached for 5 minutes. Use force_refresh=True
        to bypass cache.
    """
```

### 5.3 Class Docstrings

```python
class RAGService:
    """
    Unified RAG service for memory-augmented generation.
    
    Orchestrates the complete pipeline:
    1. Retrieve relevant memories from knowledge store
    2. Select appropriate prompt template
    3. Generate response via LLM
    4. Post-process for quality output
    
    Attributes:
        config: Service configuration
        prompt_manager: Manages prompt templates
        circuit_breaker: Prevents cascade failures
        
    Example:
        service = RAGService(config)
        response = await service.ask("What matters most?")
        print(response.text)
        
    Thread Safety:
        This class is thread-safe. Multiple coroutines can
        share a single instance.
        
    See Also:
        - HippocampalClient: Memory retrieval client
        - VoicePromptManager: Prompt template management
    """
```

### 5.4 Architecture Documentation

**Include ASCII diagrams in architecture docs:**

```
┌─────────────────────────────────────────────────────────────┐
│                    System Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   Client     │◄───────►│   Gateway    │                 │
│  │   Layer      │  HTTP   │   Service    │                 │
│  └──────────────┘         └──────┬───────┘                 │
│                                  │                          │
│                    ┌─────────────┼─────────────┐           │
│                    ▼             ▼             ▼           │
│             ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│             │ Service  │  │ Service  │  │ Service  │      │
│             │    A     │  │    B     │  │    C     │      │
│             └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│                  │             │             │             │
│                  └─────────────┼─────────────┘             │
│                                ▼                            │
│                         ┌──────────┐                       │
│                         │ Database │                       │
│                         │  Layer   │                       │
│                         └──────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Code Organization

### 6.1 Single Responsibility

**Each class/function should do ONE thing:**

```python
# BAD - Multiple responsibilities
class DataManager:
    def fetch_data(self): ...
    def transform_data(self): ...
    def store_data(self): ...
    def send_notification(self): ...

# GOOD - Separated concerns
class DataFetcher:
    def fetch(self) -> RawData: ...

class DataTransformer:
    def transform(self, raw: RawData) -> ProcessedData: ...

class DataStore:
    def save(self, data: ProcessedData) -> str: ...

class NotificationService:
    def notify(self, event: Event) -> None: ...
```

### 6.2 Dependency Injection

**Inject dependencies, don't hardcode them:**

```python
# BAD - Hardcoded dependency
class ReportGenerator:
    def __init__(self):
        self.db = PostgresDatabase()  # Hardcoded!
    
# GOOD - Injected dependency
class ReportGenerator:
    def __init__(self, database: Database):
        self.db = database
        
# Usage
db = PostgresDatabase(config)
generator = ReportGenerator(database=db)

# Testing
mock_db = MockDatabase()
test_generator = ReportGenerator(database=mock_db)
```

### 6.3 Type Hints

**Use type hints EVERYWHERE:**

```python
from typing import Optional, List, Dict, Any, TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class Result(Generic[T]):
    """Generic result container."""
    data: T
    success: bool
    error: Optional[str] = None

async def fetch_items(
    ids: List[str],
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100
) -> Result[List[Item]]:
    """Fetch items with full type safety."""
    ...
```

---

## 7. Error Handling

### 7.1 Custom Exception Hierarchy

**Define project-specific exceptions:**

```python
class ProjectError(Exception):
    """Base exception for all project errors."""
    pass

class ConfigurationError(ProjectError):
    """Invalid or missing configuration."""
    pass

class ConnectionError(ProjectError):
    """Failed to connect to external service."""
    def __init__(self, service: str, reason: str):
        self.service = service
        self.reason = reason
        super().__init__(f"Connection to {service} failed: {reason}")

class ValidationError(ProjectError):
    """Input validation failed."""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"Validation error on '{field}': {message}")
```

### 7.2 Error Handling Pattern

```python
async def safe_operation(input_data: InputData) -> Result:
    """
    Perform operation with comprehensive error handling.
    
    Returns Result object instead of raising exceptions
    for expected error conditions.
    """
    # Validation
    try:
        validated = validate(input_data)
    except ValidationError as e:
        return Result(success=False, error=f"Validation failed: {e}")
    
    # Operation
    try:
        result = await perform_operation(validated)
        return Result(success=True, data=result)
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return Result(success=False, error="Service temporarily unavailable")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        # Re-raise unexpected errors
        raise
```

### 7.3 Logging Standards

```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger(__name__)

# Usage - always include context
logger.info("Operation started", 
            operation="fetch_data", 
            user_id=user_id,
            request_id=request_id)

logger.error("Operation failed",
             operation="fetch_data",
             error=str(e),
             duration_ms=elapsed)
```

---

## 8. Testing Requirements

### 8.1 Test Structure

```
tests/
├── unit/
│   ├── test_models.py
│   ├── test_services.py
│   └── test_utils.py
├── integration/
│   ├── test_database.py
│   ├── test_api.py
│   └── test_external_services.py
├── e2e/
│   └── test_full_pipeline.py
├── fixtures/
│   ├── sample_data.json
│   └── mock_responses.py
└── conftest.py
```

### 8.2 Test Naming Convention

```python
def test_<function>_<scenario>_<expected_result>():
    """Descriptive test names."""
    pass

# Examples
def test_validate_input_empty_string_raises_validation_error():
    ...

def test_fetch_data_network_timeout_returns_empty_result():
    ...

def test_process_query_valid_input_returns_ranked_results():
    ...
```

### 8.3 Async Test Pattern

```python
import pytest
import pytest_asyncio

@pytest.fixture
async def client():
    """Async fixture for test client."""
    client = ServiceClient(test_config)
    await client.connect()
    yield client
    await client.disconnect()

@pytest.mark.asyncio
async def test_query_returns_results(client):
    """Test async operation."""
    results = await client.query("test query")
    assert len(results) > 0
    assert all(hasattr(r, 'score') for r in results)
```

### 8.4 Coverage Requirements

- **Minimum coverage:** 80%
- **Critical paths:** 100%
- **Error handling:** Explicitly tested

---

## 9. Cross-Platform Considerations

### 9.1 Path Handling

```python
from pathlib import Path

# GOOD - Cross-platform paths
config_dir = Path.home() / "project" / "config"
data_file = config_dir / "data.json"

# Check existence
if data_file.exists():
    content = data_file.read_text()

# BAD - Hardcoded paths
config_dir = "/home/user/project/config"  # Linux only!
```

### 9.2 Environment Detection

```python
import sys
import platform

def get_platform_info() -> Dict[str, str]:
    """Get current platform information."""
    return {
        "os": platform.system(),          # 'Linux', 'Darwin', 'Windows'
        "arch": platform.machine(),       # 'x86_64', 'arm64'
        "python": platform.python_version(),
        "node": platform.node()
    }

def get_default_data_dir() -> Path:
    """Get platform-appropriate data directory."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "ProjectName"
    elif system == "Windows":
        return Path(os.environ.get("APPDATA", "")) / "ProjectName"
    else:  # Linux/Unix
        return Path.home() / ".local" / "share" / "projectname"
```

### 9.3 Docker Best Practices

```dockerfile
# Use multi-stage builds
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim as runtime

# Non-root user
RUN useradd -m -u 1000 appuser
USER appuser

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --chown=appuser:appuser . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

EXPOSE 8080
CMD ["python", "-m", "service"]
```

---

## 10. Security Practices

### 10.1 Secrets Management

```python
# NEVER hardcode secrets
# BAD
API_KEY = "sk-abc123..."

# GOOD - Environment variable
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ConfigurationError("API_KEY environment variable required")

# BETTER - Secrets manager
from secrets_manager import get_secret
API_KEY = get_secret("project/api_key")
```

### 10.2 Input Validation

```python
from pydantic import BaseModel, validator, constr

class UserInput(BaseModel):
    """Validated user input."""
    query: constr(min_length=1, max_length=1000)
    limit: int = 10
    
    @validator('limit')
    def limit_must_be_reasonable(cls, v):
        if not 1 <= v <= 100:
            raise ValueError('limit must be between 1 and 100')
        return v
    
    @validator('query')
    def query_must_be_safe(cls, v):
        # Prevent injection attacks
        dangerous_patterns = ['<script>', 'DROP TABLE', '--']
        for pattern in dangerous_patterns:
            if pattern.lower() in v.lower():
                raise ValueError('Invalid characters in query')
        return v.strip()
```

### 10.3 Dependency Security

```bash
# In CI/CD pipeline
pip install safety
safety check -r requirements.txt

# Pin exact versions in production
pip freeze > requirements.lock
```

---

## Quick Reference Card

### File Creation Checklist
- [ ] Module docstring with purpose, usage, dependencies
- [ ] Type hints on all functions
- [ ] Dataclass for configuration
- [ ] Custom exceptions for error cases
- [ ] Context manager for resources
- [ ] Health check endpoint
- [ ] Unit tests file created

### Code Review Checklist
- [ ] No hardcoded values (use config)
- [ ] Error handling is comprehensive
- [ ] Logging is contextual
- [ ] Resources are properly closed
- [ ] Type hints are accurate
- [ ] Docstrings are complete
- [ ] Tests cover edge cases

### Performance Checklist
- [ ] Lazy loading for expensive resources
- [ ] Connection pooling enabled
- [ ] Appropriate caching in place
- [ ] Circuit breaker for external calls
- [ ] Async where beneficial
- [ ] No N+1 query patterns

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2025-01 | Derived from UBIK project learnings |
| 1.0.0 | 2024-01 | Initial release |

---

*"The best code is no code at all. The second best is code so clear it documents itself."*
