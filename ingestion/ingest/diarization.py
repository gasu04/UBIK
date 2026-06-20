"""
UBIK Ingestion System - Per-file Diarization Detection

Decides whether a transcript body is **mono** (one or no distinct speaker
label across the whole text) or **multi** (two or more distinct speakers),
using the heuristic from the Phase 3 spec: *if fewer than two distinct,
non-empty speaker labels appear across all turns, the file is mono.*

How it fits in: `enrich.py` calls :func:`detect_diarization` on the extracted
body of a transcript-type source. The result drives ``diarization_status`` /
``diarization_warning`` in the enrichment and, via the Python-side hard rule,
``voice_corpus_eligible`` (mono can never be voice-corpus eligible). This is
deliberately a *new* module so the existing ``transcript_processor.py`` (whose
``SpeakerTurnParser`` still uses ASCII-only ``[A-Za-z]`` classes) is left
untouched — the label scan here is **accent-safe** (Unicode ``\\w``), per
``qa/learned_rules.md`` (Ginés, Adrián, Sofía must parse).

Key functions:
    detect_diarization(text) -> "mono" | "multi"
    speaker_labels(text)     -> set[str]   (distinct normalized labels)

Usage:
    from ingest.diarization import detect_diarization
    status = detect_diarization(body)   # "mono" or "multi"

Dependencies: stdlib only (``re``, ``unicodedata``).

Tier classification: Tier 2 (standard, 80% coverage). A wrong call is loud
and recoverable: it is visible at Gate 1, and the real voice-corpus guarantee
is the content-type registry invariant (transcript sources are
``voice_corpus_eligible: false``) plus the override in ``enrich.py`` — this
detector is defense-in-depth, not the sole gate.

Version: 0.1.0
"""

import re
import unicodedata
from typing import Set

__all__ = ["detect_diarization", "speaker_labels", "MONO", "MULTI"]

MONO = "mono"
MULTI = "multi"

# Max tokens a plausible speaker label may have ("Gines Alberto" = 2,
# "Speaker 1" = 2). Longer => almost certainly a sentence, not a label.
_MAX_LABEL_TOKENS = 4
# Max characters for a label, a second guard against matching prose.
_MAX_LABEL_CHARS = 40

# Common Spanish/English line-openers that take a colon but are NOT speakers.
# Without this, prose like "Nota: ..." / "Fecha: ..." would be miscounted as
# distinct speakers and flip a mono file to multi. Compared accent-folded.
_NON_SPEAKER_OPENERS = frozenset({
    "nota", "notas", "fecha", "hora", "tema", "temas", "pregunta", "preguntas",
    "respuesta", "resumen", "accion", "acciones", "acuerdo", "acuerdos",
    "objetivo", "objetivos", "lugar", "asunto", "duracion", "participantes",
    "note", "notes", "date", "time", "topic", "topics", "question", "answer",
    "summary", "action", "actions", "agenda", "location", "subject", "attendees",
})

# Accent-safe label classes. ``\w`` is Unicode-aware for ``str`` patterns in
# Python 3, so accented names (Ginés) are included; ASCII ``[A-Za-z]`` is not.
# A label starts with a letter (never a digit) and may contain letters,
# digits, spaces, and a few name punctuation marks.
_LABEL = r"[^\W\d_][\w .'’-]{0,%d}?" % _MAX_LABEL_CHARS

# "Name: text" — colon-delimited speaker turn at line start.
_COLON_RE = re.compile(r"^\s*(%s):\s+\S" % _LABEL, re.MULTILINE)
# "Name  0:00" / "Name  00:00:00" — Otter-style label + 2+ spaces + timestamp.
_OTTER_RE = re.compile(r"^\s*(%s)\s{2,}\d{1,2}:\d{2}" % _LABEL, re.MULTILINE)
# "**Name:**" / "**Name**:" — markdown bold speaker label.
_MARKDOWN_RE = re.compile(r"^\s*\*\*\s*([^*:]{1,%d}?)\s*:?\*\*" % _MAX_LABEL_CHARS,
                          re.MULTILINE)

_ALL_PATTERNS = (_COLON_RE, _OTTER_RE, _MARKDOWN_RE)


def _normalize(label: str) -> str:
    """Lowercase + accent-fold a label for distinct-counting (NFKD)."""
    folded = unicodedata.normalize("NFKD", label.strip())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(folded.lower().split())


def _is_plausible_label(label: str) -> bool:
    """True if a captured string looks like a speaker label, not prose."""
    label = label.strip()
    if not label:
        return False
    tokens = label.split()
    if len(tokens) > _MAX_LABEL_TOKENS:
        return False
    # First token must start with a letter — allows "Speaker 1"/"Gines Alberto"
    # while rejecting "12:30". (Only the first token, so trailing speaker
    # numbers are fine.)
    if not tokens[0][:1].isalpha():
        return False
    # Reject common non-speaker line-openers ("Nota:", "Fecha:", ...).
    return _normalize(label) not in _NON_SPEAKER_OPENERS


def speaker_labels(text: str) -> Set[str]:
    """
    Return the set of distinct, normalized speaker labels found in *text*.

    Scans the colon, Otter-timestamp, and markdown-bold label forms with
    accent-safe patterns, keeps only plausible labels (<= 4 name-like tokens),
    and folds accents/case so "Ginés" and "gines" count once.

    Args:
        text: Transcript body (front matter already stripped).

    Returns:
        Set of normalized label strings (possibly empty).
    """
    found: Set[str] = set()
    for pattern in _ALL_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1)
            if _is_plausible_label(raw):
                found.add(_normalize(raw))
    return found


def detect_diarization(text: str) -> str:
    """
    Classify a transcript body as mono- or multi-speaker.

    Args:
        text: Transcript body (front matter already stripped).

    Returns:
        ``"multi"`` if two or more distinct speaker labels are present,
        otherwise ``"mono"`` (covers the unattributed / single-speaker case).

    Example:
        >>> detect_diarization("Ginés: hola\\nAdrián: qué tal")
        'multi'
        >>> detect_diarization("just a wall of text with no labels")
        'mono'
    """
    return MULTI if len(speaker_labels(text)) >= 2 else MONO
