#!/usr/bin/env bash
# Copy SESSIONS.md to GdriveMirror so Google Drive Desktop uploads it automatically.
# Run manually or via the post-commit hook.

UBIK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$UBIK_ROOT/SESSIONS.md"
DST="/Volumes/GdriveMirror/Ubik/SESSIONS.md"

if [[ ! -f "$SRC" ]]; then
    echo "sync_sessions: $SRC not found — nothing to sync."
    exit 0
fi

if [[ ! -d "/Volumes/GdriveMirror/Ubik" ]]; then
    echo "sync_sessions: GdriveMirror not mounted — skipping sync."
    exit 0
fi

cp "$SRC" "$DST" && echo "sync_sessions: SESSIONS.md → GdriveMirror/Ubik ✓"
