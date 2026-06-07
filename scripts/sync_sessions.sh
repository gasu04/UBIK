#!/usr/bin/env bash
# Copy SESSIONS.md to the live Google Drive mount (gsanchezurrutia@gmail.com).
# Google Drive for Desktop picks it up and uploads automatically.
# Run manually or via the post-commit hook.

UBIK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$UBIK_ROOT/SESSIONS.md"
DST="$HOME/Library/CloudStorage/GoogleDrive-gsanchezurrutia@gmail.com/My Drive/Ubik/SESSIONS.md"
DST_DIR="$(dirname "$DST")"

if [[ ! -f "$SRC" ]]; then
    echo "sync_sessions: $SRC not found — nothing to sync."
    exit 0
fi

if [[ ! -d "$DST_DIR" ]]; then
    echo "sync_sessions: GDrive not mounted (${DST_DIR}) — skipping sync."
    exit 0
fi

cp "$SRC" "$DST" && echo "sync_sessions: SESSIONS.md → gsanchezurrutia@gmail.com/Ubik ✓"
