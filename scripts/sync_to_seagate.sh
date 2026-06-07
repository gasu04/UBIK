#!/usr/bin/env bash
# Sync the UBIK project and Docker data to /Volumes/Seagate2T/UBIK/.
# Run daily via launchd (com.ubik.sync-to-seagate.plist).
# Uses rsync --checksum so only changed files are transferred.

set -euo pipefail

SEAGATE="/Volumes/Seagate2T"
DST="$SEAGATE/UBIK"
LOG="$DST/sync.log"

# ── Guard: Seagate must be mounted ──────────────────────────────────────────
if [[ ! -d "$SEAGATE" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [SKIP] Seagate2T not mounted." >&2
    exit 0
fi

mkdir -p "$DST"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

log "[START] UBIK → Seagate2T sync"

# ── 1. Project directory (code, scripts, configs) ───────────────────────────
log "[1/2] Syncing UBIK project..."
rsync -a --delete --checksum \
    --exclude="__pycache__/" \
    --exclude="*.pyc" \
    --exclude="*.pyo" \
    --exclude=".DS_Store" \
    --exclude="sync.log" \
    "/Volumes/990PRO 4T/UBIK/" \
    "$DST/project/"
log "[1/2] Project sync complete."

# ── 2. Docker data directory (Neo4j + ChromaDB) ─────────────────────────────
log "[2/2] Syncing UBIK data (Neo4j + ChromaDB)..."
rsync -a --delete --checksum \
    --exclude=".DS_Store" \
    "/Volumes/990PRO 4T/ubik/data/" \
    "$DST/data/"
log "[2/2] Data sync complete."

log "[DONE] UBIK → Seagate2T sync finished."
