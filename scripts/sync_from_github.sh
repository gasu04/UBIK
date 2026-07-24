#!/usr/bin/env bash
# Sync local UBIK master with origin/master via fetch + rebase.
# Designed to run unattended from cron. Exits 0 on success or already-current.
# Exits non-zero on rebase conflict (manual resolution required).

set -euo pipefail

REPO="/home/gasu/ubik"
LOG="/home/gasu/ubik/logs/git_sync.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() { echo "[$TIMESTAMP] $*" >> "$LOG"; }

cd "$REPO"

# Abort any in-progress rebase before attempting a new one
if [ -d ".git/rebase-merge" ] || [ -d ".git/rebase-apply" ]; then
    log "ERROR: rebase in progress — skipping sync. Run 'git rebase --abort' or resolve manually."
    exit 1
fi

git fetch origin >> "$LOG" 2>&1

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/master)

if [ "$LOCAL" = "$REMOTE" ]; then
    log "Already up to date ($LOCAL)."
    exit 0
fi

AHEAD=$(git rev-list origin/master..HEAD --count)
BEHIND=$(git rev-list HEAD..origin/master --count)
log "Syncing: $BEHIND new remote commit(s), $AHEAD local commit(s) ahead."

if git rebase origin/master >> "$LOG" 2>&1; then
    NEW=$(git rev-parse HEAD)
    log "OK — rebased to $NEW."
else
    log "ERROR: rebase conflict. Run 'git rebase --abort' or resolve manually in $REPO."
    exit 1
fi
