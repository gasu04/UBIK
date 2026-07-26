#!/usr/bin/env bash
# Enable UBIK's portable git hooks for THIS clone.
#
# Git does not track .git/hooks/, so each clone must opt in. This points
# core.hooksPath at the tracked githooks/ directory (and makes every hook in it
# executable). Run once after cloning UBIK:
#
#     bash scripts/setup_hooks.sh
#
# Requirements: git >= 2.9 (core.hooksPath support). No elevated privileges.
#
# What it does:
#   - Sets core.hooksPath to <repo>/githooks (so git reads the tracked hooks).
#   - chmod +x every file in githooks/ (git ignores non-executable hooks).
#   - Prints the active hooks.
#
# Note: with core.hooksPath set, git reads ONLY githooks/ and ignores
# .git/hooks/*. If you keep personal hooks in .git/hooks/, they will no longer
# fire — move them into githooks/ if you want them active.

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo -e "${RED}Error:${NC} not inside a git repository." >&2
    exit 1
}
HOOKS_DIR="$REPO_ROOT/githooks"

if [[ ! -d "$HOOKS_DIR" ]]; then
    echo -e "${RED}Error:${NC} githooks/ not found at $HOOKS_DIR" >&2
    exit 1
fi

# git >= 2.9 is required for core.hooksPath.
if ! git config core.hooksPath >/dev/null 2>&1; then :; fi
git_major="$(git --version | awk '{print $3}' | cut -d. -f1)"
if (( git_major < 2 )) || { (( git_major == 2 )) && [[ "$(git --version | awk '{print $3}' | cut -d. -f2)" -lt 9 ]]; }; then
    echo -e "${RED}Error:${NC} git >= 2.9 required for core.hooksPath (you have $(git --version))." >&2
    exit 1
fi

# Make every tracked hook executable (git skips non-executable hooks).
find "$HOOKS_DIR" -maxdepth 1 -type f ! -name '.*' -exec chmod +x {} \;

# Point git at the tracked hooks dir (repo-relative, so it survives moves).
git config core.hooksPath githooks

echo -e "${GREEN}Hooks enabled:${NC} core.hooksPath = $(git config core.hooksPath)"
echo -e "${YELLOW}Active hooks:${NC}"
for h in "$HOOKS_DIR"/*; do
    [[ -f "$h" ]] && printf '  • %s\n' "$(basename "$h")"
done
echo ""
echo "To disable: git config --unset core.hooksPath"
