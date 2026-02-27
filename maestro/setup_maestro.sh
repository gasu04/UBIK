#!/usr/bin/env bash
# =============================================================================
# UBIK Maestro — Deployment Script
# =============================================================================
# Installs Maestro dependencies, creates .env configuration (importing secrets
# from hippocampal/.env where available), creates the log directory, and runs
# a validation probe on the local node.
#
# Works on both nodes:
#   Hippocampal  macOS   arm64   venv at {UBIK_ROOT}/hippocampal/venv
#   Somatic      Linux   x86_64  venv at {UBIK_ROOT}/venv
#
# Usage:
#   cd {UBIK_ROOT}
#   bash maestro/setup_maestro.sh
#
# Environment overrides (all optional):
#   UBIK_ROOT              Override default project root path
#   UBIK_NODE_TYPE         Force "hippocampal" or "somatic" (skip auto-detect)
#   HIPPOCAMPAL_VENV_PATH  Override venv directory on Hippocampal
#   SOMATIC_VENV_PATH      Override venv directory on Somatic
#   MAESTRO_LOG_DIR        Override log directory
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers (no external deps)
# ---------------------------------------------------------------------------

_green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
_yellow() { printf '\033[1;33m%s\033[0m\n' "$*"; }
_red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
_bold()   { printf '\033[1m%s\033[0m\n' "$*"; }
_sep()    { printf '%s\n' "────────────────────────────────────────────────────"; }
_step()   { echo ""; _bold "$*"; }

# Portable sed -i: BSD sed (macOS) needs '' for no-backup; GNU sed omits it.
_sed_i() {
    local script="$1"; local file="$2"
    if [[ "$PLATFORM" == "Darwin" ]]; then
        sed -i '' "$script" "$file"
    else
        sed -i "$script" "$file"
    fi
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_sep
_bold " UBIK Maestro — Deployment"
_sep
echo ""

PLATFORM="$(uname -s)"

# ---------------------------------------------------------------------------
# Step 1 — Locate UBIK root
# ---------------------------------------------------------------------------
_step "Step 1/6 — UBIK root"

if [[ -n "${UBIK_ROOT:-}" ]]; then
    echo "  Using UBIK_ROOT from environment."
elif [[ "$PLATFORM" == "Darwin" ]]; then
    UBIK_ROOT="/Volumes/990PRO 4T/UBIK"
else
    UBIK_ROOT="/home/gasu/ubik"
fi

if [[ ! -d "$UBIK_ROOT" ]]; then
    _red "  ERROR: UBIK_ROOT not found: $UBIK_ROOT"
    _yellow "  Set the UBIK_ROOT environment variable and retry."
    exit 1
fi

MAESTRO_DIR="$UBIK_ROOT/maestro"
REQUIREMENTS="$MAESTRO_DIR/requirements.txt"
ENV_EXAMPLE="$MAESTRO_DIR/.env.example"
ENV_FILE="$MAESTRO_DIR/.env"
LOG_DIR="${MAESTRO_LOG_DIR:-$UBIK_ROOT/logs/maestro}"

_green "  UBIK_ROOT : $UBIK_ROOT"
echo   "  Maestro   : $MAESTRO_DIR"

# ---------------------------------------------------------------------------
# Step 2 — Detect node type
# ---------------------------------------------------------------------------
_step "Step 2/6 — Node detection"

HOSTNAME_LC="$(hostname -s 2>/dev/null | tr '[:upper:]' '[:lower:]' || echo "unknown")"
NODE_TYPE="unknown"

if [[ -n "${UBIK_NODE_TYPE:-}" ]]; then
    NODE_TYPE="${UBIK_NODE_TYPE,,}"  # lower-case
    echo "  UBIK_NODE_TYPE override: $NODE_TYPE"
elif [[ "$PLATFORM" == "Darwin" ]]; then
    NODE_TYPE="hippocampal"
elif [[ "$PLATFORM" == "Linux" ]]; then
    NODE_TYPE="somatic"
fi

if [[ "$NODE_TYPE" != "hippocampal" && "$NODE_TYPE" != "somatic" ]]; then
    _red "  ERROR: Cannot determine node type."
    _yellow "  Set UBIK_NODE_TYPE=hippocampal or UBIK_NODE_TYPE=somatic and retry."
    exit 1
fi

_green "  Node type : $NODE_TYPE  (platform: $PLATFORM, host: $HOSTNAME_LC)"

# ---------------------------------------------------------------------------
# Step 3 — Install Python dependencies
# ---------------------------------------------------------------------------
_step "Step 3/6 — Python dependencies"

if [[ "$NODE_TYPE" == "hippocampal" ]]; then
    VENV_PATH="${HIPPOCAMPAL_VENV_PATH:-$UBIK_ROOT/hippocampal/venv}"
    if [[ ! -d "$VENV_PATH" ]]; then
        _red "  ERROR: venv not found: $VENV_PATH"
        _yellow "  Create it with: python3 -m venv $VENV_PATH"
        exit 1
    fi
    PYTHON_BIN="$VENV_PATH/bin/python"
    PIP_BIN="$VENV_PATH/bin/pip"

    echo "  venv      : $VENV_PATH"
    echo "  Installing requirements …"
    "$PIP_BIN" install --quiet --upgrade pip
    "$PIP_BIN" install --quiet -r "$REQUIREMENTS"

    _run_python() { "$PYTHON_BIN" "$@"; }

else
    VENV_PATH="${SOMATIC_VENV_PATH:-$HOME/pytorch_env}"
    if [[ ! -d "$VENV_PATH" ]]; then
        _red "  ERROR: venv not found: $VENV_PATH"
        _yellow "  Create it with: python3 -m venv $VENV_PATH"
        _yellow "  Or set SOMATIC_VENV_PATH to an existing virtualenv."
        exit 1
    fi
    PYTHON_BIN="$VENV_PATH/bin/python"
    PIP_BIN="$VENV_PATH/bin/pip"

    echo "  venv      : $VENV_PATH"
    echo "  Installing requirements …"
    "$PIP_BIN" install --quiet --upgrade pip
    "$PIP_BIN" install --quiet -r "$REQUIREMENTS"

    _run_python() { "$PYTHON_BIN" "$@"; }
fi

_green "  Dependencies installed."

# ---------------------------------------------------------------------------
# Step 4 — Create .env from .env.example
# ---------------------------------------------------------------------------
_step "Step 4/6 — Configuration"

if [[ -f "$ENV_FILE" ]]; then
    _yellow "  .env already exists — not overwriting."
    echo   "  Edit manually if needed: $ENV_FILE"
else
    if [[ ! -f "$ENV_EXAMPLE" ]]; then
        _red "  ERROR: .env.example not found: $ENV_EXAMPLE"
        exit 1
    fi

    cp "$ENV_EXAMPLE" "$ENV_FILE"
    _green "  Created: $ENV_FILE"

    # ── Import secrets from hippocampal/.env ────────────────────────────
    HIPPO_ENV="$UBIK_ROOT/hippocampal/.env"
    if [[ -f "$HIPPO_ENV" ]]; then
        echo "  Importing secrets from: $HIPPO_ENV"

        # Extract value: strip inline comments and surrounding whitespace.
        _extract() {
            grep -E "^$1=" "$2" 2>/dev/null \
                | head -1 \
                | cut -d= -f2- \
                | sed 's/[[:space:]]*#.*//' \
                | xargs 2>/dev/null \
                || true
        }

        NEO4J_PW="$(_extract NEO4J_PASSWORD "$HIPPO_ENV")"
        CHROMA_TK="$(_extract CHROMADB_TOKEN "$HIPPO_ENV")"

        if [[ -n "$NEO4J_PW" ]]; then
            _sed_i "s|^NEO4J_PASSWORD=.*|NEO4J_PASSWORD=$NEO4J_PW|" "$ENV_FILE"
            echo "    NEO4J_PASSWORD  ✓"
        fi

        if [[ -n "$CHROMA_TK" ]]; then
            _sed_i "s|^CHROMADB_TOKEN=.*|CHROMADB_TOKEN=$CHROMA_TK|" "$ENV_FILE"
            echo "    CHROMADB_TOKEN  ✓"
        fi

        if [[ -z "$NEO4J_PW" && -z "$CHROMA_TK" ]]; then
            _yellow "  No secrets found in hippocampal/.env — set them manually."
        fi
    else
        _yellow "  hippocampal/.env not found."
        _yellow "  Set NEO4J_PASSWORD and CHROMADB_TOKEN in: $ENV_FILE"
    fi
fi

# ---------------------------------------------------------------------------
# Step 5 — Create log directory
# ---------------------------------------------------------------------------
_step "Step 5/6 — Log directory"

mkdir -p "$LOG_DIR"
_green "  Log dir: $LOG_DIR"

# ---------------------------------------------------------------------------
# Step 6 — Validation
# ---------------------------------------------------------------------------
_step "Step 6/6 — Validation"
echo "  Running: python -m maestro status"
echo ""

cd "$UBIK_ROOT"

STATUS_RC=0
_run_python -m maestro status || STATUS_RC=$?

echo ""

# ---------------------------------------------------------------------------
# Deployment summary
# ---------------------------------------------------------------------------
_sep
_bold " Deployment Summary"
_sep
echo ""
printf "  %-12s %s\n" "Node type"  "$NODE_TYPE"
printf "  %-12s %s\n" "UBIK root"  "$UBIK_ROOT"
printf "  %-12s %s\n" "Config"     "$ENV_FILE"
printf "  %-12s %s\n" "Logs"       "$LOG_DIR"
echo ""

if   [[ "$STATUS_RC" -eq 0 ]]; then
    _green "  Status : ALL SERVICES HEALTHY ✓"
elif [[ "$STATUS_RC" -eq 1 ]]; then
    _yellow " Status : Some services degraded — run 'python -m maestro health'"
else
    _yellow " Status : Services unreachable (expected on first deployment before"
    _yellow "          services are started — run 'python -m maestro start')"
fi

# ---------------------------------------------------------------------------
# Shell alias suggestion
# ---------------------------------------------------------------------------
echo ""
_sep
_bold " Shell alias (convenience — add to your shell RC file)"
_sep
echo ""

if [[ "$NODE_TYPE" == "hippocampal" ]]; then
    ALIAS_PYTHON="\"$VENV_PATH/bin/python\""
    SHELL_RC="$HOME/.zshrc"
else
    ALIAS_PYTHON="\"$VENV_PATH/bin/python\""
    SHELL_RC="$HOME/.bashrc"
fi

# Escape the UBIK_ROOT for embedding inside double-quoted string
UBIK_ESC="${UBIK_ROOT//\"/\\\"}"
ALIAS_CMD="alias maestro='cd \"$UBIK_ESC\" && $ALIAS_PYTHON -m maestro'"

echo "  # Add to $SHELL_RC:"
echo "  $ALIAS_CMD"
echo ""
echo "  # Apply immediately:"
echo "  source $SHELL_RC"

# ---------------------------------------------------------------------------
# Sync reminder
# ---------------------------------------------------------------------------
echo ""
_sep
_bold " Sync Strategy (keeping both nodes up to date)"
_sep
echo ""
echo "  Option A — git (recommended):"
echo "    git pull && bash maestro/setup_maestro.sh"
echo ""

if [[ "$NODE_TYPE" == "hippocampal" ]]; then
    REMOTE_IP="100.79.166.114"
    REMOTE_ROOT="/home/gasu/ubik"
    echo "  Option B — rsync to Somatic over Tailscale:"
    echo "    rsync -avz --delete \\"
    echo "      \"$UBIK_ROOT/maestro/\" \\"
    echo "      gasu@$REMOTE_IP:$REMOTE_ROOT/maestro/"
else
    REMOTE_IP="100.103.242.91"
    REMOTE_ROOT="/Volumes/990PRO 4T/UBIK"
    echo "  Option B — rsync to Hippocampal over Tailscale:"
    echo "    rsync -avz --delete \\"
    echo "      \"$UBIK_ROOT/maestro/\" \\"
    echo "      gasu@$REMOTE_IP:\"$REMOTE_ROOT/maestro/\""
fi

echo ""
echo "  Then on the remote node:"
echo "    bash maestro/setup_maestro.sh"
echo ""
_sep
echo ""
echo "  Next steps:"
echo "    python -m maestro health     # status + usage metrics"
echo "    python -m maestro start      # bring local services up"
echo "    python -m maestro dashboard  # live TUI dashboard"
echo "    python -m maestro --help     # full command reference"
echo ""

exit 0
