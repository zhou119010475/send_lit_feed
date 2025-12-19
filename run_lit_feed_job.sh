#!/usr/bin/env bash
set -euo pipefail

# Configuration
DEFAULT_CONDA_BASE="$HOME/miniconda3"
DETECTED_CONDA_BASE="$(command -v conda >/dev/null 2>&1 && conda info --base 2>/dev/null || true)"
CONDA_BASE="${CONDA_BASE:-${DETECTED_CONDA_BASE:-$DEFAULT_CONDA_BASE}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lit_feed}"
REPO_ROOT="${REPO_ROOT:-/Users/wenjzhou/Documents/send_lit_feed}"

# LitFeed email configuration (hardcoded for portability)
export LIT_SMTP_HOST="smtp.example.com"
export LIT_SMTP_PORT="587"
export LIT_SMTP_USER="user@example.com"
export LIT_SMTP_PASS="app-token"
export LIT_FROM="Lit Feed <lit@example.com>"
export LIT_TO="you@example.com, teammate@example.com"
export LIT_SMTP_STARTTLS="1"
export LIT_SUBJECT="[LITFeed] Recent Literature"

# Activate conda
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "Error: conda initialization script not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
    exit 1
fi

conda activate "${CONDA_ENV_NAME}"

# Run the digest generator and sender
cd "${REPO_ROOT}"
python lit_feed.py

# Send the freshest digest that was just generated
LATEST_HTML="$(ls -1t digests/digest_*.html 2>/dev/null | head -n1 || true)"
if [ -z "${LATEST_HTML}" ]; then
    echo "No digest HTML found in ${REPO_ROOT}/digests"
    exit 1
fi

python send_digest_html.py "${LATEST_HTML}"

