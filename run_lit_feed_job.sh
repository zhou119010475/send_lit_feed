#!/usr/bin/env bash
set -euo pipefail

# Configuration
DEFAULT_CONDA_BASE="$HOME/miniconda3"
DETECTED_CONDA_BASE="$(command -v conda >/dev/null 2>&1 && conda info --base 2>/dev/null || true)"
CONDA_BASE="${CONDA_BASE:-${DETECTED_CONDA_BASE:-$DEFAULT_CONDA_BASE}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lit_feed}"
REPO_ROOT="${REPO_ROOT:-/mnt/dev0/zhouw/send_lit_feed-1}"
DATA_ROOT="${DATA_ROOT:-/mnt/dev0/zhouw}"

# Keep temp files and model caches on the large data disk, not the small system
# disk. cron uses a non-interactive shell that does NOT source ~/.bashrc, so these
# must be set explicitly here.
export TMPDIR="${TMPDIR:-${DATA_ROOT}/tmp}"
export TMP="${TMPDIR}"
mkdir -p "${TMPDIR}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${DATA_ROOT}/.cache/torch}"

# LitFeed email configuration (hardcoded for portability)
export LIT_SMTP_HOST="smtp.gmail.com"
export LIT_SMTP_PORT="587"
export LIT_SMTP_USER="wenjiangz1123@gmail.com"
export LIT_SMTP_PASS="shgxnoabtuuabowx"
export LIT_FROM="<wenjiangz1123@gmail.com>"
export LIT_TO="wenjiang.zhou@ucsf.edu, peng.he@ucsf.edu, yuefei.zhu@ucsf.edu, Konstantinos.Stasinos@ucsf.edu, yujie.zhang@ucsf.edu"
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

