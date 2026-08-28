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

# Secrets live in .env beside this script, which .gitignore excludes so the Gmail
# App Password never reaches the public remote. lit_feed.py reads that file itself
# (upstream's _load_local_env), but send_digest_html.py is a separate process, so
# source it here to put LIT_SMTP_PASS in the environment for both.
# See .env.example for the expected keys.
ENV_FILE="${REPO_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    set +a
fi

# LitFeed email configuration (non-secret parts are fine to track)
export LIT_SMTP_HOST="smtp.gmail.com"
export LIT_SMTP_PORT="587"
export LIT_SMTP_USER="wenjiangz1123@gmail.com"
export LIT_FROM="<wenjiangz1123@gmail.com>"
export LIT_TO="wenjiang.zhou@ucsf.edu, peng.he@ucsf.edu, yuefei.zhu@ucsf.edu, Konstantinos.Stasinos@ucsf.edu, yujie.zhang@ucsf.edu"
export LIT_SMTP_STARTTLS="1"
export LIT_SUBJECT="[LITFeed] Recent Literature"

# Identifies us to Crossref (used for the preprints.org feed) so it places us in
# its faster, "polite" rate-limit pool instead of the anonymous one.
export LIT_CROSSREF_EMAIL="wenjiangz1123@gmail.com"

# Fail here rather than after a full 10-minute digest build. Gmail also revokes App
# Passwords periodically, so an empty value is a normal thing to hit, not a bug.
if [ -z "${LIT_SMTP_PASS:-}" ]; then
    echo "Error: LIT_SMTP_PASS is not set."
    echo "Create ${ENV_FILE} (see .env.example) with a Gmail App Password from"
    echo "https://myaccount.google.com/apppasswords -- 16 characters, no spaces."
    exit 1
fi

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

# lit_feed.py writes two files per run: an archive copy (digest_YYYY-MM-DD.html,
# carries the history blob future runs read back) and an email-shaped copy
# (digest_YYYY-MM-DD.email.html, sized to avoid Gmail's ~102KB clipping limit).
# Mail the latter, keyed off UTC to match how lit_feed.py names the file.
TODAY_UTC="$(date -u +%Y-%m-%d)"
EMAIL_HTML="digests/digest_${TODAY_UTC}.email.html"
if [ ! -f "${EMAIL_HTML}" ]; then
    echo "No email-ready digest found at ${REPO_ROOT}/${EMAIL_HTML}"
    exit 1
fi

python send_digest_html.py "${EMAIL_HTML}"

