#!/usr/bin/env bash
set -euo pipefail

# Build a digest from the WORKING TREE without touching production. Two guarantees:
#   1. Output goes to digests_dev/, so the seen-paper history in digests/ is never
#      modified. A dev run that wrote there would mark papers as seen and silently
#      empty the next real digest's "Today's Feed".
#   2. send_digest_html.py is never invoked, so nothing can reach an inbox.
#
# Usage:  ./dev_run.sh [any lit_feed.py flag, e.g. --profile single_cell_ml]

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/dev0/zhouw}"
CONDA_BASE="${CONDA_BASE:-${DATA_ROOT}/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lit_feed}"

# Same cache/temp pinning as the cron wrapper: the system disk is small.
export TMPDIR="${TMPDIR:-${DATA_ROOT}/tmp}"
export TMP="${TMPDIR}"
mkdir -p "${TMPDIR}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${DATA_ROOT}/.cache/torch}"
export LIT_CROSSREF_EMAIL="${LIT_CROSSREF_EMAIL:-wenjiangz1123@gmail.com}"

export LIT_FEED_OUTPUT_DIR="${script_dir}/digests_dev"
mkdir -p "${LIT_FEED_OUTPUT_DIR}"

# Seed the dev history from production once, so the first dev run does not report
# a month of already-seen papers as brand new.
if [[ -z "$(ls -A "${LIT_FEED_OUTPUT_DIR}" 2>/dev/null)" ]]; then
  latest="$(ls -t "${script_dir}"/digests/digest_*.html 2>/dev/null | grep -v '\.email\.' | head -1 || true)"
  if [[ -n "${latest}" ]]; then
    cp "${latest}" "${LIT_FEED_OUTPUT_DIR}/"
    echo "seeded dev history from $(basename "${latest}")"
  fi
fi

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

echo "=== DEV RUN (working tree) -> ${LIT_FEED_OUTPUT_DIR} · no email ==="
python "${script_dir}/lit_feed.py" "$@"
echo "=== done. Open the newest file in digests_dev/ to review. ==="
