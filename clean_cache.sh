#!/usr/bin/env bash
set -euo pipefail

# Periodic cache/temp cleanup for the lit_feed job.
#
# Removes stale model caches and leftover temp files from the data disk so they
# do not accumulate. Intended to run every ~2 months from cron. Safe to run any
# time: the HuggingFace cache is just re-downloaded on the next digest run.
#
# Tune with env vars:
#   DATA_ROOT   base data disk (default /mnt/dev0/zhouw)
#   AGE_DAYS    only delete temp entries older than this many days (default 60)

DATA_ROOT="${DATA_ROOT:-/mnt/dev0/zhouw}"
AGE_DAYS="${AGE_DAYS:-60}"

TMPDIR_PATH="${TMPDIR:-${DATA_ROOT}/tmp}"
HF_HOME="${HF_HOME:-${DATA_ROOT}/.cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-${DATA_ROOT}/.cache/torch}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting cache cleanup (AGE_DAYS=${AGE_DAYS})"

# 1) HuggingFace model cache: wipe entirely; it is re-downloaded on demand.
if [ -d "${HF_HOME}" ]; then
    echo "Removing HuggingFace cache: ${HF_HOME} ($(du -sh "${HF_HOME}" 2>/dev/null | cut -f1))"
    rm -rf "${HF_HOME}"
fi

# 2) Torch hub cache: same, safe to wipe.
if [ -d "${TORCH_HOME}" ]; then
    echo "Removing Torch cache: ${TORCH_HOME} ($(du -sh "${TORCH_HOME}" 2>/dev/null | cut -f1))"
    rm -rf "${TORCH_HOME}"
fi

# 3) Temp dir: only delete entries older than AGE_DAYS to avoid clobbering a
#    concurrent run. Delete files first, then now-empty directories.
if [ -d "${TMPDIR_PATH}" ]; then
    echo "Pruning temp files older than ${AGE_DAYS} days in: ${TMPDIR_PATH}"
    find "${TMPDIR_PATH}" -mindepth 1 -depth -type f -mtime "+${AGE_DAYS}" -delete 2>/dev/null || true
    find "${TMPDIR_PATH}" -mindepth 1 -depth -type d -empty -mtime "+${AGE_DAYS}" -delete 2>/dev/null || true
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cache cleanup done"
