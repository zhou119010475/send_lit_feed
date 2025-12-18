# Lit Feed Digest

This repo collects two helpers for generating and distributing a literature digest:

1. `lit_feed.py`: fetches RSS feeds (arXiv, bioRxiv, journals), filters them by keyword, ranks them by semantic similarity to canonical papers, and writes a markdown + HTML digest. It can also post top hits to Slack if you configure an incoming webhook.
2. `send_digest_html.py`: reads an existing HTML digest file and sends it via email using SMTP settings provided through environment variables.

## Requirements

- Python 3.11+ (any modern 3.x interpreter should work)
- Install dependencies with `pip install -r requirements.txt` (create this file as needed) or manually:

  ```bash
  pip install feedparser requests numpy sentence-transformers
  ```

## Installation (from original `zyj1729/lit_feed`)

```bash
git clone https://github.com/zyj1729/lit_feed.git
pip install feedparser requests sentence-transformers torch
```

This repository builds on the original `zyj1729/lit_feed` effort for the digest generation.

## Generating the Digest (`lit_feed.py`)

1. Set configuration at the top of the file if you want to tweak feeds, keywords, or output directories.
2. Optional environment variables:
   - `LIT_DIGEST_SLACK_WEBHOOK`: Slack incoming webhook URL used by `post_to_slack()` (leave empty to skip Slack).
3. Run the script to produce an HTML digest in `./digests/`:
   ```bash
   python lit_feed.py
   ```
   Output files are named `digest_YYYY-MM-DD.html` (and `*.md` when markdown is saved).

## Sending the Digest via Email (`send_digest_html.py`)

This script expects a digest HTML file path as an argument and reads SMTP settings from environment variables:

| Env Var | Description |
|---------|-------------|
| `LIT_SMTP_HOST` | SMTP server host (required) |
| `LIT_SMTP_PORT` | SMTP port (default: `587`) |
| `LIT_SMTP_USER` | Username (optional; required if SMTP auth needed) |
| `LIT_SMTP_PASS` | Password or app token (optional but needed with auth) |
| `LIT_FROM` | From address (required) |
| `LIT_TO` | Comma-separated recipients (required) |
| `LIT_SMTP_STARTTLS` | `1` (default) to use STARTTLS, `0` to disable |
| `LIT_SUBJECT` | Override email subject |

Example:
```bash
export LIT_SMTP_HOST="smtp.example.com"
export LIT_SMTP_PORT=587
export LIT_SMTP_USER="user@example.com"
export LIT_SMTP_PASS="app-token"
export LIT_FROM="Lit Feed <lit@example.com>"
export LIT_TO="you@example.com, teammate@example.com"

python send_digest_html.py digests/digest_2025-12-18.html
```

