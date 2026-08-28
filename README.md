# Lit Feed Digest

A weekly literature digest, mailed by cron. Fork of [`zyj1729/lit_feed`](https://github.com/zyj1729/lit_feed)
with local feeds, keywords, seed papers, and mail delivery.

| File | Role |
|---|---|
| `lit_feed.py` | Fetches RSS/Crossref feeds, filters, ranks by similarity to seed papers, writes `digests/digest_YYYY-MM-DD.html` (archive, carries the seen-paper history) and `digest_YYYY-MM-DD.email.html` (the copy that gets mailed). |
| `send_digest_html.py` | Mails an HTML digest over SMTP. |
| `run_lit_feed_job.sh` | What cron runs: builds the digest, then mails it. |
| `dev_run.sh` | Builds a digest from the working tree into `digests_dev/`, with no email. Use this to test changes. |

## Requirements

Python 3.11+, in the `lit_feed` conda env:

```bash
pip install feedparser requests numpy sentence-transformers torch
```

## Credentials

The Gmail App Password lives in `.env`, which `.gitignore` excludes so it never
reaches the public remote. `run_lit_feed_job.sh` sources it; `lit_feed.py` also
reads it directly.

```bash
cp .env.example .env && chmod 600 .env
# then put the 16-character App Password in LIT_SMTP_PASS
```

Get one at <https://myaccount.google.com/apppasswords> (the option only appears
once 2-Step Verification is on). **Google revokes these periodically** — when
`cron.log` shows `535 ... BadCredentials`, the digest is still building fine and
only the send step is broken. Generate a new password, put it in `.env`, done.

Non-secret mail settings (`LIT_SMTP_HOST`, `LIT_SMTP_USER`, `LIT_FROM`, `LIT_TO`,
`LIT_SUBJECT`, `LIT_SMTP_STARTTLS`) stay in `run_lit_feed_job.sh`.

## Testing a change

Never run `lit_feed.py` directly to test: it writes into `digests/`, which marks
papers as seen and silently empties the next real digest's "Today's Feed".

```bash
./dev_run.sh          # -> digests_dev/, no email
```

To test mail delivery without spamming the recipient list, override `LIT_TO` with
your own address for the one run.

## How a paper gets in

Four stages, in order:

1. **Exclusions** — `EXCLUDE_KEYWORDS`, applied at fetch time.
2. **Hard floor** — similarity to the closest seed group must reach `TODAY_MIN_SCORE`.
3. **Domain gate** — the title or abstract must contain a `DOMAIN_KEYWORDS` term.
   This is what rejects ordinary machine-learning papers (EHR transformers, remote
   sensing, battery models) that match a keyword like "foundation model" by coincidence.
4. **Admission** — an `INCLUDE_KEYWORDS` match, *or* a score of at least
   `SEMANTIC_ADMIT_SCORE` for work that is on topic but does not use our vocabulary.

## Local divergence from upstream

Upstream tunes for a mostly-cardiovascular, model-development audience. This fork
also follows organ and lineage development, so four settings differ. Each was
measured against a real digest (`digests/digest_2026-08-25.html`, 465 papers)
rather than guessed:

- **Extra feeds** — five more bioRxiv subject feeds (cancer, cell, developmental
  biology, genetics, genomics).
- **`INCLUDE_KEYWORDS`** — organ and lineage terms, plus a
  `lung, limb & hematopoietic development` seed group.
- **`DOMAIN_KEYWORDS`** — extended to match. This gate only ever *removes* papers,
  so any include keyword missing from it is silently cancelled. Upstream's list is
  phrased around assay names and dropped genuine work that names the biology
  instead (enhancer–promoter hubs, cortical organoids, a "cellular atlas" that
  never says "cell atlas"). With our additions the gate still drops 38% of the
  pool — the generic-ML noise it exists for.
- **`TODAY_MIN_SCORE = 0.30`** — upstream ships 0.465, which cuts 65% of what this
  feed carries. The upgrade to `max_seq_length=512` moved scores by a median of
  only −0.005, so 0.30 still means what it meant before.

`EMAIL_ABSTRACT_CHARS` is also restored: upstream now mails full abstracts, which
put the message at 130KB, past Gmail's ~102KB clipping limit. Truncated, it is 82KB.
