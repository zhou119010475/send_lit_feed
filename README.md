# Lit Feed Digest — erythroid / transgene edition (`send_lit_feed-2`)

A literature digest mailed by cron every 2–3 days (Mon/Wed/Fri). Fork of
[`zyj1729/lit_feed`](https://github.com/zyj1729/lit_feed) with local feeds, keywords,
seed papers, and mail delivery.

This directory started as a copy of `send_lit_feed-1`, the weekly group digest, and
keeps everything that one carries. On top of it, it follows the erythroid-atlas
project: hematology journals, erythropoiesis and globin regulation, developmental
hematopoiesis, and transgene / vector integration mapping. It mails one inbox, under
the subject `[LITFeed-Blood]`. See [What this edition adds](#what-this-edition-adds).

| File | Role |
|---|---|
| `lit_feed.py` | Fetches RSS/Crossref/PubMed feeds, filters, ranks by similarity to seed papers, writes `digests/digest_YYYY-MM-DD.html` (archive, carries the seen-paper history) and `digest_YYYY-MM-DD.email.html` (the copy that gets mailed). |
| `send_digest_html.py` | Mails an HTML digest over SMTP. |
| `run_lit_feed_job.sh` | What cron runs: builds the digest, then mails it. |
| `dev_run.sh` | Builds a digest from the working tree into `digests_dev/`, with no email. Use this to test changes. |

## Schedule

```cron
40 1 * * 1,3,5 /mnt/dev0/zhouw/send_lit_feed-2/run_lit_feed_job.sh >> /mnt/dev0/zhouw/send_lit_feed-2/cron.log 2>&1
```

Monday, Wednesday and Friday at 01:40 UTC, so gaps of 2, 2 and 3 days. Weekdays
rather than `*/2` or `*/3` in the day-of-month field, because those restart on the
1st and give uneven gaps at every month end. For a strict two-day rhythm use
`1,3,5,0`; for twice a week, `1,4`. `send_lit_feed-1` runs on Tuesdays, so the two
never build at the same time.

`run_lit_feed_job.sh` finds the repository from its own location, so the directory
can be renamed or copied again without editing it.

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
`LIT_SUBJECT`, `LIT_SMTP_STARTTLS`) stay in `run_lit_feed_job.sh`. `LIT_TO` is a
single address there; add more, comma-separated, to share this edition.

## Testing a change

Never run `lit_feed.py` directly to test: it writes into `digests/`, which marks
papers as seen and silently empties the next real digest's "Today's Feed".

```bash
./dev_run.sh          # -> digests_dev/, no email
```

`dev_run.sh` seeds its history from the newest file in `digests/`, and the history
is "the newest digest that is not today's". So a dev run on a day the cron job has
already run sees no history and reports almost nothing as new; test the day after,
or copy that digest into an empty output directory under yesterday's date and point
`LIT_FEED_OUTPUT_DIR` at it.

To test mail delivery to a different inbox, set `LIT_TO` for the one run:
`LIT_TO=me@example.org ./run_lit_feed_job.sh`.

## How a paper gets in

Four stages, in order:

1. **Exclusions** — `EXCLUDE_KEYWORDS`, applied at fetch time, matched at the start
   of a word (so `plant` does not veto "transplant").
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

## What this edition adds

Everything below is in `lit_feed.py`, marked `LOCAL (send_lit_feed-2)`.

- **PubMed feeds** (`"type": "pubmed"`, any PubMed query in `"term"`). Blood and
  Blood Advances have no usable RSS — every ashpublications.org endpoint answers
  403/404, which is why they were dropped earlier — but PubMed carries them,
  ahead-of-print included, with full abstracts. Six queries:
  - journals: Blood, Blood Advances, Haematologica, HemaSphere, Experimental
    Hematology · Cell Stem Cell, Stem Cell Reports, Developmental Cell, Nature Cell
    Biology, Development · Genome Biology, Nucleic Acids Research;
  - topics, across every journal: erythropoiesis · developmental hematopoiesis ·
    transgene mapping.

  Add a journal by appending `OR "Abbrev"[jour]` to a term, using the abbreviation
  PubMed shows. A misspelt one does not fail — PubMed drops the clause — so the
  fetcher prints `! PubMed query warning` in `cron.log` when that happens.
- **Keywords** — `BLOOD_KEYWORDS` and `TRANSGENE_KEYWORDS`, spliced into both
  `INCLUDE_KEYWORDS` and `DOMAIN_KEYWORDS` so the two cannot drift apart. Bare
  `blood` and `hemoglobin` are left out on purpose: they match blood pressure,
  HbA1c and every clinical cohort. Edit these two lists to retune the project side.
- **Seed groups** — `erythropoiesis & globin regulation` (6 papers),
  `developmental hematopoiesis` (4), `transgene & vector integration mapping` (5).
  This is what makes project papers *rank*; keywords only admit.
- **Tags** — `erythroid`, `hematopoiesis`, `transgene` chips, listed first.
- **Filters fixed for hematology**
  - Exclusions match at word start. As a substring, `plant` vetoed every
    transplant paper: 53 of the 640 in the first PubMed pool.
  - The `mouse model` veto is gone (8 otherwise-admitted papers a month).
  - `KEYWORD_FALSE_FRIENDS` blanks NRF2's full name ("nuclear factor erythroid
    2-related factor 2") before matching, so oxidative-stress papers cannot enter
    on the word "erythroid". The PubMed erythropoiesis query excludes it too, which
    took that query from 262 results to 122.
- **De-duplication by title** — PubMed and an RSS feed reach the same paper through
  different links, which the link-based key sees as two papers.
- **Cardiovascular removed** — upstream's `cardiovascular single-cell` seed group,
  the `cardiomyopathy` include keywords, the `cardiac` tag and the cardiac
  `DOMAIN_KEYWORDS` terms are all gone. A heart paper can still enter on a general
  term such as "single-cell", but nothing ranks it up any more.

Measured on 2026-10-06, before the cardiovascular removal (which took no slot in
that run), against the history copied from `send_lit_feed-1`: 3460
papers fetched, 272 newly admitted, and of the 40 shown in Today's Feed, 23 came
from the three project seed groups and 17 from the original ones. The mailed copy
was 86KB, under Gmail's ~102KB clip.
