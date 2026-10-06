#!/usr/bin/env python3
"""
Recent literature digest for arXiv / bioRxiv / journals.

- Fetches RSS feeds (and Crossref or PubMed, for servers that block scripted clients)
- Applies profile-specific domain and keyword filters
- Shortlists by seed-paper similarity and optionally judges with an LLM
- Writes Markdown digest
- Optionally posts top-N to Slack via incoming webhook

Configurable in the CONFIG section below.
"""

import argparse
import glob
import json
import math
import os
import re
import textwrap
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from html import escape
from typing import List, Dict, Any, Optional
from urllib.parse import urlsplit, urlunsplit

import feedparser
import requests
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util


def _load_local_env() -> None:
    """Load simple KEY=VALUE settings from the ignored .env beside this script."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            key, value = key.strip(), value.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


_load_local_env()


# ==========================
# ======== CONFIG ==========
# ==========================

# How many days back to look
LOOKBACK_DAYS = 30

# Max items to consider per feed before filtering/ranking.
# A feed entry may raise this for itself with "max_items" (see preprints.org).
MAX_ITEMS_PER_FEED = 200

# Crossref asks API clients to identify themselves, and rewards those that do
# with a faster rate-limit pool. Set this to your email to opt in.
CROSSREF_API = "https://api.crossref.org/works"
CROSSREF_CONTACT = os.getenv("LIT_CROSSREF_EMAIL", "")
# Crossref rate-limits deep paging. Keep pages modest and pause between them.
CROSSREF_ROWS = 500
CROSSREF_PAUSE_SEC = 1.0
CROSSREF_RETRIES = 4

# LOCAL: PubMed E-utilities, for journals whose RSS is closed to scripts (every
# ashpublications.org endpoint -- Blood, Blood Advances -- answers 403/404) and for
# topic searches that should reach a paper whatever journal it lands in. NCBI asks
# clients to identify themselves and allows 3 requests/second without an API key,
# 10 with one (NCBI_API_KEY in .env, optional).
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBMED_CONTACT = os.getenv("LIT_PUBMED_EMAIL", CROSSREF_CONTACT)
PUBMED_API_KEY = os.getenv("NCBI_API_KEY", "").strip()
PUBMED_BATCH = 200
PUBMED_PAUSE_SEC = 0.4
PUBMED_RETRIES = 4

# A new paper enters the high-recall shortlist if it matches an include keyword or
# reaches this seed-similarity score. A separate 0.40 hard floor applies first. The
# biological-domain gate removes generic keyword coincidences, and the optional LLM
# judge then applies the profile's compound scientific intent. Set this to 0 to use
# keywords as the only route into that shortlist.
# Re-derived 2026-08-26 on 745 labelled papers at max_seq_length=512, fitted on
# the training split for precision >= 0.90: held-out precision 0.964, recall 0.762.
# Governs admission WITHOUT an include-keyword match, so keyword hits still
# recover papers this rejects.
SEMANTIC_ADMIT_SCORE = 0.62

# Abstract length in the emailed copy only. Cards end up within a line of each other
# at this width, which is what keeps the message scannable. The archive copy in
# digests/ keeps full abstracts -- it is read in a browser, where length is free.
#
# LOCAL: upstream removed this in ac4de18 and now emails full abstracts, accepting
# that "very long messages may be clipped by the mail client". We cannot: Gmail
# clips at ~102KB and shows a "[Message clipped]" stub, which is the exact failure
# the previous sync introduced this constant to fix. Measured on the 2026-08-27 dev
# run, 40 cards with full abstracts came to 130KB (60KB of abstract, median 1548
# characters each) against 77KB truncated. Restored deliberately.
EMAIL_ABSTRACT_CHARS = 300

# Number of top papers to include in the digest
TOP_K = 60

TODAY_TOP_K = 40
PREV_TOP_K = 30
# Hard relevance floor for every paper in the digest. Keyword matches do not bypass
# it; papers above this floor still need a keyword match or SEMANTIC_ADMIT_SCORE.
# Re-derived at max_seq_length=512: retains 99% of labelled wanted papers
# (loses 3/253) while rejecting 72/125 unwanted, against 49/125 at the old 0.40.
#
# LOCAL: upstream ships 0.465; we stay at 0.30. Two things were checked before
# keeping it, by re-scoring the 465 papers stored in digest_2026-08-25.html --
# written by the old code, so every paper has a paired old and new score.
#
#   1. Is 0.30 still the same bar? Yes. max_seq_length 256 -> 512 plus upstream's
#      reshuffled seed groups moved scores by a median of -0.005 (mean -0.012),
#      and papers that had scored just above the old floor re-score at a median
#      0.314. The distribution barely moved, so 0.30 keeps meaning what it meant.
#   2. Would upstream's 0.465 work for us? No. It cuts 65% of what this feed
#      currently carries, because our INCLUDE_KEYWORDS reach into organ and
#      lineage development (lung/kidney/limb/hematopoietic/atlas), which sits
#      further from upstream's cardiovascular- and foundation-model-heavy seeds
#      than the work their threshold was fitted on.
#
# The generic-ML noise upstream raised the floor to remove is removed here by the
# domain gate instead -- see DOMAIN_KEYWORDS, which drops 38% of this same pool.
TODAY_MIN_SCORE = 0.30

# Number of top papers to optionally post to Slack. Upstream dropped this constant
# in ac4de18 but left post_to_slack() referencing it; keep it so the Slack path is
# a no-op rather than a NameError if LIT_DIGEST_SLACK_WEBHOOK is ever set.
TOP_K_SLACK = 15

# Keep output beside this script so cron and interactive runs use the same history.
# The default profile continues to use this directory directly. Additional profiles
# get their own subdirectory, preventing one audience's seen-paper history from
# suppressing another audience's results.
OUTPUT_ROOT = os.getenv(
    "LIT_FEED_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "digests"),
)
OUTPUT_DIR = OUTPUT_ROOT

# Optional Slack incoming webhook URL (set as env var or paste string here)
SLACK_WEBHOOK_URL = os.getenv("LIT_DIGEST_SLACK_WEBHOOK", "").strip()

# ---- Feeds ----
# Add / edit as you like. URLs are RSS / Atom endpoints.

FEEDS = [
    # arXiv (examples)
    {
        "name": "arXiv q-bio",
        "url": "https://export.arxiv.org/rss/q-bio",
    },
    {
        "name": "arXiv cs.LG",
        "url": "https://export.arxiv.org/rss/cs.LG",
    },

    # bioRxiv – you should adjust to your preferred subject feeds.
    # Check biorxiv "RSS" page for more granular subjects if you want.
    {
        "name": "bioRxiv Genomics+Bioinformatics",
        "url": "https://connect.biorxiv.org/biorxiv_xml.php?subject=genomics+bioinformatics",
    },
    {
        "name": "bioRxiv Cancer Biology",
        "url": "http://connect.biorxiv.org/biorxiv_xml.php?subject=cancer_biology",
    },
    {
        "name": "bioRxiv Cell Biology",
        "url": "http://connect.biorxiv.org/biorxiv_xml.php?subject=cell_biology",
    },
    {
        "name": "bioRxiv Developmental Biology",
        "url": "http://connect.biorxiv.org/biorxiv_xml.php?subject=developmental_biology",
    },
    {
        "name": "bioRxiv Genetics",
        "url": "http://connect.biorxiv.org/biorxiv_xml.php?subject=genetics",
    },
    {
        "name": "bioRxiv Genomics",
        "url": "http://connect.biorxiv.org/biorxiv_xml.php?subject=genomics",
    },

    # Journals – many have RSS links in their “Alerts” or “RSS” pages.
    # Put whatever you care about here.
    {
        # Genome Research serves a duplicate XML attribute that feedparser's own
        # fetcher rejects outright, and its recent.xml and ahead_of_print.xml feeds
        # have the identical defect. The requests fallback in _parse_rss recovers it.
        "name": "Genome Research",
        "url": "https://genome.cshlp.org/rss/current.xml",
    },
    {
        "name": "Nature Genetics",
        "url": "https://www.nature.com/ng.rss",
    },
    {
        "name": "Nature Methods",
        "url": "https://www.nature.com/nmeth.rss",
    },
    {
        # The /current_issue/rss form serves malformed XML (invalid token at line 15)
        # and yields zero entries; the canonical per-journal .rss feed is clean.
        "name": "Nature Biotechnology",
        "url": "https://www.nature.com/nbt.rss",
    },
    {
        "name": "Nature",
        "url": "https://www.nature.com/nature.rss",
    },

    {
        # science.org rather than the legacy sciencemag.org host: the content is
        # identical (same 30 items, same summaries) but sciencemag.org answers 403 to
        # requests, so the _parse_rss fallback could never rescue it if feedparser's
        # own fetch failed. This host answers both, so the feed has a safety net.
        #
        # Not the science.org table-of-contents feed, though: that one carries author
        # names but its summaries are boilerplate ("Science, Volume 393, Issue 6812"),
        # and the ranker needs the abstract more than the digest needs an author line.
        "name": "Science",
        "url": "https://www.science.org/rss/current.xml",
    },

    {
        "name": "Cell",
        "url": "https://www.cell.com/cell/current.rss",
    },

    {
        "name": "Bioinformatics",
        "url": "https://academic.oup.com/rss/site_5139/3001.xml",
    },
    {
        "name": "PLOS Computational Biology",
        "url": "https://journals.plos.org/ploscompbiol/feed/atom",
    },

    # preprints.org (MDPI Preprints). Read through Crossref rather than RSS:
    # every preprints.org RSS endpoint answers 403 to scripted clients, while
    # Crossref indexes the same records -- title, abstract, posting date -- with
    # no bot wall and lets us ask for a date range instead of the latest N.
    # It is multidisciplinary and high volume (~100 posts/day across all fields),
    # so the keyword filters are what make it useful, and it gets a larger item
    # budget than a journal feed needs.
    {
        "name": "preprints.org",
        "url": "https://www.preprints.org/",
        "type": "crossref",
        "prefix": "10.20944",
        "max_items": 3000,
    },

    # LOCAL: PubMed feeds (see fetch_pubmed). "term" is any PubMed query; the
    # lookback window is added for you. Each paper is labelled with its own journal,
    # not with the feed name, so a topic search reads the same as a journal feed.
    # These come last on purpose: when PubMed and an RSS feed above carry the same
    # paper, the title de-duplication in main() keeps the first copy fetched.
    {
        # Blood and Blood Advances were dropped in 4783e61 because their RSS never
        # yielded an item. PubMed carries both, ahead-of-print included.
        "name": "PubMed: hematology journals",
        "type": "pubmed",
        "term": '"Blood"[jour] OR "Blood Adv"[jour] OR "Haematologica"[jour] '
                'OR "Hemasphere"[jour] OR "Exp Hematol"[jour]',
        "max_items": 400,
    },
    {
        "name": "PubMed: stem cell & development journals",
        "type": "pubmed",
        "term": '"Cell Stem Cell"[jour] OR "Stem Cell Reports"[jour] OR "Dev Cell"[jour] '
                'OR "Nat Cell Biol"[jour] OR "Development"[jour]',
        "max_items": 400,
    },
    {
        "name": "PubMed: genomics journals",
        "type": "pubmed",
        "term": '"Genome Biol"[jour] OR "Nucleic Acids Res"[jour]',
        "max_items": 400,
    },
    {
        "name": "PubMed: erythropoiesis",
        "type": "pubmed",
        "term": '(erythropoiesis[tiab] OR erythroid[tiab] OR erythroblast*[tiab] '
                'OR "fetal hemoglobin"[tiab] OR "fetal haemoglobin"[tiab] '
                'OR "hemoglobin switching"[tiab] OR "globin switching"[tiab] '
                'OR "globin gene"[tiab] OR "globin genes"[tiab]) '
                # NRF2 spells out as "nuclear factor erythroid 2-related factor 2".
                'NOT (NRF2[tiab] OR NFE2L2[tiab] OR "erythroid 2"[tiab])',
        "max_items": 600,
    },
    {
        # Hematopoiesis alone is mostly clinical transplant and leukemia work, so
        # this one is narrowed to development and regulation.
        "name": "PubMed: developmental hematopoiesis",
        "type": "pubmed",
        "term": '(hematopoiesis[tiab] OR haematopoiesis[tiab] '
                'OR "hematopoietic stem"[tiab] OR "haematopoietic stem"[tiab] '
                'OR "hematopoietic progenitor"[tiab] OR "hematopoietic progenitors"[tiab]) '
                'AND ("single-cell"[tiab] OR "single cell"[tiab] OR fetal[tiab] '
                'OR embryonic[tiab] OR "yolk sac"[tiab] OR "fetal liver"[tiab] '
                'OR "gene regulatory"[tiab] OR chromatin[tiab] '
                'OR "transcription factor"[tiab] OR "transcription factors"[tiab])',
        "max_items": 400,
    },
    {
        # Two halves: locating a transgene / vector in the genome, and detecting or
        # quantifying one in sequencing reads.
        "name": "PubMed: transgene mapping",
        "type": "pubmed",
        "term": '"transgene integration"[tiab] OR "transgene insertion"[tiab] '
                'OR "transgene mapping"[tiab] OR "transgene copy number"[tiab] '
                'OR "targeted locus amplification"[tiab] '
                # Vectors only. Bare retrovir*/provir*/transpos* here pulled in HIV and
                # HTLV integration virology and bacterial insertion sequences.
                'OR ((transgene*[tiab] OR lentivir*[tiab] OR "retroviral vector"[tiab] '
                'OR "retroviral vectors"[tiab] OR "viral vector"[tiab] '
                'OR "viral vectors"[tiab] OR AAV[tiab] OR "CAR T"[tiab] OR "CAR-T"[tiab]) '
                'AND ("integration site"[tiab] OR "integration sites"[tiab] '
                'OR "insertion site"[tiab] OR "insertion sites"[tiab])) '
                'OR (transgene*[tiab] AND ("RNA-seq"[tiab] OR "single-cell"[tiab] '
                'OR "single cell"[tiab] OR "long-read"[tiab] OR nanopore[tiab] '
                'OR "whole-genome sequencing"[tiab]))',
        "max_items": 300,
    },
]

# ---- Keyword filters ----
# If INCLUDE_KEYWORDS is non-empty, keep items that match at least one of them in
# title or summary. Matching is case-insensitive simple substring.

# Matching is plain case-insensitive substring, which is unforgiving about morphology:
# "multiomics" does NOT match "multiomic data", and "cardiomyopathy" does NOT match
# "cardiomyopathies". A single character silently drops a paper -- that is how a Genome
# Research paper on spatial multiomic integration was missed. So each concept lists its
# spellings, singular/plural forms, and the assay names that never spell the concept out.
#
# Every variant below was measured against a real day's pool of 1113 papers. Most add
# nothing today, because a paper using one term nearly always uses another that already
# matches; they are here as insurance against the day it does not. Counts in comments
# are papers each term newly admitted that no other keyword caught.

# LOCAL (send_lit_feed-2): the erythroid-atlas project. These two lists are spliced
# into BOTH INCLUDE_KEYWORDS and DOMAIN_KEYWORDS below. The domain gate only ever
# removes papers, so a term that admits but is missing from the gate does nothing;
# sharing one list makes that mistake impossible.
#
# Deliberately absent: bare "blood" and bare "hemoglobin". The first matches blood
# pressure, blood glucose and every clinical cohort; the second matches HbA1c and
# "hemoglobin level" as a trial endpoint. The journal Blood is covered as a FEED
# instead, and its papers still have to earn a place on these terms or on similarity.
BLOOD_KEYWORDS = [
    # erythroid lineage. Spelled out rather than a bare "erythro", which would also
    # match erythromycin and erythroderma.
    "erythroid",
    "erythropoie",        # erythropoiesis, erythropoietic, erythropoietin
    "erythroblast",
    "erythrocyte",
    "red blood cell",
    " red cell",          # leading space: "cultured cells" and "engineered cells" contain "red cell"
    "reticulocyte",
    "enucleation",
    # globin regulation / hemoglobin switching
    "fetal hemoglobin",
    "fetal haemoglobin",
    "hemoglobin switch",
    "haemoglobin switch",
    "globin gene",
    "globin switch",
    "globin locus",
    "beta-globin", "β-globin",
    "gamma-globin", "γ-globin",
    "embryonic globin",
    # hematopoiesis, both spellings ("hematopoietic"/"hematopoiesis" above are US only)
    "hematopoie",
    "haematopoie",
    "hemogenic",
    "haemogenic",
    "HSPC",
    "megakaryo",
    "yolk sac",
    "fetal liver",
    "bone marrow",
    # the erythroid transcription-factor core, and the two being overexpressed
    "GATA1", "GATA2", "GATA5", "LMO2", "TAL1", "KLF1", "BCL11A",
]

TRANSGENE_KEYWORDS = [
    "transgene",          # transgenes, transgene mapping / integration / insertion
    "integration site",   # substring also covers "integration sites"
    "integration-site",
    "insertion site",
    "insertion-site",
    "vector integration",
    "targeted locus amplification",
    "vector copy number",
    "lentiviral vector",
    "insertional mutagenesis",
]

# Phrases that contain one of our keywords but mean something else. They are blanked
# out of the text before any include / domain / tag match, which is gentler than an
# EXCLUDE_KEYWORDS veto: the paper is not rejected, it just cannot get in on this
# phrase alone. Today that is NRF2, whose full name -- "nuclear factor erythroid
# 2-related factor 2", or "(erythroid-derived 2)-like 2" -- puts "erythroid" into
# every oxidative-stress paper; they were 15 of the 21 lowest-scoring matches in the
# first PubMed pool.
KEYWORD_FALSE_FRIENDS = re.compile(
    r"erythroid[\s\-\u2010-\u2015]*(?:derived[\s\-\u2010-\u2015]*)?2", re.IGNORECASE
)

INCLUDE_KEYWORDS = [
    # single-cell, and the single-nucleus assays the cardiac literature actually uses
    "single-cell",
    "single cell",
    "scRNA-seq",
    "scrna",              # also catches scRNAseq, scRNA seq
    "snRNA-seq",
    "single-nucleus",
    "single nucleus",

    # perturbation
    "perturbation",       # substring also covers perturbations, perturbational
    "perturb-seq",        # assay name; contains no "perturbation"

    # multi-omics
    "multi-omic",         # substring also covers multi-omics
    "multi omic",
    "multiomics",
    "multiomic",          # +2 papers; "multiomics" cannot match "multiomic data"
    "multiome",           # 10x Multiome
    "CITE-seq",
    "phosphoproteomic",

    # spatial
    "spatial transcriptomics",
    "spatial transcriptomic",   # singular

    # chromatin
    "chromatin accessibility",
    "open chromatin",
    "accessible chromatin",
    "ATAC-seq",           # substring also covers scATAC-seq, snATAC-seq
    "ATACseq",            # unhyphenated

    # gene regulation
    "gene regulatory",
    "gene regulation",
    "GRN",

    # models
    "foundation model",   # substring also covers foundation models
    "transformer",
    "deep learning",
    "deep-learning",
    "neural network",
    "virtual cell",
    "virtual-cell",

    # organ / lineage development (local additions)
    "lung",
    "kidney",
    "limb",
    "hematopoietic",
    "hematopoiesis",
    "CPAM",
    "organoid",
    "neuroendocrine",
    "multimodal",
    "bioinformatics",
    "computational biology",
    "genomics",
    "developmental biology",
    "atlas",

    # erythroid-atlas project (send_lit_feed-2 additions)
    *BLOOD_KEYWORDS,
    *TRANSGENE_KEYWORDS,
]

# If any of these appear, drop the item.
EXCLUDE_KEYWORDS = [
    "plant",
    "fungus",
    "yeast",
    "microbiome",
    "bacterial community",
    "ecology",
    "behavioral",
    # LOCAL (send_lit_feed-2): upstream's "mouse model" veto is dropped here.
    # "In a mouse model of beta-thalassemia" is routine wording in hematology, and
    # PubMed's full abstracts trip it far more often than an RSS teaser did. Measured
    # on a 3530-paper pool it cost 8 otherwise-admitted papers a month.
]

# ---- Topic tags ----
# Coloured chips shown on each paper in the digest. Purely presentational: the
# include/exclude lists above still decide what gets in, this only labels what
# already passed. Keywords overlap INCLUDE_KEYWORDS on purpose -- keeping the two
# lists separate is what lets you admit a paper without also tagging it.
#
# Colours are Okabe-Ito, chosen to stay distinguishable with colour vision
# deficiency. Three are darkened from stock (perturbation, multi-omics, spatial) so
# that white 10px chip text clears WCAG AA 4.5:1 rather than only AA-large -- do not
# "restore" them to the published hexes without re-checking contrast. Matching is
# case-insensitive substring on title + abstract, same as the filters.
TAG_MAX = 3  # chips per paper, so a card cannot sprout a wall of labels

TAGS = [
    # LOCAL (send_lit_feed-2): project chips first, so TAG_MAX never crowds them out.
    # Yellow is the one stock Okabe-Ito colour the list had not used; teal and brown
    # are outside the palette but clear AA with white text (5.1:1 and 6.8:1).
    ("erythroid",        "#F0E442", ["erythroid", "erythropoie", "erythroblast", "erythrocyte",
                                     "red blood cell", " red cell", "reticulocyte",
                                     "fetal hemoglobin", "fetal haemoglobin",
                                     "globin gene", "globin switch", "globin locus",
                                     "beta-globin", "β-globin", "gamma-globin", "γ-globin"]),
    ("hematopoiesis",    "#00798C", ["hematopoie", "haematopoie", "hemogenic", "haemogenic",
                                     "HSPC", "megakaryo", "yolk sac", "fetal liver",
                                     "bone marrow"]),
    ("transgene",        "#7A5230", TRANSGENE_KEYWORDS),
    ("single-cell",      "#0072B2", ["single-cell", "single cell", "scRNA-seq", "scrna"]),
    ("perturbation",     "#C05500", ["perturbation"]),
    ("multi-omics",      "#008561", ["multi-omic", "multi omic", "multiomics",
                                     "phosphoproteomic"]),
    ("spatial",          "#A36186", ["spatial transcriptomics"]),
    ("chromatin",        "#E69F00", ["chromatin accessibility", "ATAC-seq"]),
    ("gene regulation",  "#56B4E9", ["gene regulatory"]),
    ("foundation model", "#3B3B3B", ["foundation model", "transformer"]),
    ("virtual cell",     "#6F4C9B", ["virtual cell", "virtual-cell"]),
]

# ---- Canonical papers ----
# These seed the semantic similarity - think of them as "prototypes" for what you
# care about. Title plus a real abstract works much better than title plus a single
# sentence; the ranker only knows what these say.
#
# "group" sorts a seed into a research interest. Seeds in a group are averaged into
# one direction, and a paper is scored against the CLOSEST group. Both halves matter:
#
#   * averaging inside a group cancels the quirks of any one paper (its dataset
#     names, its method name) and keeps what the group has in common, so a single
#     oddly-written seed cannot skew that interest;
#   * taking the max ACROSS groups means a paper close to any one of your interests
#     scores well, instead of being penalised for not resembling the average of all
#     of them.
#
# The averaging only starts denoising once a group holds two or three seeds -- with
# one seed a group is just that paper. A seed with no "group" gets a group to itself.
#
# This is the highest-leverage setting in the file. Two or three good seeds per
# interest is what would let you lower SEMANTIC_ADMIT_SCORE and widen coverage
# without generic machine-learning papers coming along with it.

CANONICAL_PAPERS = [
    # ---- single-cell foundation models -----------------------------
    {
        "group": "single-cell foundation models",
        "title": "Large-scale foundation model on single-cell transcriptomics",
        "summary": "A pretrained foundation model on single-cell transcriptomics with 100 "
                   "million parameters covering about 20,000 genes, pretrained on over 50 "
                   "million human single-cell transcriptomic profiles. Its asymmetric "
                   "transformer-like architecture captures context relations among genes "
                   "across cell types and states, reaching state-of-the-art performance on "
                   "gene expression enhancement, drug response prediction, perturbation "
                   "prediction and cell type annotation.",
        # 10.1038/s41592-024-02305-7
    },
    {
        "group": "single-cell foundation models",
        "title": "Zero-shot evaluation reveals limitations of single-cell foundation models",
        "summary": "Foundation models such as scGPT and Geneformer have not been rigorously "
                   "evaluated in the zero-shot setting, where they are used without any "
                   "further training. Zero-shot performance is critical to applications that "
                   "exclude fine-tuning, such as discovery settings where labels are "
                   "unknown. Evaluation of Geneformer and scGPT suggests these models may "
                   "face reliability challenges and can be outperformed by simpler methods.",
        # 10.1186/s13059-025-03574-x
    },
    {
        "group": "single-cell foundation models",
        "title": "Universal cell embedding provides a foundation model for cell biology",
        "summary": "The universal cell embedding (UCE) foundation model is trained on a "
                   "large corpus of cell data by self-supervision, creating a unified "
                   "biological latent space that represents cells across diverse tissues and "
                   "species. New cells are embedded with no data labelling, model training "
                   "or fine-tuning. UCE was used to build the Integrated Mega-scale Atlas of "
                   "36 million cells and more than 1,000 cell types across eight species.",
        # 10.1038/s41586-026-10689-z
    },
    {
        "group": "single-cell foundation models",
        "title": "Transfer learning enables predictions in network biology.",
        "summary": "Context-aware, attention-based model Geneformer pretrained on ~30M "
                   "single-cell transcriptomes to learn gene network dynamics.",
    },
    {
        "group": "single-cell foundation models",
        "title": "scGPT: towards building a foundation model for single-cell multi-omics",
        "summary": "A generative pretrained transformer trained on over 30M cells for "
                   "single-cell multi-omic analysis.",
    },
    {
        "group": "single-cell foundation models",
        "title": "Tahoe-x1: scaling perturbation-trained single-cell foundation models",
        "summary": "Tx1 is pretrained on 200M+ perturbation-rich scRNA profiles and "
                   "fine-tuned for cancer-relevant prediction tasks.",
    },
    {
        "group": "single-cell foundation models",
        "title": "Novae: a graph-based foundation model for spatial transcriptomics data",
        "summary": "Spatial transcriptomics gives high-resolution insight into gene "
                   "expression within the spatial context of tissues, essential for "
                   "identifying spatial domains and microenvironment organization. Novae is "
                   "a graph-based foundation model that extracts representations of cells "
                   "within their spatial contexts, enabling zero-shot domain inference "
                   "across gene panels, tissues and technologies, batch-effect correction "
                   "and a nested hierarchy of spatial domains.",
        # 10.1038/s41592-025-02899-6
    },
    # ---- perturbation prediction -----------------------------------
    {
        "group": "perturbation prediction",
        "title": "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq",
        "summary": "Genome-scale Perturb-seq combines CRISPR interference (CRISPRi) with "
                   "single-cell RNA-sequencing readouts across more than 2.5 million human "
                   "cells, targeting all expressed genes. Transcriptional phenotypes predict "
                   "the function of poorly characterized genes and allow in-depth dissection "
                   "of complex cellular phenomena, from RNA processing to differentiation, "
                   "yielding an information-rich genotype-phenotype map.",
        # 10.1016/j.cell.2022.05.013
    },
    {
        "group": "perturbation prediction",
        "title": "Predicting transcriptional outcomes of novel multigene perturbations with GEARS",
        "summary": "GEARS (graph-enhanced gene activation and repression simulator) "
                   "integrates deep learning with a knowledge graph of gene-gene "
                   "relationships to predict transcriptional responses to single and "
                   "multigene perturbations from single-cell RNA-sequencing perturbational "
                   "screens. It predicts outcomes of perturbing gene combinations that were "
                   "never experimentally perturbed and distinguishes genetic interaction "
                   "subtypes in combinatorial screens.",
        # 10.1038/s41587-023-01905-6
    },
    {
        "group": "perturbation prediction",
        "title": "Predicting cellular responses to complex perturbations in high-throughput screens",
        "summary": "The compositional perturbation autoencoder (CPA) combines the "
                   "interpretability of linear models with the flexibility of deep learning "
                   "for single-cell response modeling. CPA learns to in silico predict "
                   "transcriptional perturbation response at the single-cell level for "
                   "unseen dosages, cell types, time points and species, predicts unseen "
                   "drug combinations, and imputes missing combinations in a Perturb-seq "
                   "genetic screen.",
        # 10.15252/msb.202211517
    },
    # ---- multi-omics integration -----------------------------------
    {
        "group": "multi-omics integration",
        "title": "Integrated analysis of multimodal single-cell data",
        "summary": "Simultaneous measurement of multiple modalities in single cells requires "
                   "methods that define cellular states from multimodal data. "
                   "Weighted-nearest neighbor analysis is an unsupervised framework that "
                   "learns the relative utility of each data type in each cell, enabling "
                   "integrative analysis of multiple modalities. Applied to a CITE-seq "
                   "dataset of 211,000 human PBMCs, it builds a multimodal reference atlas "
                   "of the circulating immune system.",
        # 10.1016/j.cell.2021.04.048
    },
    {
        "group": "multi-omics integration",
        "title": "Multi-omics single-cell data integration and regulatory inference with graph-linked embedding",
        "summary": "Most single-cell datasets include only one omics modality, and different "
                   "omics layers typically have distinct feature spaces. GLUE (graph-linked "
                   "unified embedding) bridges this gap by explicitly modeling regulatory "
                   "interactions across omics layers. Benchmarking on heterogeneous "
                   "single-cell multi-omics data covers triple-omics integration, "
                   "integrative regulatory inference and multi-omics human cell atlas "
                   "construction.",
        # 10.1038/s41587-022-01284-4
    },
    {
        "group": "multi-omics integration",
        "title": "MultiVI: deep generative model for the integration of multimodal data",
        "summary": "Jointly profiling the transcriptome, chromatin accessibility and other "
                   "molecular properties of single cells offers a powerful way to study "
                   "cellular diversity. MultiVI is a probabilistic deep generative model for "
                   "such multiomic data that creates a joint representation of all input "
                   "modalities and leverages it to enhance single-modality datasets, even "
                   "for cells in which one or more modalities are missing.",
        # 10.1038/s41592-023-01909-9
    },
    {
        "group": "multi-omics integration",
        "title": "Integration of spatial and single-cell data across modalities with weakly linked features",
        "summary": "No technology captures all modalities within the same cell, and "
                   "cross-modal integration usually depends on highly correlated, a priori "
                   "linked features. MaxFuse (matching X-modality via fuzzy smoothed "
                   "embedding) uses iterative coembedding, data smoothing and cell matching "
                   "to integrate weakly linked modalities, enabling spatial consolidation of "
                   "proteomic, transcriptomic and epigenomic information at single-cell "
                   "resolution.",
        # 10.1038/s41587-023-01935-0
    },
    # ---- gene regulation and chromatin -----------------------------
    {
        "group": "gene regulation and chromatin",
        "title": "SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks",
        "summary": "Joint profiling of chromatin accessibility and gene expression in single "
                   "cells enables inference of enhancer-driven gene regulatory networks "
                   "(GRNs). SCENIC+ predicts enhancers along with candidate upstream "
                   "transcription factors (TFs) and links these enhancers to candidate "
                   "target genes, using a curated collection of >30,000 motifs. It resolves "
                   "gene regulation along differentiation trajectories and the effect of TF "
                   "perturbations on cell state.",
        # 10.1038/s41592-023-01938-4
    },
    {
        "group": "gene regulation and chromatin",
        "title": "Mapping enhancer–gene regulatory interactions from single-cell data",
        "summary": "Predicting enhancer-gene regulatory interactions from single-cell data "
                   "has been challenging. scE2G is a family of classification models that "
                   "predict enhancer-gene regulation from single-cell ATAC-seq or multiomic "
                   "RNA and ATAC-seq features, trained on a CRISPR perturbation dataset of "
                   ">10,000 element-gene pairs. It is benchmarked against CRISPR "
                   "perturbations, fine-mapped eQTLs and GWAS variant-gene associations.",
        # 10.1038/s41588-026-02695-8
    },
    # ---- lung, limb & hematopoietic development ---------------------
    {
        "group": "lung, limb & hematopoietic development",
        "title": "The changing mouse embryo transcriptome at whole tissue and single-cell resolution",
        "summary": "Cross-tissue bulk RNA-seq study of organogenesis; Rest/NRSF-regulated "
                   "H3K27me3 de-repression; house-keeping gene transcriptional and "
                   "post-transcriptional regulation; scRNA-seq of mouse limb development; "
                   "gene regulatory network inference from multi-omic sequencing data.",
    },
    {
        "group": "lung, limb & hematopoietic development",
        "title": "A human fetal lung cell atlas uncovers proximal-distal gradients of differentiation and key regulators of epithelial fates",
        "summary": "Comprehensive map of 144 human fetal lung cell populations built from "
                   "scRNA-seq and scATAC-seq.",
    },
    {
        "group": "lung, limb & hematopoietic development",
        "title": "A human embryonic limb cell atlas resolved in space and time",
        "summary": "Spatiotemporal mapping of human limb development using scRNA-seq and "
                   "Visium.",
    },
    {
        "group": "lung, limb & hematopoietic development",
        "title": "Early human lung immune cell development and its role in epithelial cell fate",
        "summary": "Temporal single-cell mapping of immune cells in the lung, using "
                   "scBCR/abTCR/gdTCR-seq and lymphoid lineage inference.",
    },
    # ---- LOCAL (send_lit_feed-2): the erythroid-atlas project ---------------
    # Three groups rather than one, because a group is averaged into a single
    # direction: erythroid regulation, developmental blood atlases and transgene
    # mapping share little vocabulary, and one centroid for all three would sit
    # close to none of them. Summaries are condensed from the published abstracts.
    # ---- erythropoiesis & globin regulation --------------------------------
    {
        "group": "erythropoiesis & globin regulation",
        "title": "Transcriptional States and Chromatin Accessibility Underlying Human Erythropoiesis",
        "summary": "Deep transcriptomic and accessible chromatin profiling of a faithful "
                   "ex vivo human erythroid differentiation system from hematopoietic stem "
                   "and progenitor cells reveals stage-specific transcriptional states and "
                   "chromatin accessibility during erythropoiesis, with differentiation "
                   "stage-predominant roles for master regulators including GATA1 and "
                   "KLF1. Chromatin profiles are integrated with common and rare genetic "
                   "variants associated with erythroid cell traits and diseases.",
        # 10.1016/j.celrep.2019.05.046
    },
    {
        "group": "erythropoiesis & globin regulation",
        "title": "Population snapshots predict early haematopoietic and erythroid hierarchies",
        "summary": "The formation of red blood cells begins with the differentiation of "
                   "multipotent haematopoietic progenitors. Single-cell transcriptomics, "
                   "fate assays and a theory that predicts cell fates from population "
                   "snapshots show that haematopoietic progenitors differentiate through a "
                   "continuous, hierarchical structure into seven blood lineages, with "
                   "coupling between erythroid and basophil or mast cell fates and a sharp "
                   "transcriptional switch that activates terminal erythroid "
                   "differentiation.",
        # 10.1038/nature25741
    },
    {
        "group": "erythropoiesis & globin regulation",
        "title": "Global transcriptome analyses of human and murine terminal erythroid differentiation",
        "summary": "Pure populations of human and murine erythroblasts at distinct stages "
                   "of terminal erythroid differentiation were sorted and subjected to RNA "
                   "sequencing, creating unbiased, stage-specific transcriptomes. There "
                   "are vast temporal changes in gene expression across differentiation "
                   "stages and numerous differences between human and murine "
                   "transcriptomes, providing a resource for studies of normal and "
                   "perturbed erythropoiesis.",
        # 10.1182/blood-2014-01-548305
    },
    {
        "group": "erythropoiesis & globin regulation",
        "title": "Development and differentiation of the erythroid lineage in mammals",
        "summary": "The primitive erythroid lineage is the first to be specified in the "
                   "developing embryo. Two transient waves of hematopoietic "
                   "progenitor-derived erythropoiesis are observed before hematopoietic "
                   "stem cells take over to produce definitive red blood cells in the "
                   "fetal liver and later the bone marrow. Key aspects of mammalian "
                   "erythroid development and maturation and the differences among the "
                   "primitive and definitive erythroid cell lineages are highlighted.",
        # 10.1016/j.dci.2015.12.012
    },
    {
        "group": "erythropoiesis & globin regulation",
        "title": "Human fetal hemoglobin expression is regulated by the developmental stage-specific repressor BCL11A",
        "summary": "Differences in the amount of fetal hemoglobin (HbF) that persists into "
                   "adulthood affect the severity of sickle cell disease and the "
                   "beta-thalassemia syndromes. Expression of full-length BCL11A is "
                   "developmentally restricted to adult erythroid cells, its "
                   "down-regulation in primary adult erythroid cells leads to robust HbF "
                   "expression, and BCL11A occupies discrete sites in the beta-globin gene "
                   "cluster, consistent with a direct role in globin gene regulation.",
        # 10.1126/science.1165409
    },
    {
        "group": "erythropoiesis & globin regulation",
        "title": "Defining the Minimal Factors Required for Erythropoiesis through Direct Lineage Conversion",
        "summary": "Erythroid commitment and differentiation proceed through a "
                   "lineage-restricted transcriptional network. A transcription factor "
                   "screen shows that Gata1, Tal1, Lmo2 and c-Myc rapidly convert murine "
                   "and human fibroblasts directly to induced erythroid progenitors, whose "
                   "transcriptional signature resembles primitive erythroid progenitors in "
                   "the yolk sac, whereas adding Klf1 or Myb gives a more adult-type "
                   "globin expression pattern.",
        # 10.1016/j.celrep.2016.05.027
    },
    # ---- developmental hematopoiesis --------------------------------------
    {
        "group": "developmental hematopoiesis",
        "title": "Decoding human fetal liver haematopoiesis",
        "summary": "Definitive haematopoiesis in the fetal liver supports self-renewal and "
                   "differentiation of haematopoietic stem cells and multipotent "
                   "progenitors. Single-cell transcriptome profiling of fetal liver, skin, "
                   "kidney and yolk sac cells identifies the repertoire of human blood and "
                   "immune cells during development, infers differentiation trajectories "
                   "from HSC/MPPs, and shows a shift in fetal liver haematopoietic "
                   "composition during gestation away from being predominantly erythroid.",
        # 10.1038/s41586-019-1652-y
    },
    {
        "group": "developmental hematopoiesis",
        "title": "Yolk sac cell atlas reveals multiorgan functions during human early development",
        "summary": "A comprehensive multiomic reference of the human yolk sac from 3 to 8 "
                   "postconception weeks, integrating single-cell protein and gene "
                   "expression data. Beyond its role as a site of hematopoiesis, the yolk "
                   "sac has roles in metabolism, coagulation, vascular development and "
                   "hematopoietic regulation. The emergence and decline of yolk sac "
                   "hematopoietic stem and progenitor cells from hemogenic endothelium is "
                   "reconstructed.",
        # 10.1126/science.add7564
    },
    {
        "group": "developmental hematopoiesis",
        "title": "Mapping human haematopoietic stem cells from haemogenic endothelium to birth",
        "summary": "A single-cell transcriptome map of human haematopoietic tissues from "
                   "the first trimester to birth identifies an HSC signature that "
                   "distinguishes haematopoietic stem cells from progenitors throughout "
                   "gestation. Nascent HSCs populate the aorta-gonad-mesonephros region, "
                   "placenta and yolk sac before colonizing the liver, and HSC origin is "
                   "tracked to haemogenic endothelial cells using spatial transcriptomics.",
        # 10.1038/s41586-022-04571-x
    },
    {
        "group": "developmental hematopoiesis",
        "title": "Blood and immune development in human fetal bone marrow and Down syndrome",
        "summary": "Development of human fetal bone marrow, including stroma, is detailed "
                   "using multi-omic assessment of mRNA and multiplexed protein epitope "
                   "expression. The full blood and immune cell repertoire is established "
                   "in a short window early in the second trimester. Haematopoietic "
                   "progenitors from fetal liver, fetal bone marrow and cord blood differ "
                   "transcriptionally and functionally, and B lymphocyte, erythroid and "
                   "myeloid development is selectively disrupted in Down syndrome.",
        # 10.1038/s41586-021-03929-x
    },
    # ---- transgene & vector integration mapping ---------------------------
    {
        "group": "transgene & vector integration mapping",
        "title": "Large-scale discovery of mouse transgenic integration sites reveals frequent structural variation and insertional mutagenesis",
        "summary": "Transgenesis through random integration of DNA fragments into the host "
                   "genome can cause insertional mutagenesis and structural variation, yet "
                   "the insertion sites of most transgenic lines are unknown. Targeted "
                   "locus amplification identifies both the insertion site and content of "
                   "transgenes by deep sequencing. Across 40 transgenic mouse lines, "
                   "transgenes disrupt endogenous coding sequence in half, often with "
                   "large deletions or structural variations at the insertion site.",
        # 10.1101/gr.233866.117
    },
    {
        "group": "transgene & vector integration mapping",
        "title": "Efficient mapping of transgene integration sites and local structural changes in Cre transgenic mice using targeted locus amplification",
        "summary": "For lines generated by pronuclear microinjection of a transgene "
                   "construct, the integration site is random and in most cases not known, "
                   "and integration can disrupt an endogenous gene. Targeted locus "
                   "amplification efficiently maps the transgene location in Cre and "
                   "CreERT2 transgenic lines, identifying the exact integration site and "
                   "breakpoint sequences together with structural changes around the "
                   "integration site.",
        # 10.1093/nar/gkw1329
    },
    {
        "group": "transgene & vector integration mapping",
        "title": "Locating and Characterizing a Transgene Integration Site by Nanopore Sequencing",
        "summary": "Locating sites of foreign DNA incorporation in mammalian genomes has "
                   "proven burdensome, so the genomic location of most transgenes remains "
                   "unknown. Nanopore sequencing identifies the site of transgene "
                   "integration of a widely used fluorescent reporter and simultaneously "
                   "yields an estimate of transgene copy number, direct evidence of "
                   "transgene inversions, contaminating DNA within the transgene array, "
                   "and definitive genotyping.",
        # 10.1534/g3.119.300582
    },
    {
        "group": "transgene & vector integration mapping",
        "title": "INSPIIRED: A Pipeline for Quantitative Analysis of Sites of New DNA Integration in Cellular Genomes",
        "summary": "Integration of new DNA into cellular genomes mediates replication of "
                   "retroviruses and transposons and is used in human gene therapy. "
                   "Tracking distributions of integration sites characterizes populations "
                   "of transduced cells and monitors outgrowth of pathogenic clones. "
                   "INSPIIRED is a pipeline for quantitative analysis of integration site "
                   "distributions from paired-end sequencing, with software for alignment, "
                   "quality control and inference of the abundance of gene-modified cells.",
        # 10.1016/j.omtm.2016.11.002
    },
    {
        "group": "transgene & vector integration mapping",
        "title": "Joint profiling of chromatin accessibility and CAR-T integration site analysis at population and single-cell levels",
        "summary": "In CAR-T immunotherapy the T cell genome is modified by integration of "
                   "lentiviral vectors. EpiVIA jointly profiles chromatin accessibility "
                   "and lentiviral integration sites at population and single-cell "
                   "levels, validated in clonal cells with defined integration sites, "
                   "measuring lentiviral integration sites and chromatin accessibility of "
                   "host and viral genomes at single-cell resolution.",
        # 10.1073/pnas.1919259117
    },
]


# ---- Feed profiles ---------------------------------------------------------
# A profile is the complete definition of one audience. To add another feed, copy
# this entry, give it a unique key/output_subdir, and provide that audience's
# keywords, seed groups, and relevance rubric. Run it with --profile NAME. Keeping
# the configuration in this tracked script lets the repository remain a two-file
# tool while each profile's generated history stays local and separate.
DEFAULT_PROFILE_NAME = "single_cell_ml"

DOMAIN_KEYWORDS = [
    "single-cell", "single cell", "single-nucleus", "single nucleus",
    "scrna", "snrna", "scatac", "snatac", "perturb-seq",
    "spatial transcriptomic", "spatial omic", "spatially resolved transcript",
    "multi-omic", "multi omic", "multiomic", "multiome", "cite-seq",
    "chromatin accessibility", "open chromatin", "gene regulatory",
    "gene expression", "transcriptomic", "cell atlas", "cell state",
    "virtual cell", "virtual-cell",

    # LOCAL: this gate only ever REMOVES papers, so every biological term we admit
    # on in INCLUDE_KEYWORDS has to appear here too or the gate silently undoes it.
    # Upstream's list is cardiovascular- and assay-centric; ours also covers organ
    # and lineage development, which is most of what our extra bioRxiv subject
    # feeds carry. Deliberately NOT mirrored from INCLUDE_KEYWORDS: "multimodal"
    # (an ML term as often as a biological one -- letting it through would defeat
    # the generic-ML rejection this gate exists for) and bare "atlas" (a genuine
    # cell atlas already matches "cell atlas", or its organ term below).
    "lung", "kidney", "limb", "hematopoietic", "hematopoiesis",
    "organoid", "organogenesis", "CPAM", "neuroendocrine",
    "developmental biology", "embryo", "fetal", "morphogenesis",
    "genomics", "bioinformatics", "computational biology",

    # LOCAL: molecular-biology vocabulary, added after measuring the gate against
    # one real digest (465 papers). Upstream's list is phrased around assay names,
    # so it dropped genuine gene-regulation and development work that simply names
    # the biology instead -- an enhancer-promoter hub paper, a collective cell
    # migration paper, a "cellular atlas" that never says "cell atlas". These terms
    # have no machine-learning reading, so they recover that work without letting
    # the EHR/imaging/remote-sensing papers back in. Note "differentiation" is
    # absent on purpose: automatic differentiation is an ML term.
    "cellular atlas", "cell fate", "cell type", "cell differentiation",
    "transcription factor", "enhancer", "promoter", "regulatory element",
    "epigenom", "lineage", "rna-seq", "atac-seq", "chip-seq",

    # LOCAL (send_lit_feed-2): see BLOOD_KEYWORDS.
    *BLOOD_KEYWORDS,
    *TRANSGENE_KEYWORDS,
]

PROFILES = {
    DEFAULT_PROFILE_NAME: {
        "display_name": "Single-Cell ML + Erythroid / Transgene",
        "output_subdir": "",  # preserve the existing cron and digest history
        # Unused since upstream ac4de18 removed the LLM judge; kept as the written
        # statement of what this feed is for.
        "description": (
            "Deep-learning model development, evaluation, and representation learning for "
            "single-cell or multi-omics data, perturbation prediction, and virtual-cell "
            "modeling are the primary interests. Biological or mechanistic studies that "
            "mainly use an omics assay are background priority. LOCAL: single-cell and "
            "multi-omic studies of "
            "organ and lineage development -- lung, kidney, limb and hematopoiesis, "
            "including developmental cell atlases -- are a first-class interest here, not "
            "background. send_lit_feed-2 adds the erythroid-atlas project: primitive "
            "versus definitive erythropoiesis, globin switching, the GATA/LMO2/TAL1 "
            "erythroid regulatory network, developmental hematopoiesis, and methods for "
            "locating or quantifying transgenes and integrated vectors."
        ),
        "include_keywords": INCLUDE_KEYWORDS,
        "domain_keywords": DOMAIN_KEYWORDS,
        "exclude_keywords": EXCLUDE_KEYWORDS,
        "tags": TAGS,
        "canonical_papers": CANONICAL_PAPERS,
    },
}

ACTIVE_PROFILE_NAME = DEFAULT_PROFILE_NAME
ACTIVE_PROFILE = PROFILES[DEFAULT_PROFILE_NAME]


# Sentence-transformer model (small but decent)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



# ==========================
# ======= DATA MODEL ========
# ==========================

@dataclass
class Paper:
    title: str
    summary: str
    link: str
    published: datetime
    source: str
    score: float = math.nan  # semantic similarity score (filled later)
    authors: List[str] = field(default_factory=list)
    semantic_group: str = ""



# ==========================
# ====== HELPERS ===========
# ==========================

def strip_html(text: str) -> str:
    """Very small HTML stripper for titles and summaries."""
    # We keep it minimal; if you want, swap for 'beautifulsoup4'.
    # Unescape before stripping: some sources (Crossref especially) deliver markup
    # already escaped, e.g. "R&lt;sup&gt;3&lt;/sup&gt;". Stripping first would find
    # no tags to remove and leave the escaped ones in the text as literal noise.
    import re
    from html import unescape
    return re.sub(r"<[^>]+>", "", unescape(text or ""))


# arXiv's RSS wraps every abstract in a fixed preamble:
#   "arXiv:2608.14846v1 Announce Type: new \nAbstract: <the actual abstract>"
# Left in place it burns ~50 characters of the digest's abstract budget and reads
# like a machine artefact.
_ARXIV_PREAMBLE_RE = re.compile(
    r"^\s*arXiv:\S+\s*Announce Type:\s*\S+\s*Abstract:\s*", re.IGNORECASE
)


def clean_summary(text: str) -> str:
    """Strip feed-specific boilerplate from an abstract."""
    return _ARXIV_PREAMBLE_RE.sub("", text or "").strip()


def parse_datetime(entry: Dict[str, Any]) -> datetime:
    """Try to get a timezone-aware datetime for the entry."""
    if "published_parsed" in entry and entry["published_parsed"]:
        return datetime(*entry["published_parsed"][:6], tzinfo=timezone.utc)
    if "updated_parsed" in entry and entry["updated_parsed"]:
        return datetime(*entry["updated_parsed"][:6], tzinfo=timezone.utc)
    # Fallback: now
    return datetime.now(timezone.utc)


def tags_for(paper: Paper) -> List[tuple]:
    """Topic chips for one paper, as (label, colour) pairs.

    Derived from TAGS at render time rather than stored on the Paper: it costs
    nothing, and a paper reloaded from an old digest gets today's tag vocabulary
    instead of whatever was current when it was first seen.
    """
    text = match_text(paper)
    hits = [
        (label, colour)
        for label, colour, keywords in TAGS
        if any(k.lower() in text for k in keywords)
    ]
    return hits[:TAG_MAX]


# bioRxiv packs every author into one dc:creator string as "Surname, I.-I.", so a
# plain split on ", " would tear each name in half. This matches surname/initials
# pairs instead.
#
# Two details the feed forces, both learned from real entries:
#   * initials must carry real periods. Without that a bare capital matches and the
#     pattern starts eating "Given Family" names from arXiv and Cell ("Haining Lin"
#     -> "H").
#   * initials are separated by a space as often as a hyphen, and may include
#     lowercase particles: "Yavuz, B. R.", "Fiuza, T. d. S.", "Zhang, Q.-Q.".
_INITIALS = r"[A-Z]\.(?:[\s-]*[A-Za-z]{1,3}\.)*"
_SURNAME_INITIALS_RE = re.compile(rf"[^,]+,\s*{_INITIALS}(?=\s*(?:,|;|$))")


def _split_author_blob(blob: str) -> List[str]:
    """Split a one-string author list into names, handling both common shapes."""
    blob = (blob or "").strip()
    if not blob:
        return []

    # Shape 1, bioRxiv: "Zhang, Q.-Q., Zhang, S.-W." -- surname, initials, repeated.
    pairs = [m.group(0).strip() for m in _SURNAME_INITIALS_RE.finditer(blob)]
    if pairs and sum(len(x) for x in pairs) >= 0.6 * len(blob):
        # Flip to "Q.-Q. Zhang" so the rendered line reads like every other feed
        # and its commas separate authors rather than appearing inside each name.
        flipped = []
        for name in pairs:
            surname, _, initials = name.partition(",")
            flipped.append(f"{initials.strip()} {surname.strip()}".strip())
        return flipped

    # Shape 2, arXiv and Cell: "Yashrajsinh Jadeja, Haining Lin" -- given name first.
    return [part.strip() for part in blob.split(",") if part.strip()]


def extract_authors(entry: Dict[str, Any]) -> List[str]:
    """Author names from a feed entry, whatever shape the feed uses.

    Feeds disagree badly here. Nature and PLOS give a real list in entry.authors,
    while entry.author on those feeds is only the LAST author. arXiv, bioRxiv and
    Cell instead cram every author into a single string. Science and OUP
    Bioinformatics supply none at all. Reading entry.author uniformly would
    silently mean "last author" on some feeds and "everyone" on others.
    """
    listed = entry.get("authors") or []
    names = [
        strip_html(a.get("name", "")).strip()
        for a in listed
        if isinstance(a, dict) and a.get("name")
    ]
    # A 1-element list is how feedparser reports a single crammed string too.
    if len(names) > 1:
        return names
    blob = names[0] if names else strip_html(entry.get("author", "") or "").strip()
    return _split_author_blob(blob)


def format_authors(authors: List[str], max_shown: int = 3) -> str:
    """First … last, so a 40-author consortium paper stays one line."""
    clean = [a for a in (x.strip() for x in authors) if a]
    if not clean:
        return ""
    if len(clean) <= max_shown:
        return ", ".join(clean)
    return f"{clean[0]}, …, {clean[-1]}"


def activate_profile(name: str) -> dict:
    """Activate one named audience without mixing its output/history with another."""
    if name not in PROFILES:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile {name!r}; choose one of: {choices}")

    global ACTIVE_PROFILE_NAME, ACTIVE_PROFILE
    global INCLUDE_KEYWORDS, EXCLUDE_KEYWORDS, TAGS, CANONICAL_PAPERS, OUTPUT_DIR

    profile = PROFILES[name]
    ACTIVE_PROFILE_NAME = name
    ACTIVE_PROFILE = profile
    INCLUDE_KEYWORDS = profile["include_keywords"]
    EXCLUDE_KEYWORDS = profile["exclude_keywords"]
    TAGS = profile["tags"]
    CANONICAL_PAPERS = profile["canonical_papers"]
    subdir = profile.get("output_subdir", name)
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, subdir) if subdir else OUTPUT_ROOT
    return profile


def match_text(paper: Paper) -> str:
    """Lower-cased title + abstract for keyword matching, false friends removed."""
    return KEYWORD_FALSE_FRIENDS.sub(" ", f"{paper.title} {paper.summary}").lower()


def is_excluded(paper: Paper) -> bool:
    """Hard veto. Applied at fetch time, before anything is embedded."""
    if not EXCLUDE_KEYWORDS:
        return False
    text = f"{paper.title} {paper.summary}".lower()
    # LOCAL: match at the START of a word, not anywhere inside one. As a plain
    # substring "plant" vetoed every transplant, implant and explant paper, and
    # "ecology" every gynecology one -- invisible while the feed was single-cell
    # methods, fatal for a hematology feed where transplantation is everywhere.
    # Suffixes still match ("plants", "plant-derived", "behaviorally").
    return any(re.search(r"\b" + re.escape(k.lower()), text) for k in EXCLUDE_KEYWORDS)


def matches_include_keywords(paper: Paper) -> bool:
    """Did the paper use your vocabulary? One of two routes into the digest."""
    if not INCLUDE_KEYWORDS:
        return True  # no include list means everything is eligible
    text = match_text(paper)
    return any(k.lower() in text for k in INCLUDE_KEYWORDS)


def matches_domain_keywords(paper: Paper) -> bool:
    """High-recall biological gate applied before semantic ranking.

    This keeps generic uses of terms such as "foundation model", "transformer",
    and "perturbation" out of the ranked pool. It is the stage that rejects
    ordinary machine-learning papers: measured pairwise separation of omics model
    work from generic ML is 0.954, so the finer distinction between a relevant
    model paper and a descriptive analysis is left to the seed-group similarity.
    """
    keywords = ACTIVE_PROFILE.get("domain_keywords", [])
    if not keywords:
        return True
    text = match_text(paper)
    return any(k.lower() in text for k in keywords)


def passes_keyword_filters(paper: Paper) -> bool:
    """Both keyword tests at once. Kept for callers that want the old behaviour."""
    return matches_include_keywords(paper) and not is_excluded(paper)


def canonicalize_url(url: str) -> str:
    """Normalize URLs so trivial differences don't create new IDs."""
    if not url:
        return ""
    try:
        u = urlsplit(url.strip())
        # drop fragment, normalize scheme+netloc+path; keep query (sometimes DOI/IDs live there)
        return urlunsplit((u.scheme.lower() or "https", u.netloc.lower(), u.path.rstrip("/"), u.query, ""))
    except Exception:
        return url.strip()

def paper_key(p: Paper) -> str:
    """
    Stable identity for dedup / 'seen before' comparison.
    Prefer canonicalized link; fallback to (source|title|date).
    """
    link = canonicalize_url(p.link)
    if link:
        return link
    date_str = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return f"{p.source}|{p.title.strip().lower()}|{date_str}"

def title_key(p: Paper) -> str:
    """Identity by title, for the same paper reached through two different links.

    paper_key cannot see that https://www.nature.com/articles/... from an RSS feed
    and https://doi.org/10.1038/... from PubMed are one paper. Short titles
    ("Correction", "Reply") are not distinctive enough to merge on, so they get no
    key at all.
    """
    key = re.sub(r"[^a-z0-9]+", " ", p.title.lower()).strip()
    return key if len(key) >= 25 else ""


def digest_date_from_path(path: str) -> str | None:
    # expects digest_YYYY-MM-DD.html
    base = os.path.basename(path)
    m = re.match(r"digest_(\d{4}-\d{2}-\d{2})\.html$", base)
    return m.group(1) if m else None

def most_recent_non_today_digest_path(output_dir: str) -> str | None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    paths = sorted(glob.glob(os.path.join(output_dir, "digest_*.html")))
    # walk backward until we find a non-today one
    for p in reversed(paths):
        if digest_date_from_path(p) and digest_date_from_path(p) != today:
            return p
    return None

_KEYS_BLOB_RE = re.compile(r"<!--\s*DIGEST_KEYS_JSON\s*(.*?)\s*-->", re.DOTALL)


def _paper_from_dict(data: dict) -> Paper:
    dt = datetime.fromisoformat(data["published"])
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return Paper(
        title=data.get("title", ""),
        summary=data.get("summary", ""),
        link=data.get("link", ""),
        published=dt,
        source=data.get("source", ""),
        score=float(data["score"]) if data.get("score") is not None else math.nan,
        # Older digests predate this field; absent means "unknown", not "no authors".
        authors=list(data.get("authors") or []),
        semantic_group=data.get("semantic_group", ""),
    )


def _digest_payload(html: str) -> dict:
    match = _KEYS_BLOB_RE.search(html)
    payload = json.loads(match.group(1)) if match else {}
    return payload if isinstance(payload, dict) else {}


def _papers_from_payload(payload: dict, field_name: str) -> Optional[List[Paper]]:
    records = payload.get(field_name)
    if not isinstance(records, list):
        return None
    papers = [_paper_from_dict(record) for record in records]
    return [paper for paper in papers if paper.title]

def load_papers_from_html(path: str) -> List[Paper]:
    """
    Load papers from a prior digest.

    Supports two formats:
      (A) JSON blob contains {"papers":[...]}  (new format)
      (B) JSON blob contains {"keys":[...]}    (old format) -> fallback: parse HTML cards
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        payload = _digest_payload(html)

        # (A) Preferred: structured papers in JSON blob
        structured = _papers_from_payload(payload, "papers")
        if structured is not None:
            return structured

        # (B) Fallback: parse the rendered HTML cards (works for your 01-04 file)
        card_re = re.compile(
            r'<div class="paper">.*?'
            r'<h3><a href="(?P<link>[^"]+)".*?>\s*(?P<title>.*?)\s*</a></h3>.*?'
            r'Source:\s*<strong>(?P<source>.*?)</strong>\s*·.*?'
            r'Date:\s*(?P<date>\d{4}-\d{2}-\d{2}).*?'
            r'(?:Relevance|Semantic) score:\s*(?P<score>[\d.]+|n/a).*?'
            r'<div class="summary">(?P<summary>.*?)</div>.*?'
            r'</div>',
            re.DOTALL
        )

        out: List[Paper] = []
        for m in card_re.finditer(html):
            title = strip_html(m.group("title")).strip()
            link = strip_html(m.group("link")).strip()
            source = strip_html(m.group("source")).strip()
            date_str = strip_html(m.group("date")).strip()
            score_str = strip_html(m.group("score")).strip()
            summary = strip_html(m.group("summary")).strip()

            published = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            score = float(score_str) if score_str != "n/a" else math.nan

            if title:
                out.append(Paper(
                    title=title,
                    summary=summary,
                    link=link,
                    published=published,
                    source=source,
                    score=score,
                ))
        return out

    except Exception:
        return []


def split_new_vs_previous(papers_ranked: List[Paper], seen_keys: set[str]) -> tuple[List[Paper], List[Paper]]:
    new_items, prev_items = [], []
    for p in papers_ranked:
        k = paper_key(p)
        if k in seen_keys:
            prev_items.append(p)
        else:
            new_items.append(p)
    return new_items, prev_items


def _crossref_page(filter_expr: str, rows: int, cursor: str, agent: str) -> Optional[Dict[str, Any]]:
    """One Crossref page, retrying on rate limits. None if it never succeeds."""
    delay = CROSSREF_PAUSE_SEC
    for attempt in range(CROSSREF_RETRIES):
        try:
            resp = requests.get(
                CROSSREF_API,
                params={
                    "filter": filter_expr,
                    "rows": rows,
                    "cursor": cursor,
                    "select": "DOI,title,abstract,posted,created,URL,author",
                },
                headers={"User-Agent": agent},
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json().get("message", {})
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = float(resp.headers.get("Retry-After") or delay)
                time.sleep(min(wait, 30.0))
                delay *= 2
                continue
            resp.raise_for_status()
        except requests.RequestException:
            time.sleep(min(delay, 30.0))
            delay *= 2
    return None


def _crossref_date(field: Any) -> Optional[datetime]:
    """Turn Crossref's {"date-parts": [[YYYY, MM, DD]]} into a UTC datetime."""
    try:
        parts = (field or {}).get("date-parts") or []
        if not parts or not parts[0] or parts[0][0] is None:
            return None
        y, m, d = (list(parts[0]) + [1, 1])[:3]
        return datetime(int(y), int(m or 1), int(d or 1), tzinfo=timezone.utc)
    except Exception:
        return None


def _crossref_authors(raw: Any) -> List[str]:
    """Crossref gives structured {given, family}; join into display order."""
    names = []
    for a in raw or []:
        if not isinstance(a, dict):
            continue
        full = " ".join(x for x in (a.get("given"), a.get("family")) if x).strip()
        if not full:
            full = (a.get("name") or "").strip()  # consortium/organisation authors
        if full:
            names.append(full)
    return names


def fetch_crossref(feed: Dict[str, Any]) -> List[Paper]:
    """Pull one preprint server by DOI prefix from the Crossref REST API.

    Same contract as the RSS path: return keyword-passing Paper objects inside the
    lookback window. Crossref filters by date server-side, so unlike an RSS feed
    this sees the whole window rather than whatever the latest N items happen to be.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    budget = int(feed.get("max_items", MAX_ITEMS_PER_FEED))

    agent = "lit_feed/1.0 (+https://github.com/zyj1729/lit_feed)"
    if CROSSREF_CONTACT:
        agent += f" mailto:{CROSSREF_CONTACT}"

    papers: List[Paper] = []
    seen = 0
    cursor = "*"
    page = 0
    while seen < budget:
        if page:
            time.sleep(CROSSREF_PAUSE_SEC)  # deep paging is rate-limited
        message = _crossref_page(
            filter_expr=(
                f"prefix:{feed['prefix']},type:posted-content,"
                f"from-created-date:{cutoff:%Y-%m-%d}"
            ),
            rows=min(CROSSREF_ROWS, budget - seen),
            cursor=cursor,
            agent=agent,
        )
        page += 1
        if message is None:
            # Rate-limited or down. Keep whatever we already have rather than
            # throwing the feed away; the next run will see the rest.
            print(f"  ! Crossref unavailable after {CROSSREF_RETRIES} tries; "
                  f"continuing with {len(papers)} item(s) from {seen} record(s)")
            break
        items = message.get("items") or []
        if not items:
            break
        seen += len(items)

        for item in items:
            title = strip_html(" ".join(item.get("title") or [])).strip()
            if not title:
                continue
            # "posted" is when the preprint went up; "created" is when Crossref
            # first saw it. Prefer the former, fall back to the latter.
            published = _crossref_date(item.get("posted")) or _crossref_date(item.get("created"))
            if published is None or published < cutoff:
                continue

            doi = item.get("DOI", "")
            paper = Paper(
                title=title,
                # Crossref abstracts are JATS XML; strip_html handles the tags.
                summary=strip_html(item.get("abstract", "")).strip(),
                link=item.get("URL") or (f"https://doi.org/{doi}" if doi else feed["url"]),
                published=published,
                source=feed["name"],
                authors=_crossref_authors(item.get("author")),
            )
            # Only the veto here. The include test moves to admission, after
            # ranking, so a paper can still get in on semantic similarity alone.
            if not is_excluded(paper):
                papers.append(paper)

        cursor = message.get("next-cursor")
        if not cursor:
            break

    return papers


def _pubmed_request(endpoint: str, params: Dict[str, Any]) -> Optional[requests.Response]:
    """One E-utilities call, retrying on rate limits. None if it never succeeds."""
    payload = {"db": "pubmed", "tool": "lit_feed", **params}
    if PUBMED_CONTACT:
        payload["email"] = PUBMED_CONTACT
    if PUBMED_API_KEY:
        payload["api_key"] = PUBMED_API_KEY
    delay = 1.0
    for attempt in range(PUBMED_RETRIES):
        try:
            # POST, because an efetch id list overruns the URL length limit.
            resp = requests.post(PUBMED_API + endpoint, data=payload, timeout=60)
            # NCBI reports rate limiting as 429, but also as a 200 whose body is
            # {"error": "API rate limit exceeded"}.
            if resp.status_code == 200 and b"API rate limit exceeded" not in resp.content[:300]:
                return resp
        except requests.RequestException:
            pass
        time.sleep(min(delay, 30.0))
        delay *= 2
    return None


def _pubmed_date(article: ET.Element) -> Optional[datetime]:
    """When the record reached PubMed.

    Not the journal's PubDate: that is an issue date, often weeks in the future for
    an ahead-of-print paper and sometimes only a year. The "pubmed" history date is
    always complete, and is the same clock the edat search window runs on.
    """
    for status in ("pubmed", "entrez"):
        node = article.find(f"PubmedData/History/PubMedPubDate[@PubStatus='{status}']")
        if node is None:
            continue
        try:
            return datetime(int(node.findtext("Year")), int(node.findtext("Month") or 1),
                            int(node.findtext("Day") or 1), tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
    return None


def _pubmed_text(node: Optional[ET.Element]) -> str:
    """Flatten an element that may carry inline markup (<i>, <sub>, <sup>)."""
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def fetch_pubmed(feed: Dict[str, Any]) -> List[Paper]:
    """Run one PubMed query (feed["term"]) over the lookback window.

    Same contract as the RSS and Crossref paths. Like Crossref, PubMed filters by
    date server-side, so this sees the whole window, and it returns full abstracts
    where a publisher RSS feed gives a teaser.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    budget = int(feed.get("max_items", MAX_ITEMS_PER_FEED))

    # edat = the date the record entered PubMed, which is what "new" means here.
    resp = _pubmed_request("esearch.fcgi", {
        "term": feed["term"], "datetype": "edat", "reldate": LOOKBACK_DAYS,
        "retmax": budget, "retmode": "json",
    })
    try:
        result = resp.json()["esearchresult"] if resp is not None else None
    except (ValueError, KeyError):
        result = None
    if result is None:
        print(f"  ! PubMed search unavailable after {PUBMED_RETRIES} tries; skipping")
        return []
    if result.get("errorlist") or result.get("warninglist"):
        # A misspelt journal or field tag does not fail: PubMed quietly drops that
        # clause and answers the rest of the query. Surface it.
        print(f"  ! PubMed query warning: {result.get('errorlist') or result.get('warninglist')}")
    ids = result.get("idlist") or []
    if int(result.get("count") or 0) > len(ids):
        print(f"  ! {result['count']} matches but max_items is {budget}; raise it to see the rest")

    papers: List[Paper] = []
    for start in range(0, len(ids), PUBMED_BATCH):
        time.sleep(PUBMED_PAUSE_SEC)
        resp = _pubmed_request("efetch.fcgi", {
            "id": ",".join(ids[start:start + PUBMED_BATCH]), "retmode": "xml",
        })
        try:
            root = ET.fromstring(resp.content) if resp is not None else None
        except ET.ParseError:
            root = None
        if root is None:
            print(f"  ! PubMed fetch failed for records {start}-{start + PUBMED_BATCH}; "
                  f"continuing with {len(papers)} item(s)")
            continue

        for article in root.iter("PubmedArticle"):
            body = article.find("MedlineCitation/Article")
            if body is None:
                continue
            # A corrigendum repeats the original abstract, so it ranks exactly like
            # the paper it corrects.
            kinds = {k.text for k in body.findall("PublicationTypeList/PublicationType")}
            if kinds & {"Published Erratum", "Retraction of Publication",
                        "Expression of Concern"}:
                continue
            title = _pubmed_text(body.find("ArticleTitle")).rstrip(".")
            # Structured abstracts arrive as several labelled sections.
            summary = " ".join(
                _pubmed_text(part) for part in body.findall("Abstract/AbstractText")
            ).strip()
            # No abstract means an erratum, a commentary or a cover caption. With
            # only a title to embed these rank on noise, so leave them out.
            if not title or not summary:
                continue
            published = _pubmed_date(article)
            if published is None or published < cutoff:
                continue

            doi = next(
                (i.text.strip() for i in article.findall("PubmedData/ArticleIdList/ArticleId")
                 if i.get("IdType") == "doi" and i.text),
                "",
            )
            pmid = article.findtext("MedlineCitation/PMID", "")
            authors = []
            for a in body.findall("AuthorList/Author"):
                full = " ".join(
                    x for x in (a.findtext("ForeName"), a.findtext("LastName")) if x
                ).strip() or (a.findtext("CollectiveName") or "").strip()
                if full:
                    authors.append(full)

            paper = Paper(
                title=title,
                summary=summary,
                link=(f"https://doi.org/{doi}" if doi
                      else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                published=published,
                # The paper's own journal, so a topic search does not label a Nature
                # paper "PubMed: erythropoiesis".
                source=(body.findtext("Journal/ISOAbbreviation")
                        or body.findtext("Journal/Title") or feed["name"]),
                authors=authors,
            )
            if not is_excluded(paper):
                papers.append(paper)

    return papers


# Some publishers serve XML that feedparser's own fetcher rejects outright --
# Genome Research sends a duplicate "version" attribute and Nature Biotechnology an
# invalid token -- and feedparser then returns zero entries with bozo set, so those
# feeds silently contribute nothing. Refetching with requests and a browser
# User-Agent yields a document it will parse.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


def _parse_rss(url: str):
    """feedparser.parse with a requests fallback for feeds it cannot fetch."""
    parsed = feedparser.parse(url)
    if parsed.entries:
        return parsed
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": _BROWSER_UA,
                     "Accept": "application/rss+xml, application/xml, text/xml, */*"},
            timeout=45,
        )
        resp.raise_for_status()
        retry = feedparser.parse(resp.content)
        if retry.entries:
            print(f"  (recovered {len(retry.entries)} entries via browser User-Agent)")
            return retry
    except Exception as exc:
        print(f"  (fallback fetch failed: {exc})")
    return parsed


def fetch_feed(feed: Dict[str, Any]) -> List[Paper]:
    print(f"Fetching feed: {feed['name']}  "
          f"({feed.get('url') or textwrap.shorten(feed.get('term', ''), 80)})")

    if feed.get("type") in ("crossref", "pubmed"):
        fetch = fetch_crossref if feed["type"] == "crossref" else fetch_pubmed
        papers = fetch(feed)
        print(f"  -> kept {len(papers)} items (after exclusions; admission comes later)")
        return papers

    parsed = _parse_rss(feed["url"])
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=LOOKBACK_DAYS)

    papers: List[Paper] = []

    for entry in parsed.entries[:MAX_ITEMS_PER_FEED]:
        published = parse_datetime(entry)
        if published < cutoff:
            continue

        # Titles need the same cleaning summaries get: several feeds ship markup
        # like <i>Drosophila</i> or R<sup>2</sup> inside the title text.
        title = strip_html(entry.get("title", "")).strip()
        summary = clean_summary(strip_html(entry.get("summary", "")))
        link = entry.get("link", "").strip() or feed["url"]

        if not title:
            continue

        paper = Paper(
            title=title,
            summary=summary,
            link=link,
            published=published,
            source=feed["name"],
            authors=extract_authors(entry),
        )

        if not is_excluded(paper):
            papers.append(paper)

    print(f"  -> kept {len(papers)} items (after exclusions; admission comes later)")
    return papers


def rank_papers(papers: List[Paper]) -> List[Paper]:
    if not papers:
        return papers

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # all-MiniLM-L6-v2 defaults to 256 wordpieces, which truncated 75% of the
    # candidate abstracts (mean 119 tokens discarded) while leaving every seed
    # intact. That asymmetry biased every similarity score.
    model.max_seq_length = 512

    # Build canonical embedding
    # One direction per interest: average the seeds within a group, then score each
    # paper against whichever group it is closest to.
    groups: Dict[str, List[str]] = {}
    for i, c in enumerate(CANONICAL_PAPERS):
        key = c.get("group") or f"seed {i + 1}"
        groups.setdefault(key, []).append(c["title"] + ". " + c.get("summary", ""))

    flat = [t for texts in groups.values() for t in texts]
    seed_emb = model.encode(flat, convert_to_tensor=True)
    centroids, offset = [], 0
    for texts in groups.values():
        centroids.append(seed_emb[offset:offset + len(texts)].mean(dim=0))
        offset += len(texts)
    canon_emb = torch.stack(centroids)
    print("Seed groups: " + ", ".join(f"{k} ({len(v)})" for k, v in groups.items()))

    # Encode paper texts
    texts = [p.title + ". " + p.summary for p in papers]
    paper_emb = model.encode(texts, convert_to_tensor=True, batch_size=64)

    # Similarity to the CLOSEST group, not to the average of everything. One global
    # average outvotes any interest unlike the others: with this list the
    # cardiovascular seed sits 0.29-0.54 from its neighbours and 0.68 from the global
    # centroid, so cardiovascular papers were scored against a point dominated by the
    # foundation-model seeds.
    similarity_matrix = util.cos_sim(paper_emb, canon_emb)
    best = similarity_matrix.max(dim=1)
    scores = best.values.cpu().numpy().reshape(-1)
    group_names = list(groups)

    for p, s, group_index in zip(papers, scores, best.indices.tolist()):
        p.score = float(s)
        p.semantic_group = group_names[group_index]

    # Sort by score descending, then by recency
    papers_sorted = sorted(
        papers,
        key=lambda p: (p.score, p.published),
        reverse=True
    )
    return papers_sorted


def sort_for_profile(papers: List[Paper]) -> List[Paper]:
    """Order by similarity to the closest seed group, then recency."""
    return sorted(papers, key=lambda p: (p.score, p.published), reverse=True)


def format_paper_md(p: Paper) -> str:
    date_str = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
    score_str = f"{p.score:.3f}" if not math.isnan(p.score) else "n/a"
    summary = p.summary or "_No abstract/summary available._"
    interest = p.semantic_group or "unassigned"
    priority = ""

    return textwrap.dedent(f"""
    ### [{p.title}]({p.link})
    - Source: **{p.source}**  
      Date: {date_str} · Semantic score: {score_str} · Interest: {interest}{priority}

    {summary}
    """)


def build_markdown_digest(papers: List[Paper]) -> str:
    now = datetime.now(timezone.utc)
    header = (
        f"# Literature Digest\n\n"
        f"Generated on {now:%Y-%m-%d %H:%M UTC}\n"
    )
    intro = textwrap.dedent(f"""
    Time window: last {LOOKBACK_DAYS} days
    Feeds: {', '.join(f['name'] for f in FEEDS)}

    Ranked by the active profile's relevance rubric, with semantic similarity used
    for shortlisting and tie-breaking.
    """)

    body_parts = []
    for i, p in enumerate(papers, start=1):
        body_parts.append(f"---\n\n**#{i}**\n")
        body_parts.append(format_paper_md(p))

    return header + "\n" + intro + "\n\n" + "".join(body_parts)


def save_markdown(md: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    now = datetime.now(timezone.utc)
    fname = f"digest_{now:%Y-%m-%d}.md"
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved digest to {path}")
    return path

def build_html_digest(
    new_papers: List[Paper],
    prev_papers: List[Paper],
    history_papers: List[Paper],
) -> str:
    """Return a full HTML document string with two sections:
    - Today's Feed: new since last digest
    - Previous Feed: already seen in last digest

    Also embeds a machine-readable JSON blob of paper keys as an HTML comment
    so future runs can detect duplicates.
    """
    now = datetime.now(timezone.utc)

    # standalone inline CSS
    css = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
        max-width: 900px;
        margin: 2rem auto;
        padding: 0 1rem 3rem;
        line-height: 1.5;
        color: #111827;
        background-color: #f9fafb;
    }
    h1 {
        font-size: 1.8rem;
        margin-bottom: 0.25rem;
    }
    h2 {
        font-size: 1.25rem;
        margin-top: 1.75rem;
        margin-bottom: 0.75rem;
    }
    .meta {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 1.25rem;
    }
    .paper {
        background: #ffffff;
        margin: 0.75rem 0 1.0rem;
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(15,23,42,0.08);
    }
    .paper h3 {
        font-size: 1.05rem;
        margin: 0 0 0.25rem;
        font-weight: 650;
    }
    .paper a {
        color: #2563eb;
        text-decoration: none;
    }
    .paper a:hover {
        text-decoration: underline;
    }
    .paper .info {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .paper .summary {
        font-size: 0.95rem;
        white-space: pre-wrap;
    }
    .index {
        font-weight: 700;
        color: #4b5563;
        margin-bottom: 0.25rem;
    }
    .section-note {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }
    """

    header = f"""
    <h1>Literature Digest</h1>
    <div class="meta">
        Generated on {now:%Y-%m-%d %H:%M UTC}<br>
        Time window: last {LOOKBACK_DAYS} days<br>
        Feeds: {escape(", ".join(f["name"] for f in FEEDS))}
    </div>
    """

    def render_section(title: str, note: str, papers: List[Paper], start_index: int) -> str:
        if not papers:
            return f"""
            <h2>{escape(title)}</h2>
            <div class="section-note">{escape(note)}</div>
            <div class="meta">No items.</div>
            """

        blocks = [f"<h2>{escape(title)}</h2>"]
        blocks.append(f"<div class='section-note'>{escape(note)}</div>")

        for i, p in enumerate(papers, start=start_index):
            date_str = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
            score_str = f"{p.score:.3f}" if not math.isnan(p.score) else "n/a"
            summary = p.summary or "No abstract/summary available."
            interest = p.semantic_group or "unassigned"
            priority = ""

            blocks.append(f"""
            <div class="paper">
              <div class="index">#{i}</div>
              <h3><a href="{escape(p.link)}" target="_blank" rel="noopener noreferrer">
                  {escape(p.title)}
              </a></h3>
              <div class="info">
                Source: <strong>{escape(p.source)}</strong> ·
                Date: {escape(date_str)} ·
                Semantic score: {escape(score_str)} ·
                Interest: {escape(interest)}{escape(priority)}
              </div>
              <div class="summary">{escape(summary)}</div>
            </div>
            """)

        return "\n".join(blocks)

    # Build the two sections
    todays_html = render_section(
        title="Today's Feed",
        note="New since the most recent digest in the output directory.",
        papers=new_papers,
        start_index=1,
    )
    prev_html = render_section(
        title="Previous Feed",
        note="Items that also appeared in the most recent digest (overlap due to RSS windows / TOCs).",
        papers=prev_papers,
        start_index=len(new_papers) + 1,
    )

    # Embed keys so next run can read them back
    def _paper_to_dict(p: Paper) -> dict:
        return {
            "title": p.title,
            "summary": p.summary,
            "link": p.link,
            "published": p.published.astimezone(timezone.utc).isoformat(),
            "source": p.source,
            "score": p.score,
            "authors": p.authors,
            "semantic_group": p.semantic_group,
        }

    payload = {"papers": [_paper_to_dict(p) for p in history_papers]}

    # json.dumps escapes quotes and backslashes but not "<", ">" or "-", so an
    # abstract containing "-->" would terminate this comment early: raw feed text
    # would spill into the rendered page and _KEYS_BLOB_RE would stop matching,
    # losing the entire accumulated history on the next run. Neutralise the only
    # sequences that can end or confuse a comment. json.loads restores them.
    # Re-encode only the characters that can end or confuse an HTML comment, using
    # JSON's own \uXXXX escapes so json.loads gives back the exact original text.
    # Backslashes are deliberately left alone: escaping them here would corrupt
    # every \" and \\ that json.dumps already produced.
    keys_json = (
        json.dumps(payload)
        .replace("--", "\\u002d\\u002d")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    keys_blob = f"<!-- DIGEST_KEYS_JSON {keys_json} -->"

    body = header + todays_html + prev_html + keys_blob

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Literature Digest</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""
    return html

def _chip_ink(bg: str) -> str:
    """Readable text colour for a chip background, by relative luminance.

    The tag palette spans very dark (#3B3B3B) to fairly light (#E69F00); white text
    on the light end is unreadable, so pick per colour instead of guessing once.
    """
    try:
        r, g, b = (int(bg[i:i + 2], 16) / 255 for i in (1, 3, 5))
    except Exception:
        return "#ffffff"
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#1f2933" if lum > 0.6 else "#ffffff"


def _chips_html(paper: Paper, escape) -> str:
    """Coloured topic chips, inline-styled so they survive a stripped <style>."""
    chips = []
    for label, colour in tags_for(paper):
        chips.append(
            f'<span style="display:inline-block;background:{colour};'
            f'color:{_chip_ink(colour)};font-family:Helvetica,Arial,sans-serif;'
            f'font-size:10px;font-weight:700;padding:2px 6px;border-radius:3px;'
            f'margin:0 4px 4px 0;white-space:nowrap;">{escape(label)}</span>'
        )
    if not chips:
        return ""
    return f'<div style="margin:0 0 6px;line-height:1.6;">{"".join(chips)}</div>'


def build_email_digest(new_papers: List[Paper], prev_papers: List[Paper]) -> str:
    """Render the digest as an email rather than a web page.

    Differences from build_html_digest, all of them forced by mail clients:

    * No DIGEST_KEYS_JSON blob. That comment is ~95% of the on-disk file, and Gmail
      clips any message over roughly 102 KB, so mailing the archive copy means
      recipients see "[Message clipped]" instead of the digest.
    * Every rule is inline on the element. Gmail and Outlook.com drop or rewrite
      <style> in many contexts, and a class-only stylesheet degrades to unstyled
      black text.
    * Tables, not divs, with a fixed 600px shell -- Outlook's Word rendering engine
      has no flexbox and ignores width on <body>.
    * px units, weight 700, Helvetica/Arial ahead of the system stack, a solid
      border instead of box-shadow: each of the browser-only choices has an Outlook
      equivalent here.
    * Today's Feed is expanded; the previous 30 days sit in a <details> element as a
      compact one-line list. <details> only collapses in Apple Mail and Thunderbird
      -- Gmail rewrites the tags as <u></u>, Outlook.com and Yahoo strip them -- so
      it fails open, and the content has to stay short enough that failing open is
      still readable.
    """
    from html import escape

    now = datetime.now(timezone.utc)
    PAGE, CARD, LINE = "#f4f5f7", "#ffffff", "#e3e6ea"
    INK, MUTED, LINK = "#1f2933", "#6b7280", "#1d4ed8"
    FONT = "Helvetica,Arial,sans-serif"

    def card(p: Paper) -> str:
        date_str = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
        score_str = f"{p.score:.3f}" if not math.isnan(p.score) else "n/a"
        abstract = textwrap.shorten(
            p.summary or "No abstract available.",
            width=EMAIL_ABSTRACT_CHARS,
            placeholder="…",
        )
        interest = p.semantic_group or "unassigned"
        priority = ""
        who = format_authors(p.authors)
        author_row = (
            f'<div style="margin:0 0 3px;font-family:{FONT};font-size:12px;'
            f'color:#4b5563;">{escape(who)}</div>' if who else ""
        )
        return (
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0"'
            f' border="0" style="border-collapse:collapse;background:{CARD};'
            f'border:1px solid {LINE};border-radius:6px;margin:0 0 12px;">'
            f'<tr><td style="padding:14px 16px;">'
            f"{_chips_html(p, escape)}"
            f'<div style="margin:0 0 4px;font-family:{FONT};font-size:15px;'
            f'line-height:1.35;font-weight:700;">'
            f'<a href="{escape(p.link)}" style="color:{LINK};text-decoration:none;">'
            f"{escape(p.title)}</a></div>"
            f"{author_row}"
            f'<div style="margin:0 0 8px;font-family:{FONT};font-size:11px;'
            f'color:{MUTED};">{escape(p.source)} &middot; {escape(date_str)}'
            f" &middot; semantic score {escape(score_str)} &middot; {escape(interest)}"
            f"{priority}</div>"
            f'<div style="font-family:{FONT};font-size:13px;line-height:1.5;'
            f'color:{INK};">{escape(abstract)}</div>'
            f"</td></tr></table>"
        )

    def compact_row(p: Paper) -> str:
        date_str = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
        return (
            f'<tr><td style="padding:5px 0;border-bottom:1px solid #eef0f2;'
            f'font-family:{FONT};font-size:12px;line-height:1.45;">'
            f'<a href="{escape(p.link)}" style="color:{LINK};text-decoration:none;">'
            f"{escape(p.title)}</a>"
            f'<span style="color:{MUTED};"> &middot; {escape(p.source)}'
            f" &middot; {escape(date_str)}</span></td></tr>"
        )

    # Preheader: the grey snippet the inbox shows next to the subject. Left alone it
    # is whatever text happens to come first, which is the date line.
    lead = new_papers[0].title if new_papers else "No new papers today"
    preheader = (
        f'<div style="display:none;max-height:0;overflow:hidden;mso-hide:all;'
        f'font-size:1px;line-height:1px;color:{PAGE};">'
        f"{escape(f'{len(new_papers)} new: {lead}')}</div>"
    )

    if new_papers:
        today_block = "".join(card(p) for p in new_papers)
    else:
        today_block = (
            f'<div style="font-family:{FONT};font-size:13px;color:{MUTED};'
            f'padding:8px 0 16px;">Nothing new since the last digest.</div>'
        )

    if prev_papers:
        prev_block = (
            f"<details>"
            f'<summary style="font-family:{FONT};font-size:13px;font-weight:700;'
            f'color:{INK};cursor:pointer;padding:6px 0;">'
            f"Previous {LOOKBACK_DAYS} days &middot; {len(prev_papers)} papers"
            f"</summary>"
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0"'
            f' border="0" style="border-collapse:collapse;background:{CARD};'
            f'border:1px solid {LINE};border-radius:6px;margin:6px 0 0;">'
            f'<tr><td style="padding:8px 16px;">'
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0"'
            f' border="0" style="border-collapse:collapse;">'
            f"{''.join(compact_row(p) for p in prev_papers)}"
            f"</table></td></tr></table></details>"
        )
    else:
        prev_block = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="supported-color-schemes" content="light dark">
<title>Literature Digest {now:%Y-%m-%d}</title>
<style>
  /* Progressive enhancement only -- every rule above is already inline, so a
     client that drops this block loses nothing structural. */
  a:hover {{ text-decoration: underline !important; }}
  @media (max-width: 620px) {{
    .shell {{ width: 100% !important; }}
  }}
</style>
</head>
<body style="margin:0;padding:0;background:{PAGE};-webkit-text-size-adjust:100%;">
{preheader}
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"
       style="border-collapse:collapse;background:{PAGE};">
<tr><td align="center" style="padding:20px 10px 32px;">
<table role="presentation" class="shell" width="600" cellpadding="0" cellspacing="0"
       border="0" style="border-collapse:collapse;width:600px;max-width:600px;">
<tr><td>
  <div style="font-family:{FONT};font-size:20px;font-weight:700;color:{INK};
              padding:0 0 2px;">Literature Digest</div>
  <div style="font-family:{FONT};font-size:12px;color:{MUTED};padding:0 0 14px;">
    {now:%Y-%m-%d} &middot; {len(new_papers)} new &middot; last {LOOKBACK_DAYS} days
    &middot; {len(FEEDS)} sources
  </div>
  {today_block}
  {prev_block}
  <div style="font-family:{FONT};font-size:11px;color:{MUTED};padding:18px 0 0;
              border-top:1px solid {LINE};margin-top:8px;">
    Ranked by similarity to the closest seed group of canonical papers. Abstracts are
    truncated to {EMAIL_ABSTRACT_CHARS} characters &mdash; open a title for the full text.
  </div>
</td></tr>
</table>
</td></tr>
</table>
</body>
</html>
"""

def save_html(html: str, path: str | None = None) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if path is None:
        now = datetime.now(timezone.utc)
        fname = f"digest_{now:%Y-%m-%d}.html"
        path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved HTML digest to {path}")
    return path


def post_to_slack(papers: List[Paper]) -> None:
    if not SLACK_WEBHOOK_URL:
        print("No Slack webhook configured; skipping Slack notification.")
        return

    top = papers[:TOP_K_SLACK]
    if not top:
        print("No papers to post to Slack.")
        return

    lines = ["*literature digest – top hits*"]
    for p in top:
        date_str = p.published.astimezone(timezone.utc).strftime("%Y-%m-%d")
        score_str = f"{p.score:.3f}"
        line = f"• <{p.link}|{p.title}>  _(source: {p.source}, {date_str}, score {score_str})_"
        lines.append(line)

    text = "\n".join(lines)

    resp = requests.post(
        SLACK_WEBHOOK_URL,
        json={"text": text},
        timeout=10,
    )
    if resp.status_code // 100 == 2:
        print("Posted top papers to Slack.")
    else:
        print(f"Slack post failed: {resp.status_code} {resp.text}")


# ==========================
# ========= MAIN ===========
# ==========================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default=os.getenv("LIT_FEED_PROFILE", DEFAULT_PROFILE_NAME),
        help="Audience profile; non-default profiles use separate digest histories.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    profile = activate_profile(args.profile)
    print(
        f"Active profile: {profile['display_name']} ({args.profile}); "
        f"output: {OUTPUT_DIR}"
    )

    fetched: List[Paper] = []
    for feed in FEEDS:
        try:
            fetched.extend(fetch_feed(feed))
        except Exception as e:
            print(f"Error fetching {feed['name']}: {e}")

    dedup = {}
    for p in fetched:
        dedup.setdefault(paper_key(p), p)
    # LOCAL: then by title. The PubMed feeds overlap the RSS ones (a topic search
    # finds the Nature paper the Nature feed already carried) under a different
    # link, which paper_key treats as a different paper. First fetched wins.
    by_title = {}
    for p in dedup.values():
        by_title.setdefault(title_key(p) or paper_key(p), p)
    fetched = list(by_title.values())

    if not fetched:
        print("No papers found after filtering.")
        return

    # ranked = rank_papers(all_papers)
    # ranked = ranked[:TOP_K]

    # Load keys from most recent prior digest (if any)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_date = datetime.now(timezone.utc).date()

    # Path we will write to (overwrite if exists)
    today_path = os.path.join(OUTPUT_DIR, f"digest_{today}.html")

    # Use the most recent NON-today digest as the accumulated history baseline
    prev_path = most_recent_non_today_digest_path(OUTPUT_DIR)
    prev_papers = load_papers_from_html(prev_path) if prev_path else []

    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    prev_papers = [p for p in prev_papers if p.published >= cutoff]

    seen_keys = {paper_key(p) for p in prev_papers}

    # LOCAL: a paper already in the history under another link is the same paper,
    # not a new one. Drop the fresh copy so the stored one stays in Previous Feed
    # instead of being announced a second time.
    prev_by_title = {title_key(p): paper_key(p) for p in prev_papers if title_key(p)}
    fetched = [
        p for p in fetched
        if prev_by_title.get(title_key(p), paper_key(p)) == paper_key(p)
    ]

    yesterday_date = today_date - timedelta(days=1)

    # Merge today's fetched papers with accumulated previous papers, then dedup by key
    merged = {}
    # Cached judgments come last so their versioned LLM metadata survives a fresh
    # copy of the same feed entry. Only prev_papers contributes to seen_keys, so a
    # formerly rejected paper can still appear as new after a rubric revision.
    for p in (fetched + prev_papers):
        merged[paper_key(p)] = p
    merged_papers = list(merged.values())

    # Rank first, so admission can use the score.
    ranked = rank_papers(merged_papers)

    # ---- Admission and profile judgment -------------------------------------
    # Stage 1 is deliberately high recall: hard semantic floor, biological-domain
    # gate, then either keyword or semantic admission. Stage 2 applies the profile's
    # compound scientific intent to this much smaller set with one batched LLM call.
    prefiltered, by_keyword, by_score = [], 0, 0
    history_candidates, below_floor, outside_domain = 0, 0, 0
    for p in ranked:
        if math.isnan(p.score) or p.score < TODAY_MIN_SCORE:
            below_floor += 1
            continue
        if not matches_domain_keywords(p):
            outside_domain += 1
            continue
        seen = paper_key(p) in seen_keys
        kw = matches_include_keywords(p)
        sem = (SEMANTIC_ADMIT_SCORE > 0
               and not math.isnan(p.score)
               and p.score >= SEMANTIC_ADMIT_SCORE)
        if seen or kw or sem:
            prefiltered.append(p)
            history_candidates += bool(seen)
            by_keyword += bool(kw and not seen)
            by_score += bool(sem and not kw and not seen)

    failed_admission = (
        len(ranked) - len(prefiltered) - below_floor - outside_domain
    )
    admitted = prefiltered

    print(
        f"Prefiltered {len(prefiltered)} of {len(ranked)} ranked: "
        f"{history_candidates} from history, {by_keyword} new on keywords, "
        f"{by_score} new on similarity (>= {SEMANTIC_ADMIT_SCORE}); rejected "
        f"{below_floor} below the hard floor (< {TODAY_MIN_SCORE}), "
        f"{outside_domain} outside the biological domain, and "
        f"{failed_admission} by admission criteria."
    )

    # Everything rendered and marked "seen" uses only the accepted set. Rejected
    # decisions are cached separately so they avoid repeat API cost while remaining
    # eligible to surface as new after a rubric-version change.
    ranked = sort_for_profile(admitted)
    
    # Split AFTER ranking, but keep accumulated previous even if not in today's RSS
    new_items, prev_items = [], []
    has_history = bool(prev_papers)  # i.e., we found a prior digest

    for p in ranked:
        k = paper_key(p)

        # Already shown in an earlier digest -> Previous Feed.
        if k in seen_keys:
            prev_items.append(p)
            continue

        # Never shown before, so it is new *to the reader*. Deliberately not gated on
        # the publication date.
        #
        # The old rule required a paper to be both unseen AND published within the last
        # two days, which meant anything reaching us later than that matched neither
        # branch and was dropped from both sections -- while still being recorded in the
        # history blob, so it could never appear again. Measured on one real run that
        # silently discarded 114 papers, 40 of them above TODAY_MIN_SCORE. It hit every
        # source that does not post daily: preprints.org (Crossref registers its DOIs a
        # median of 3 days after posting), and any journal publishing by issue, where
        # a table of contents routinely surfaces papers a week or more old.
        #
        # On the very first run there is no history at all, so everything is unseen;
        # keep the backlog out of Today's Feed in that one case.
        if has_history or p.published.date() in (today_date, yesterday_date):
            new_items.append(p)
        else:
            prev_items.append(p)

    # Cap separately
    new_items = new_items[:TODAY_TOP_K]
    prev_items = prev_items[:PREV_TOP_K]

    # Keep only currently admitted papers in memory. In particular, a historical
    # paper that falls below the hard floor must not leak back into a later digest
    # merely because it appeared before the threshold changed.
    history_papers = ranked

    html = build_html_digest(new_items, prev_items, history_papers)

    # overwrite today's file (your existing overwrite logic)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_path = os.path.join(OUTPUT_DIR, f"digest_{today}.html")
    save_html(html, path=today_path)

    # Second, email-shaped copy. The archive file above carries the history blob and
    # is what the next run reads back; this one is the version to mail.
    email_path = os.path.join(OUTPUT_DIR, f"digest_{today}.email.html")
    save_html(build_email_digest(new_items, prev_items), path=email_path)

    # For Slack: usually you want only the NEW items; change if you want both.
    post_to_slack(new_items if new_items else ranked)

if __name__ == "__main__":
    main()
