#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
miRNA mention detection and canonicalization.

Canonical form (miRBase-like): optional species prefix, mir-/let- stem, id,
optional letter/paralog/5p|3p. Bare class words (miRNA, microRNAs) are not entities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple


# miRBase organism prefixes commonly seen in biomed abstracts
SPECIES = (
    "hsa|mmu|rno|ptr|bta|ssc|gga|dre|cel|dme|ath|oar|eca|cfa|mml|age|ppy|"
    "ocu|cgr|tgu|xtr|xla|osa|zma|sly"
)

# Mature / precursor id: 21, 125b, 125b-1, 9-5, ...
_ID = r"\d{1,3}[a-z]?(?:-\d{1,2})?"

# Synonyms for the "miR" prefix
_MIR_PREFIX = r"(?:micro[\s\-]?RNA|miRNA|miR|mir)"

RE_MIRNA = re.compile(
    rf"""
    \b
    (?:
        # --- species-prefixed or bare miR / microRNA / miRNA ---
        (?:(?P<sp1>{SPECIES})-)?
        (?P<pre>{_MIR_PREFIX})
        [\s\-]?
        (?P<id1>{_ID})
        (?:-(?P<arm1>[35]p))?
      |
        # --- let-7 family (special stem, not "miR-7") ---
        (?:(?P<sp2>{SPECIES})-)?
        let-?
        (?P<id2>7[a-z]?(?:-\d{{1,2}})?)
        (?:-(?P<arm2>[35]p))?
    )
    \b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class MirnaMention:
    start: int
    end: int
    text: str
    canonical: str


def _norm_species(sp: Optional[str]) -> str:
    return (sp or "").lower()


def _norm_id(raw: str) -> str:
    # digits stay; trailing letter lowercased; paralog suffix kept
    return raw.lower()


def _norm_arm(arm: Optional[str]) -> str:
    return (arm or "").lower()


def canonicalize_mirna(text: str) -> Optional[str]:
    """
    Normalize a miRNA surface string to canonical form.
    Returns None if the string is not a specific miRNA mention.
    """
    m = RE_MIRNA.fullmatch(text.strip())
    if not m:
        # Retry after normalizing whitespace / unicode dashes
        t = text.strip().replace("\u2013", "-").replace("\u2014", "-")
        t = re.sub(r"\s+", " ", t)
        m = RE_MIRNA.fullmatch(t)
    if not m:
        return None

    if m.group("id1") is not None:
        sp = _norm_species(m.group("sp1"))
        body = _norm_id(m.group("id1"))
        arm = _norm_arm(m.group("arm1"))
        core = f"mir-{body}" + (f"-{arm}" if arm else "")
        return f"{sp}-{core}" if sp else core

    # let-7 branch
    sp = _norm_species(m.group("sp2"))
    body = _norm_id(m.group("id2"))  # e.g. 7a, 7a-1
    arm = _norm_arm(m.group("arm2"))
    core = f"let-{body}" + (f"-{arm}" if arm else "")
    return f"{sp}-{core}" if sp else core


def find_mirnas(text: str) -> List[MirnaMention]:
    """Find non-overlapping miRNA mentions (leftmost-longest via finditer order)."""
    if not text:
        return []
    # Match on dash-normalized text; report spans on the cleaned copy
    cleaned = text.replace("\u2013", "-").replace("\u2014", "-")
    out: List[MirnaMention] = []
    for m in RE_MIRNA.finditer(cleaned):
        surface = m.group(0)
        canon = canonicalize_mirna(surface)
        if not canon:
            continue
        out.append(
            MirnaMention(
                start=m.start(),
                end=m.end(),
                text=surface,
                canonical=canon,
            )
        )
    return out


def iter_mirna_pairs(text: str) -> Iterator[Tuple[str, str]]:
    """Yield (surface, canonical) for each mention."""
    for hit in find_mirnas(text):
        yield hit.text, hit.canonical
