#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load association rules (6-column clean format)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Rule:
    entity_pair: str  # e.g. Chemical-Disease
    type_a: str
    type_b: str
    predicate: str
    stem: str
    direction: str  # -> | <- | <->
    trigger: str  # surface form (lowercase)
    trigger_re: re.Pattern


def _pair_types(entity_pair: str) -> Tuple[str, str]:
    parts = entity_pair.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Bad EntityPair: {entity_pair!r}")
    return parts[0], parts[1]


def _trigger_regex(trigger: str) -> re.Pattern:
    # Flexible whitespace inside multi-word triggers; word boundaries outside.
    parts = [re.escape(p) for p in trigger.split() if p]
    if not parts:
        raise ValueError(f"Empty trigger: {trigger!r}")
    body = r"\s+".join(parts)
    return re.compile(rf"(?<![A-Za-z0-9_]){body}(?![A-Za-z0-9_])", re.IGNORECASE)


def load_rules(path: Path) -> List[Rule]:
    rules: List[Rule] = []
    seen: set[Tuple[str, str, str]] = set()  # pair, predicate, trigger
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 6:
                raise ValueError(f"{path}:{line_no}: expected 6 columns, got {len(cols)}")
            entity_pair, _group, predicate, stem, forms_raw, direction = cols[:6]
            direction = direction.strip()
            if direction not in {"->", "<-", "<->"}:
                raise ValueError(f"{path}:{line_no}: bad Direction {direction!r}")
            type_a, type_b = _pair_types(entity_pair.strip())
            for form in forms_raw.split(","):
                trigger = form.strip().lower()
                if not trigger:
                    continue
                key = (entity_pair, predicate, trigger)
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    Rule(
                        entity_pair=entity_pair.strip(),
                        type_a=type_a,
                        type_b=type_b,
                        predicate=predicate.strip(),
                        stem=stem.strip(),
                        direction=direction,
                        trigger=trigger,
                        trigger_re=_trigger_regex(trigger),
                    )
                )
    # Longest triggers first for overlapping matches
    rules.sort(key=lambda r: (-len(r.trigger), r.entity_pair, r.predicate, r.trigger))
    return rules


def index_rules_by_pair(rules: List[Rule]) -> Dict[str, List[Rule]]:
    """Index by 'Chemical-Disease' and reverse 'Disease-Chemical'."""
    out: Dict[str, List[Rule]] = {}
    for r in rules:
        out.setdefault(r.entity_pair, []).append(r)
        rev = f"{r.type_b}-{r.type_a}"
        if rev != r.entity_pair:
            out.setdefault(rev, []).append(r)
    return out
