#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge approved pair drafts into pipeline/data/rules.txt.

Drafts: Predicate | Direction | Wordforms | Notes (Direction relative to A-B).
Output: EntityPair | Group | Concept | Stem | Wordforms | Direction.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANNING = ROOT / "planning"
DATA = ROOT / "pipeline" / "data"
OUT_MERGED = DATA / "rules.txt"

# Entity pairs included in the merged rules file
PAIRS = [
    ("rules_Chemical-Disease", "Chemical-Disease"),
    ("rules_Disease-Disease", "Disease-Disease"),
    ("rules_Disease-Gene", "Disease-Gene"),
    ("rules_Chemical-Gene", "Chemical-Gene"),
    ("rules_Gene-Gene", "Gene-Gene"),
    ("rules_Chemical-Chemical", "Chemical-Chemical"),
    ("rules_Species-Disease", "Species-Disease"),
    ("rules_Chemical-Species", "Chemical-Species"),
    ("rules_DeprTerms-Disease", "DeprTerms-Disease"),
    ("rules_DeprTerms-Gene", "DeprTerms-Gene"),
    ("rules_DeprTerms-Chemical", "DeprTerms-Chemical"),
    ("rules_miRNA-Gene", "miRNA-Gene"),
    ("rules_miRNA-Disease", "miRNA-Disease"),
    ("rules_Chemical-miRNA", "Chemical-miRNA"),
    ("rules_Species-Species", "Species-Species"),
    ("rules_Species-Gene", "Species-Gene"),
]

STEM_FALLBACK = {
    "is used in therapy of": "therapy",
    "prevents": "prevent",
    "is investigated regarding": "investigate",
    "has beneficial effect on": "beneficial",
    "worsens": "worsen",
    "has no effect on": "no-effect",
    "may cause": "may-cause",
    "induces": "induce",
    "increases risk of": "increase-risk",
    "decreases risk of": "decrease-risk",
    "is a biomarker of": "biomarker",
    "is comorbid with": "comorbid",
    "leads to": "leads-to",
    "manifests as": "manifests",
    "is part of": "part-of",
    "mimics": "mimic",
    "is associated with susceptibility to": "susceptibility",
    "contributes to": "contributes",
    "is a therapeutic target in": "therapeutic-target",
    "alters level of": "alters-level",
    "positively regulates": "pos-reg",
    "negatively regulates": "neg-reg",
    "interacts with": "interacts",
    "is substrate of": "substrate",
    "increases level of": "inc-level",
    "decreases level of": "dec-level",
    "affects": "affects",
    "is coadministered with": "coadmin",
    "acts synergistically with": "synergy",
    "acts antagonistically with": "antagonism",
    "is transformed into": "biotransform",
    "is a prodrug of": "prodrug",
    "causes": "causes",
    "is associated with": "associated",
    "may confer resistance in": "resistance",
    "is a risk factor for": "risk-factor",
    "precipitates": "precipitates",
    "modulates expression of": "mod-expr",
    "modulates response to": "mod-response",
    "regulates": "regulates",
    "regulates pathogenesis of": "reg-pathogenesis",
    "coinfects with": "coinfect",
    "induces expression of": "ind-expr",
}


def forms_from_draft(forms_raw: str) -> list[str]:
    forms = [x.strip() for x in forms_raw.split(";") if x.strip()]
    seen: set[str] = set()
    uniq: list[str] = []
    for form in forms:
        key = form.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(form)
    return uniq


def stem_for(pred: str) -> str:
    if pred in STEM_FALLBACK:
        return STEM_FALLBACK[pred]
    return pred.split()[0].lower().replace("/", "-")


def load_pair(draft_stem: str, entity_pair: str) -> list[str]:
    path = PLANNING / f"{draft_stem}_draft.tsv"
    if not path.is_file():
        raise SystemExit(f"Missing draft: {path}")
    rows: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "Direction" not in (reader.fieldnames or []):
            raise SystemExit(f"{path.name}: missing Direction column")
        for row in reader:
            pred = (row.get("Predicate") or "").strip()
            direction = (row.get("Direction") or "").strip()
            forms_raw = (row.get("Wordforms") or "").strip()
            if not pred or not forms_raw:
                continue
            if direction not in {"->", "<-", "<->"}:
                raise SystemExit(
                    f"{path.name}: bad Direction {direction!r} for {pred!r} "
                    f"(expected ->, <-, or <->)"
                )
            uniq = forms_from_draft(forms_raw)
            line = "\t".join(
                [
                    entity_pair,
                    pred,
                    pred,
                    stem_for(pred),
                    ", ".join(uniq),
                    direction,
                ]
            )
            rows.append(line)
    pair_out = DATA / f"{draft_stem}.txt"
    pair_out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"  {entity_pair}: {len(rows)} predicates -> {pair_out.name}")
    return rows


def main() -> None:
    all_rows: list[str] = []
    print("Building rules from drafts:")
    for draft_stem, entity_pair in PAIRS:
        all_rows.extend(load_pair(draft_stem, entity_pair))
    OUT_MERGED.write_text("\n".join(all_rows) + "\n", encoding="utf-8")
    print(f"Merged {len(all_rows)} rows -> {OUT_MERGED}")


if __name__ == "__main__":
    main()
