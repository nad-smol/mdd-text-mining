#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Semi-automatic DeprTerms (Env_factors) expansion.

Ranked candidate TSV for manual review (does not modify Env_factors).
Sources: MeSH entry/narrower terms; corpus phrases near depression mentions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from mdd_pipeline.config_utils import load_config, resolve_path

csv.field_size_limit(10_000_000)

# Curated MeSH roots relevant to environmental / psychosocial risk for MDD.
# Infections / specific diseases intentionally omitted (Disease NER covers them).
DEFAULT_MESH_ROOTS = [
    "Adverse Childhood Experiences",
    "Life Change Events",
    "Psychosocial Deprivation",
    "Social Isolation",
    "Social Support",
    "Poverty",
    "Unemployment",
    "Domestic Violence",
    "Child Abuse",
    "Child Neglect",
    "Bullying",
    "Bereavement",
    "Divorce",
    "Caregivers",
    "Occupational Stress",
    "Prejudice",
    "Emigration and Immigration",
    "Housing",
    "Air Pollution",
    "Sedentary Behavior",
    "Sleep Deprivation",
    "Alcohol Drinking",
    "Marijuana Use",
    "Postpartum Period",
    "Object Attachment",
    "Parenting",
    "Stress, Psychological",
]

MDD_HINT = re.compile(
    r"\b("
    r"major depressive disorder|major depression|unipolar depression|"
    r"depressive disorder|depression|mdd|depressive episode"
    r")\b",
    flags=re.IGNORECASE,
)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']*(?:\s+[A-Za-z][A-Za-z0-9\-']*){1,4}")

STOP_PHRASES = {
    "in this study",
    "the present study",
    "results showed",
    "we found",
    "et al",
    "major depressive disorder",
    "major depression",
    "depressive disorder",
    "depressive symptoms",
    "depressive episode",
}


def load_existing_variants(tsv_path: Path) -> Tuple[Set[str], List[re.Pattern]]:
    """Return lowercased variants and compiled patterns from Env_factors.tsv."""
    variants: Set[str] = set()
    patterns: List[re.Pattern] = []
    with tsv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            canonical = (row.get("CanonicalTerm") or "").strip()
            raw = row.get("Variants") or ""
            parts = [v.strip() for v in raw.split(";") if v.strip()]
            if canonical:
                parts = [canonical] + parts
            for v in parts:
                variants.add(v.lower())
                # simple escaped pattern for coverage checks
                esc = re.escape(v).replace(r"\ ", r"\s+")
                try:
                    patterns.append(re.compile(rf"\b{esc}\b", re.IGNORECASE))
                except re.error:
                    continue
    return variants, patterns


def covered_by_existing(text: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


# ---------------------------------------------------------------------------
# MeSH expansion
# ---------------------------------------------------------------------------

def _entrez_ready(email: str, api_key: Optional[str]):
    from Bio import Entrez

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    return Entrez


def mesh_uid_for_descriptor(Entrez, name: str) -> Optional[str]:
    handle = Entrez.esearch(db="mesh", term=f'{name}[MeSH Terms]', retmax=5)
    record = Entrez.read(handle)
    handle.close()
    ids = record.get("IdList") or []
    return ids[0] if ids else None


def mesh_entry_and_children(Entrez, uid: str, sleep_s: float) -> Tuple[str, List[str], List[str]]:
    """Return (descriptor_name, entry_terms, child_descriptor_names)."""
    handle = Entrez.efetch(db="mesh", id=uid, retmode="xml")
    records = Entrez.read(handle)
    handle.close()
    time.sleep(sleep_s)

    if not records:
        return "", [], []

    rec = records[0]
    # Structured MeSH fields when BioPython parsing succeeds
    name = ""
    entry_terms: List[str] = []
    children: List[str] = []

    # records from Entrez.read for mesh are often dict-like with DescriptorName / ConceptList
    try:
        dn = rec.get("DescriptorName") if isinstance(rec, dict) else None
        if isinstance(dn, dict):
            name = str(dn.get("String") or dn.get("", "") or "")
        elif isinstance(dn, str):
            name = dn
    except Exception:
        pass

    try:
        concepts = rec.get("ConceptList", []) if isinstance(rec, dict) else []
        if isinstance(concepts, dict):
            concepts = concepts.get("Concept", [])
        if not isinstance(concepts, list):
            concepts = [concepts]
        for concept in concepts:
            if not isinstance(concept, dict):
                continue
            terms = concept.get("TermList", [])
            if isinstance(terms, dict):
                terms = terms.get("Term", [])
            if not isinstance(terms, list):
                terms = [terms]
            for term in terms:
                if isinstance(term, dict):
                    s = term.get("String") or term.get("Term") or ""
                else:
                    s = str(term)
                s = str(s).strip()
                if s:
                    entry_terms.append(s)
    except Exception:
        pass

    # TreeNumber / SeeRelated: child expansion via esearch on tree
    return name.strip(), sorted(set(entry_terms)), children


def expand_mesh_roots(
    roots: List[str],
    email: str,
    api_key: Optional[str],
    sleep_s: float,
) -> List[Dict[str, str]]:
    Entrez = _entrez_ready(email, api_key)
    rows: List[Dict[str, str]] = []
    seen: Set[str] = set()

    for root in roots:
        try:
            uid = mesh_uid_for_descriptor(Entrez, root)
            time.sleep(sleep_s)
        except Exception as exc:
            rows.append(
                {
                    "Source": "mesh",
                    "Candidate": root,
                    "ParentMeSH": root,
                    "Score": "0",
                    "Notes": f"esearch failed: {exc}",
                }
            )
            continue
        if not uid:
            rows.append(
                {
                    "Source": "mesh",
                    "Candidate": root,
                    "ParentMeSH": root,
                    "Score": "0",
                    "Notes": "MeSH UID not found",
                }
            )
            continue

        # Fetch entry terms for the root itself
        try:
            name, entry_terms, _ = mesh_entry_and_children(Entrez, uid, sleep_s)
        except Exception as exc:
            rows.append(
                {
                    "Source": "mesh",
                    "Candidate": root,
                    "ParentMeSH": root,
                    "Score": "0",
                    "Notes": f"efetch failed: {exc}",
                }
            )
            continue

        label = name or root
        bundle = [label] + entry_terms

        # Narrower terms: search for terms that list this descriptor in the hierarchy
        try:
            handle = Entrez.esearch(
                db="mesh",
                term=f'"{root}"[MeSH Terms:noexp]',
                retmax=1,
            )
            # Use broader: all descriptors under the same tree via "{root}"[mh:noexp] is exact;
            # for children use: root[MeSH:noexp] is the term itself. Better: "{root}"[MeSH Terms]
            handle.close()
            handle = Entrez.esearch(
                db="mesh",
                term=f"{root}[MeSH Terms] NOT {root}[MeSH Terms:noexp]",
                retmax=80,
            )
            child_rec = Entrez.read(handle)
            handle.close()
            time.sleep(sleep_s)
            child_ids = child_rec.get("IdList") or []
        except Exception:
            child_ids = []

        for cid in child_ids[:40]:
            try:
                cname, centry, _ = mesh_entry_and_children(Entrez, cid, sleep_s)
            except Exception:
                continue
            if cname:
                bundle.append(cname)
            bundle.extend(centry)

        for cand in bundle:
            c = cand.strip()
            if len(c) < 4:
                continue
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "Source": "mesh",
                    "Candidate": c,
                    "ParentMeSH": root,
                    "Score": "1",
                    "Notes": f"entry/narrower under {label}",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Corpus phrase mining
# ---------------------------------------------------------------------------

def iter_corpus_texts(corpus_tsv: Path, max_docs: Optional[int]) -> Iterable[str]:
    with corpus_tsv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        n = 0
        for row in reader:
            title = (row.get("ArticleTitle") or "").strip()
            abstract = (row.get("Abstract") or "").strip()
            if abstract in {"", "-", "NA", "N/A"}:
                abstract = ""
            text = f"{title}. {abstract}".strip()
            if not text or text == ".":
                continue
            if not MDD_HINT.search(text):
                continue
            yield text
            n += 1
            if max_docs is not None and n >= max_docs:
                break


def mine_corpus_phrases(
    corpus_tsv: Path,
    existing_variants: Set[str],
    existing_patterns: List[re.Pattern],
    max_docs: Optional[int],
    min_count: int,
    window_chars: int = 80,
) -> List[Dict[str, str]]:
    counts: Counter = Counter()
    examples: Dict[str, str] = {}

    for text in iter_corpus_texts(corpus_tsv, max_docs):
        for m in MDD_HINT.finditer(text):
            start = max(0, m.start() - window_chars)
            end = min(len(text), m.end() + window_chars)
            window = text[start:end]
            for pm in TOKEN_RE.finditer(window):
                phrase = pm.group(0).strip()
                key = phrase.lower()
                if key in STOP_PHRASES:
                    continue
                if key in existing_variants:
                    continue
                if covered_by_existing(phrase, existing_patterns):
                    continue
                # drop very generic single-ish biomedical noise
                if len(phrase) < 8:
                    continue
                if phrase.lower().startswith(("the ", "and ", "with ", "from ", "this ", "that ")):
                    continue
                counts[key] += 1
                examples.setdefault(key, phrase)

    rows: List[Dict[str, str]] = []
    for key, score in counts.most_common():
        if score < min_count:
            break
        rows.append(
            {
                "Source": "corpus",
                "Candidate": examples[key],
                "ParentMeSH": "",
                "Score": str(score),
                "Notes": f"co-occurs near depression mention (window={window_chars})",
            }
        )
    return rows


def filter_novel(
    candidates: List[Dict[str, str]],
    existing_variants: Set[str],
    existing_patterns: List[re.Pattern],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for row in candidates:
        cand = row["Candidate"].strip()
        key = cand.lower()
        if key in seen or key in existing_variants:
            continue
        if covered_by_existing(cand, existing_patterns):
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand DeprTerms candidates for manual review")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--skip-mesh", action="store_true")
    parser.add_argument("--skip-corpus", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit corpus docs (smoke tests)")
    parser.add_argument("--min-count", type=int, default=15, help="Min corpus phrase frequency")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output candidates TSV (default: ../outputs/deprterms/deprterms_candidates.tsv)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    email = (cfg.get("ncbi") or {}).get("email") or "anonymous@example.com"
    api_key = args.api_key or os.environ.get("NCBI_API_KEY") or (cfg.get("ncbi") or {}).get("api_key") or None
    sleep_s = float((cfg.get("ncbi") or {}).get("sleep_between_requests") or 0.34)

    paths = cfg.get("paths") or {}
    if paths.get("env_factors_tsv"):
        tsv_path = resolve_path(cfg, "env_factors_tsv")
    else:
        tsv_path = PIPELINE_DIR / "data" / "Env_factors.tsv"
    if not tsv_path.is_file():
        raise SystemExit(f"Env_factors TSV not found: {tsv_path}")
    existing_variants, existing_patterns = load_existing_variants(tsv_path)

    corpus_tsv = resolve_path(cfg, "corpus_tsv")
    out_path = args.out or (PIPELINE_DIR / "../outputs/deprterms/deprterms_candidates.tsv").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, str]] = []

    if not args.skip_mesh:
        print(f"MeSH expansion from {len(DEFAULT_MESH_ROOTS)} roots...")
        mesh_rows = expand_mesh_roots(DEFAULT_MESH_ROOTS, email=email, api_key=api_key, sleep_s=sleep_s)
        all_rows.extend(mesh_rows)
        print(f"  MeSH raw candidates: {len(mesh_rows)}")

    if not args.skip_corpus:
        print(f"Corpus mining: {corpus_tsv}")
        corpus_rows = mine_corpus_phrases(
            corpus_tsv,
            existing_variants,
            existing_patterns,
            max_docs=args.max_docs,
            min_count=args.min_count,
        )
        all_rows.extend(corpus_rows)
        print(f"  Corpus raw candidates: {len(corpus_rows)}")

    novel = filter_novel(all_rows, existing_variants, existing_patterns)
    # Sort: corpus first (by score), then mesh
    def sort_key(r: Dict[str, str]):
        src_rank = 0 if r["Source"] == "corpus" else 1
        try:
            score = int(r["Score"])
        except ValueError:
            score = 0
        return (src_rank, -score, r["Candidate"].lower())

    novel.sort(key=sort_key)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Source", "Candidate", "ParentMeSH", "Score", "Notes", "Decision"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in novel:
            row = dict(row)
            row["Decision"] = ""  # keep / reject / merge-into:<CanonicalTerm>
            writer.writerow(row)

    meta = {
        "n_existing_variants": len(existing_variants),
        "n_candidates": len(novel),
        "skip_mesh": args.skip_mesh,
        "skip_corpus": args.skip_corpus,
        "max_docs": args.max_docs,
        "min_count": args.min_count,
        "out": str(out_path),
    }
    meta_path = out_path.with_name("deprterms_candidates_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(novel)} novel candidates -> {out_path}")
    print("Fill Decision column: keep | reject | merge-into:<CanonicalTerm>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
