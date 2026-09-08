#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build Env_factors.json from Env_factors.tsv (EnvGroup, EnvSubgroup, CanonicalTerm, Variants)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List


PIPELINE_DIR = Path(__file__).resolve().parent.parent


def variant_to_pattern(variant: str) -> str:
    """Escape a surface form into a word-boundary regex (case-insensitive at match time)."""
    v = variant.strip()
    if not v:
        return ""
    # Normalize fancy dashes/spaces lightly for pattern building
    v = v.replace("\u2013", "-").replace("\u2014", "-")
    parts = re.split(r"(\W+)", v)
    out: List[str] = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"\W+", part):
            # collapse whitespace runs; escape other punctuation
            if part.isspace():
                out.append(r"\s+")
            else:
                out.append(re.escape(part))
        else:
            out.append(re.escape(part))
    body = "".join(out)
    # Allow flexible whitespace where we already put \s+
    return rf"\b{body}\b"


def load_tsv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"EnvGroup", "EnvSubgroup", "CanonicalTerm", "Variants"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(f"TSV must have columns {sorted(required)}; got {reader.fieldnames}")
        for row in reader:
            term = (row.get("CanonicalTerm") or "").strip()
            if not term:
                continue
            rows.append(row)
    return rows


def build_json(rows: List[Dict[str, str]]) -> Dict:
    data: Dict = {}
    for row in rows:
        canonical = row["CanonicalTerm"].strip()
        variants_raw = row.get("Variants") or ""
        variants = [v.strip() for v in variants_raw.split(";") if v.strip()]
        # Ensure canonical itself is searchable if not already listed
        if canonical not in variants:
            variants = [canonical] + variants
        patterns = []
        seen = set()
        for variant in variants:
            key = variant.lower()
            if key in seen:
                continue
            seen.add(key)
            pat = variant_to_pattern(variant)
            if not pat:
                continue
            patterns.append({"variant": variant, "pattern": pat})
        data[canonical] = {
            "env_group": (row.get("EnvGroup") or "").strip(),
            "env_subgroup": (row.get("EnvSubgroup") or "").strip(),
            "notes": (row.get("Notes") or "").strip(),
            "patterns": patterns,
        }
    return data


def write_flat_terms(rows: List[Dict[str, str]], path: Path) -> None:
    """One line per canonical term: variant1; variant2; ... (legacy seed format)."""
    lines: List[str] = []
    for row in rows:
        variants = [v.strip() for v in (row.get("Variants") or "").split(";") if v.strip()]
        canonical = row["CanonicalTerm"].strip()
        if canonical not in variants:
            variants = [canonical] + variants
        # de-dupe case-insensitively, keep order
        seen = set()
        uniq = []
        for v in variants:
            k = v.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(v)
        lines.append("; ".join(uniq))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Env_factors.json from TSV")
    parser.add_argument(
        "--tsv",
        type=Path,
        default=PIPELINE_DIR / "data" / "Env_factors.tsv",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PIPELINE_DIR / "data" / "Env_factors.json",
    )
    parser.add_argument(
        "--out-flat",
        type=Path,
        default=PIPELINE_DIR / "data" / "Terms-list-external-factors.txt",
        help="Also rewrite flat terms list (legacy format)",
    )
    parser.add_argument("--skip-flat", action="store_true")
    args = parser.parse_args()

    rows = load_tsv(args.tsv)
    data = build_json(rows)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(data)} canonical terms -> {args.out_json}")

    if not args.skip_flat:
        write_flat_terms(rows, args.out_flat)
        print(f"Wrote flat terms -> {args.out_flat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
