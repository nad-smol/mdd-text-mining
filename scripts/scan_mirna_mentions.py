#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic scan of corpus for miRNA mentions (not a pipeline stage)."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from mdd_pipeline.config_utils import load_config, resolve_path
from mdd_pipeline.mirna import find_mirnas

csv.field_size_limit(10_000_000)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan corpus for miRNA mentions")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional TSV of surface -> canonical counts",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    corpus = resolve_path(cfg, "corpus_tsv")
    out = args.out or (PIPELINE_DIR / "../outputs/mirna/mirna_mention_counts.tsv").resolve()

    surface_counts: Counter = Counter()
    canon_counts: Counter = Counter()
    n_docs = 0
    n_docs_with = 0

    with corpus.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            title = (row.get("ArticleTitle") or "").strip()
            abstract = (row.get("Abstract") or "").strip()
            if abstract in {"", "-", "NA", "N/A"}:
                abstract = ""
            text = f"{title}. {abstract}".strip()
            hits = find_mirnas(text)
            n_docs += 1
            if hits:
                n_docs_with += 1
                for h in hits:
                    surface_counts[h.text] += 1
                    canon_counts[h.canonical] += 1
            if args.max_docs is not None and n_docs >= args.max_docs:
                break

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["Canonical", "Surface", "SurfaceCount", "CanonicalTotal"],
            delimiter="\t",
        )
        w.writeheader()
        # one row per surface form
        for surface, scount in surface_counts.most_common():
            # re-derive canonical from first matching scan already counted
            # look up via a fresh canonicalize through find on the surface alone
            from mdd_pipeline.mirna import canonicalize_mirna

            canon = canonicalize_mirna(surface) or ""
            w.writerow(
                {
                    "Canonical": canon,
                    "Surface": surface,
                    "SurfaceCount": scount,
                    "CanonicalTotal": canon_counts.get(canon, 0),
                }
            )

    print(f"Docs scanned: {n_docs}")
    print(f"Docs with >=1 miRNA: {n_docs_with}")
    print(f"Unique surfaces: {len(surface_counts)}")
    print(f"Unique canonical: {len(canon_counts)}")
    print(f"Top canonical:")
    for canon, c in canon_counts.most_common(15):
        print(f"  {c:5d}  {canon}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
