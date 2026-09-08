#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MDD text-mining pipeline CLI.

Examples:
  python run_pipeline.py corpus --mode update
  python run_pipeline.py ner --mode full
  python run_pipeline.py associations --max-docs 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Handle Windows console encoding gracefully
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from mdd_pipeline.config_utils import load_config
from mdd_pipeline.corpus import run_corpus_stage
from mdd_pipeline.ner import run_ner_stage
from mdd_pipeline.associations import run_associations_stage
from mdd_pipeline.normalize import run_normalize_stage
from mdd_pipeline.unique_associations import run_unique_associations_stage
from mdd_pipeline.graphs import run_graphs_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MDD text-mining pipeline (corpus -> NER -> associations -> export)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: pipeline/config.yaml)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="Download / update PubMed titles+abstracts")
    corpus.add_argument(
        "--mode",
        choices=["update", "full"],
        default=None,
        help="update = append only new PMIDs; full = rebuild TSV (default from config)",
    )
    corpus.add_argument(
        "--query",
        default=None,
        help="Custom PubMed/Entrez query (overrides MeSH terms in config.yaml)",
    )
    corpus.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="NCBI Entrez API key (overrides NCBI_API_KEY env and config)",
    )
    corpus.add_argument("--start-year", type=int, default=None)
    corpus.add_argument("--end-year", type=int, default=None)
    corpus.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Stop after N newly written records (useful for smoke tests)",
    )
    corpus.add_argument(
        "--out-tsv",
        type=Path,
        default=None,
        help="Override corpus TSV path",
    )
    corpus.add_argument(
        "--out-metadata",
        type=Path,
        default=None,
        help="Override corpus metadata JSON path",
    )

    ner = sub.add_parser(
        "ner",
        help="Named entity recognition: HunFlair2 + miRNA + DeprTerms",
    )
    ner.add_argument(
        "--mode",
        choices=["update", "full"],
        default=None,
        help="full = rewrite entities TSV; update = append PMIDs not yet in entities TSV",
    )
    ner.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Process at most N documents (smoke test)",
    )
    ner.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )
    ner.add_argument(
        "--corpus-tsv",
        type=Path,
        default=None,
        help="Override input corpus TSV",
    )
    ner.add_argument(
        "--out-tsv",
        type=Path,
        default=None,
        help="Override entities TSV path",
    )
    ner.add_argument(
        "--env-json",
        type=Path,
        default=None,
        help="Override Env_factors.json path",
    )

    assoc = sub.add_parser(
        "associations",
        help="Extract entity-trigger-entity associations from NER + rules",
    )
    assoc.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Process at most N corpus documents (smoke test)",
    )
    assoc.add_argument("--corpus-tsv", type=Path, default=None)
    assoc.add_argument("--entities-tsv", type=Path, default=None)
    assoc.add_argument("--rules-txt", type=Path, default=None)
    assoc.add_argument(
        "--out-tsv",
        type=Path,
        default=None,
        help="Override associations TSV path",
    )

    norm = sub.add_parser(
        "normalize",
        help="Normalize association entities to primary DB IDs (+ catalog metadata)",
    )
    norm.add_argument(
        "--max-mentions",
        type=int,
        default=None,
        help="Normalize at most N unique mentions (smoke test)",
    )
    norm.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="NCBI API key for Gene/Taxonomy",
    )
    norm.add_argument(
        "--associations-tsv",
        type=Path,
        default=None,
        help="Input associations TSV (default from config)",
    )
    norm.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Delay between HTTP/Entrez calls (seconds)",
    )
    norm.add_argument(
        "--types",
        default=None,
        help="Comma-separated entity types to normalize (e.g. Species,Gene)",
    )

    uniq = sub.add_parser(
        "unique-associations",
        help="Aggregate unique associations and compute confidence scores",
    )
    uniq.add_argument(
        "--associations-tsv",
        type=Path,
        default=None,
        help="Input associations_normalized.tsv (default: outputs/normalization/...)",
    )
    uniq.add_argument(
        "--out-tsv",
        type=Path,
        default=None,
        help="Output unique_associations.tsv",
    )

    graphs = sub.add_parser(
        "graphs",
        help="Build XGMML (Cytoscape) + HTML graphs from unique associations",
    )
    graphs.add_argument(
        "--predicates",
        default=None,
        help="Comma-separated predicates to include (exact Predicate strings)",
    )
    graphs.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Limit number of nodes (greedy by ConfidenceScore)",
    )
    graphs.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Keep associations with ConfidenceScore >= this value",
    )
    graphs.add_argument(
        "--normalized-only",
        action="store_true",
        help="Keep only associations where both entities have DB IDs",
    )
    graphs.add_argument(
        "--keep-multi-edges",
        action="store_true",
        help="Keep all predicates between the same node pair (default: one best-CS edge)",
    )
    graphs.add_argument(
        "--unique-tsv",
        type=Path,
        default=None,
        help="Input unique_associations.tsv",
    )
    graphs.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/graphs)",
    )
    graphs.add_argument(
        "--stem",
        default="associations_graph",
        help="Output filename stem (default: associations_graph)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "corpus":
        summary = run_corpus_stage(
            cfg,
            mode=args.mode,
            api_key=args.api_key,
            start_year=args.start_year,
            end_year=args.end_year,
            max_records=args.max_records,
            corpus_tsv=args.out_tsv,
            metadata_path=args.out_metadata,
            query=args.query,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "ner":
        summary = run_ner_stage(
            cfg,
            mode=args.mode,
            max_docs=args.max_docs,
            prefer_gpu=not args.cpu,
            corpus_tsv=args.corpus_tsv,
            entities_tsv=args.out_tsv,
            env_json=args.env_json,
        )
        return 0 if summary.get("n_docs_processed", 0) >= 0 else 1

    if args.command == "associations":
        run_associations_stage(
            cfg,
            max_docs=args.max_docs,
            corpus_tsv=args.corpus_tsv,
            entities_tsv=args.entities_tsv,
            rules_txt=args.rules_txt,
            associations_tsv=args.out_tsv,
        )
        return 0

    if args.command == "normalize":
        types = None
        if args.types:
            types = [t.strip() for t in args.types.split(",") if t.strip()]
        run_normalize_stage(
            cfg,
            api_key=args.api_key,
            associations_tsv=args.associations_tsv,
            max_mentions=args.max_mentions,
            sleep_s=args.sleep,
            entity_types=types,
        )
        return 0

    if args.command == "unique-associations":
        run_unique_associations_stage(
            cfg,
            associations_tsv=args.associations_tsv,
            out_tsv=args.out_tsv,
        )
        return 0

    if args.command == "graphs":
        preds = None
        if args.predicates:
            preds = [p.strip() for p in args.predicates.split(",") if p.strip()]
        run_graphs_stage(
            cfg,
            unique_tsv=args.unique_tsv,
            predicates=preds,
            max_nodes=args.max_nodes,
            min_confidence=args.min_confidence,
            normalized_only=args.normalized_only,
            keep_multi_edges=args.keep_multi_edges,
            out_dir=args.out_dir,
            stem=args.stem,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
