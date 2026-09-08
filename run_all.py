#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MDD Text-Mining Master Workflow Runner.

This script manages the end-to-end literature mining pipeline:
  1. Inspects the current state of outputs and reports what is ready.
  2. Automatically recovers / rebuilds any missing stages if files were deleted.
  3. Supports incremental updates with new PubMed publications.
  4. Provides a fast demo mode for peer review / quick smoke tests.
  5. Exports Cytoscape networks (.xgmml) and interactive HTML graphs.

Usage:
  python run_all.py                  # Auto mode: inspects status, runs any missing stages
  python run_all.py --mode update    # Incremental update: fetches only new PubMed papers
  python run_all.py --mode demo      # Quick demo: runs end-to-end on 50 papers (~2 min)
  python run_all.py --mode graphs    # Regenerates all standard Cytoscape and HTML graphs
  python run_all.py --mode full      # Rebuilds the entire pipeline from scratch
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Handle Windows console encoding gracefully
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path
REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from mdd_pipeline.config_utils import (
    ensure_parent,
    get_ncbi_credentials,
    load_config,
    resolve_path,
)
from mdd_pipeline.corpus import run_corpus_stage
from mdd_pipeline.ner import run_ner_stage
from mdd_pipeline.associations import run_associations_stage
from mdd_pipeline.normalize import run_normalize_stage
from mdd_pipeline.unique_associations import run_unique_associations_stage
from mdd_pipeline.graphs import run_graphs_stage


# -----------------------------------------------------------------------------
# Color & Formatting Helpers
# -----------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty() and os.name != "nt" or "WT_SESSION" in os.environ

def _c(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def c_bold(text: str) -> str: return _c(text, "1")
def c_green(text: str) -> str: return _c(text, "32")
def c_cyan(text: str) -> str: return _c(text, "36")
def c_yellow(text: str) -> str: return _c(text, "33")
def c_red(text: str) -> str: return _c(text, "31")
def c_dim(text: str) -> str: return _c(text, "2")


def print_banner() -> None:
    sep = "=" * 76
    print(c_cyan(sep))
    print(c_bold("  MDD Text-Mining Pipeline: Automated Association & Knowledge Graph System"))
    print(c_dim("  Major Depressive Disorder literature mining, entity normalization, & graph export"))
    print(c_cyan(sep))


# -----------------------------------------------------------------------------
# Environment & Hardware Checks
# -----------------------------------------------------------------------------

def check_environment() -> Dict[str, Any]:
    env_info: Dict[str, Any] = {}
    py_ver = sys.version_info
    env_info["python"] = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    if py_ver < (3, 9):
        print(c_yellow(f"[!] Warning: Python 3.9+ is recommended (current: {env_info['python']})."))

    # Check PyTorch & CUDA
    try:
        import torch
        env_info["torch"] = torch.__version__
        env_info["cuda"] = torch.cuda.is_available()
        if env_info["cuda"]:
            env_info["device_name"] = torch.cuda.get_device_name(0)
        else:
            env_info["device_name"] = "CPU"
    except ImportError:
        env_info["torch"] = None
        env_info["cuda"] = False
        env_info["device_name"] = "CPU (torch not installed)"

    return env_info


def print_environment_status(env_info: Dict[str, Any], email: str, api_key: Optional[str]) -> None:
    dev = env_info["device_name"]
    gpu_badge = c_green(f"[GPU: {dev}]") if env_info.get("cuda") else c_yellow(f"[{dev}]")
    key_badge = c_green("[NCBI API key: set]") if api_key else c_yellow("[NCBI API key: unset (3 req/s)]")
    email_display = email if email and "example.com" not in email else c_yellow(email + " (please configure in config.yaml)")

    print(f"  * Runtime: Python {env_info['python']} {gpu_badge} {key_badge}")
    print(f"  * NCBI Entrez contact: {email_display}")
    print()


# -----------------------------------------------------------------------------
# Status Inspection
# -----------------------------------------------------------------------------

def get_file_info(p: Path) -> Tuple[bool, str]:
    if not p.is_file():
        return False, "missing"
    sz_bytes = p.stat().st_size
    if sz_bytes < 1024:
        sz_str = f"{sz_bytes} B"
    elif sz_bytes < 1024 * 1024:
        sz_str = f"{sz_bytes / 1024:.1f} KB"
    else:
        sz_str = f"{sz_bytes / (1024 * 1024):.1f} MB"
    return True, sz_str


def inspect_status(cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths_cfg = cfg["paths"]
    
    stages = {
        "rules": {
            "name": "Syntax rules & predicates",
            "path": resolve_path(cfg, "rules_txt"),
            "required_for": "associations",
        },
        "deprterms": {
            "name": "DeprTerms dictionary",
            "path": resolve_path(cfg, "env_factors_json"),
            "required_for": "ner",
        },
        "corpus": {
            "name": "PubMed Corpus TSV",
            "path": resolve_path(cfg, "corpus_tsv"),
            "meta": resolve_path(cfg, "corpus_metadata"),
        },
        "ner": {
            "name": "Named Entity Mentions TSV",
            "path": resolve_path(cfg, "entities_tsv"),
        },
        "associations": {
            "name": "Raw Associations TSV",
            "path": resolve_path(cfg, "associations_tsv"),
        },
        "normalize": {
            "name": "Normalized Associations TSV",
            "path": resolve_path(cfg, "normalize_dir") / "associations_normalized.tsv",
            "catalog": resolve_path(cfg, "normalize_dir") / "entity_catalog.tsv",
            "cache": resolve_path(cfg, "normalize_dir") / "normalize_cache.json",
        },
        "unique": {
            "name": "Unique Associations (Scored)",
            "path": resolve_path(cfg, "unique_associations_tsv"),
        },
        "graphs": {
            "name": "Exported Graphs (Cytoscape / HTML)",
            "path": resolve_path(cfg, "graphs_dir"),
        },
    }

    status: Dict[str, Any] = {}
    for key, info in stages.items():
        p = info["path"]
        exists, sz_str = get_file_info(p) if key != "graphs" else (p.is_dir() and any(p.iterdir()), "dir")
        status[key] = {
            "name": info["name"],
            "path": p,
            "exists": exists,
            "size": sz_str,
        }
    return status


def print_status_dashboard(status: Dict[str, Any]) -> None:
    print(c_bold("Current Pipeline Artifacts:"))
    for key, info in status.items():
        mark = c_green("[OK]") if info["exists"] else c_yellow("[--]")
        name = info["name"]
        size = c_dim(f"({info['size']})") if info["exists"] else c_yellow("(not found)")
        rel_path = info["path"].relative_to(REPO_DIR) if str(info["path"]).startswith(str(REPO_DIR)) else info["path"]
        print(f"  {mark} {name:<36} {size:<14} -> {rel_path}")
    print()


# -----------------------------------------------------------------------------
# Graph Generation Suite
# -----------------------------------------------------------------------------

def generate_standard_graphs(cfg: Dict[str, Any], out_dir: Optional[Path] = None) -> None:
    target_dir = out_dir or resolve_path(cfg, "graphs_dir")
    ensure_parent(target_dir / "_placeholder")

    print(c_cyan("\n[graphs] Generating standard research subgraphs (Cytoscape XGMML + interactive HTML)..."))

    graph_tasks = [
        {
            "stem": "fig05_therapy",
            "predicates": ["is used in therapy of"],
            "max_nodes": 80,
            "normalized_only": True,
            "min_confidence": None,
            "desc": "MDD Pharmacotherapy (antidepressants, lithium, ketamine)",
        },
        {
            "stem": "fig06_risk_factors",
            "predicates": ["increases risk of", "is a risk factor for", "contributes to"],
            "max_nodes": 80,
            "normalized_only": False,
            "min_confidence": None,
            "desc": "Environmental & Psychosocial Risk Factors (DeprTerms)",
        },
        {
            "stem": "fig07_comorbid",
            "predicates": ["is comorbid with"],
            "max_nodes": 80,
            "normalized_only": True,
            "min_confidence": None,
            "desc": "Comorbidity Landscape (Psychiatric & Somatic Multimorbidity)",
        },
        {
            "stem": "fig08_molecular",
            "predicates": [
                "is associated with susceptibility to",
                "is a biomarker of",
                "is a therapeutic target in",
                "alters level of",
            ],
            "max_nodes": 100,
            "normalized_only": True,
            "min_confidence": None,
            "desc": "Molecular Hubs & Biomarkers (BDNF, SLC6A4, COMT, FKBP5, cytokines)",
        },
        {
            "stem": "fig09_infections",
            "predicates": ["increases risk of", "causes", "leads to", "is associated with"],
            "max_nodes": 80,
            "normalized_only": False,
            "min_confidence": None,
            "desc": "Infectious & Microbial Species (HIV, HCV, SARS-CoV-2, microbiota)",
        },
        {
            "stem": "fig10_bdnf",
            "predicates": None,  # all predicates around BDNF
            "max_nodes": 60,
            "normalized_only": False,
            "min_confidence": 0.30,
            "desc": "BDNF Multimodal Subgraph (neurotrophin neighborhood)",
        },
        {
            "stem": "overview_high_confidence",
            "predicates": None,
            "max_nodes": 150,
            "normalized_only": True,
            "min_confidence": 0.35,
            "desc": "High-Confidence Multi-Domain Overview",
        },
    ]

    for task in graph_tasks:
        print(f"  * Exporting {task['stem']} ({task['desc']})...")
        run_graphs_stage(
            cfg,
            unique_tsv=resolve_path(cfg, "unique_associations_tsv"),
            predicates=task["predicates"],
            max_nodes=task["max_nodes"],
            min_confidence=task["min_confidence"],
            normalized_only=task["normalized_only"],
            out_dir=target_dir,
            stem=task["stem"],
        )
    print(c_green(f"[graphs] Completed! Interactive HTML and XGMML files saved to {target_dir}"))


# -----------------------------------------------------------------------------
# Quick Demo Runner
# -----------------------------------------------------------------------------

def run_demo_pipeline(cfg: Dict[str, Any], max_records: int = 50, api_key: Optional[str] = None) -> None:
    demo_dir = resolve_path(cfg, "outputs_dir") / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo_corpus = demo_dir / "demo_corpus.tsv"
    demo_meta = demo_dir / "demo_corpus_meta.json"
    demo_entities = demo_dir / "demo_entities.tsv"
    demo_assoc = demo_dir / "demo_associations.tsv"
    demo_norm_dir = demo_dir / "normalization"
    demo_unique = demo_dir / "demo_unique_associations.tsv"
    demo_graphs = demo_dir / "graphs"

    print(c_cyan(f"\n[DEMO] Starting fast smoke-test pipeline ({max_records} recent publications)..."))
    print(c_dim(f"All demo outputs will be saved to: {demo_dir}\n"))

    t0 = time.time()

    # Step 1: Download sample
    print(c_bold("[Step 1/5] Fetching 50 recent MDD PubMed abstracts..."))
    current_year = datetime.now().year
    run_corpus_stage(
        cfg,
        mode="full",
        api_key=api_key,
        start_year=current_year - 1,
        end_year=current_year,
        max_records=max_records,
        corpus_tsv=demo_corpus,
        metadata_path=demo_meta,
    )

    # Step 2: NER
    print(c_bold("\n[Step 2/5] Running NER (HunFlair2 + miRNA + DeprTerms)..."))
    run_ner_stage(
        cfg,
        mode="full",
        max_docs=max_records,
        corpus_tsv=demo_corpus,
        entities_tsv=demo_entities,
    )

    # Step 3: Associations
    print(c_bold("\n[Step 3/5] Extracting syntactic associations..."))
    run_associations_stage(
        cfg,
        corpus_tsv=demo_corpus,
        entities_tsv=demo_entities,
        associations_tsv=demo_assoc,
    )

    # Step 4: Unique Associations & Confidence
    print(c_bold("\n[Step 4/5] Computing confidence scores and unique associations..."))
    run_unique_associations_stage(
        cfg,
        associations_tsv=demo_assoc,
        out_tsv=demo_unique,
    )

    # Step 5: Graph Export
    print(c_bold("\n[Step 5/5] Generating Cytoscape XGMML and interactive HTML graph..."))
    run_graphs_stage(
        cfg,
        unique_tsv=demo_unique,
        out_dir=demo_graphs,
        stem="demo_graph",
        max_nodes=50,
    )

    elapsed = time.time() - t0
    print(c_green(f"\n[DEMO] Complete in {elapsed:.1f}s!"))
    print(c_bold("Demo outputs:"))
    print(f"  * Unique associations: {demo_unique}")
    print(f"  * Interactive graph:   {demo_graphs / 'demo_graph.html'}")
    print(c_cyan("Open demo_graph.html in your browser to explore the extracted network."))


# -----------------------------------------------------------------------------
# Incremental Update Workflow
# -----------------------------------------------------------------------------

def run_update_workflow(cfg: Dict[str, Any], api_key: Optional[str] = None) -> None:
    print(c_cyan("\n[UPDATE] Checking PubMed for newly published articles on MDD..."))

    corpus_path = resolve_path(cfg, "corpus_tsv")
    entities_path = resolve_path(cfg, "entities_tsv")
    assoc_path = resolve_path(cfg, "associations_tsv")
    norm_dir = resolve_path(cfg, "normalize_dir")
    norm_assoc = norm_dir / "associations_normalized.tsv"
    unique_path = resolve_path(cfg, "unique_associations_tsv")

    # Step 1: Update corpus
    print(c_bold("[1/5] Updating PubMed corpus (mode: update)..."))
    summary = run_corpus_stage(cfg, mode="update", api_key=api_key)
    n_added = summary.get("n_added", 0)

    if n_added == 0:
        print(c_green(f"\n[UPDATE] Corpus is already up to date ({summary.get('n_records', 0)} total records)."))
        print("No new PubMed records were found matching the MeSH query.")
        return

    print(c_green(f"[1/5] Added {n_added} new PubMed records to corpus."))

    # Step 2: Update NER
    print(c_bold("\n[2/5] Running NER on new records (mode: update)..."))
    run_ner_stage(cfg, mode="update", corpus_tsv=corpus_path, entities_tsv=entities_path)

    # Step 3: Re-extract associations
    print(c_bold("\n[3/5] Extracting associations from updated corpus & entities..."))
    run_associations_stage(
        cfg,
        corpus_tsv=corpus_path,
        entities_tsv=entities_path,
        associations_tsv=assoc_path,
    )

    # Step 4: Normalize (leverages local cache)
    print(c_bold("\n[4/5] Normalizing entities against PubChem, OLS4, and NCBI..."))
    run_normalize_stage(
        cfg,
        api_key=api_key,
        associations_tsv=assoc_path,
    )

    # Step 5: Recompute Unique Associations & Confidence Scores
    print(c_bold("\n[5/5] Re-aggregating unique associations and calculating confidence..."))
    input_assoc = norm_assoc if norm_assoc.is_file() else assoc_path
    run_unique_associations_stage(
        cfg,
        associations_tsv=input_assoc,
        out_tsv=unique_path,
    )

    # Regenerate graphs
    generate_standard_graphs(cfg)

    print(c_green("\n[UPDATE] Pipeline successfully updated! All new articles and associations integrated."))


# -----------------------------------------------------------------------------
# Automated Pipeline Recovery / Execution
# -----------------------------------------------------------------------------

def run_auto_or_full(cfg: Dict[str, Any], force_full: bool = False, api_key: Optional[str] = None) -> None:
    status = inspect_status(cfg)
    corpus_p = status["corpus"]["path"]
    ner_p = status["ner"]["path"]
    assoc_p = status["associations"]["path"]
    norm_p = status["normalize"]["path"]
    unique_p = status["unique"]["path"]

    # Check if everything is already complete
    if not force_full and status["unique"]["exists"]:
        print(c_green("[OK] Precomputed unique associations file is already present!"))
        print(f"     Location: {unique_p} ({status['unique']['size']})")
        
        # Check if graphs exist
        if not status["graphs"]["exists"]:
            print(c_yellow("[*] Graphs directory is empty. Generating standard graphs..."))
            generate_standard_graphs(cfg)
        else:
            print(c_green("     Interactive HTML and Cytoscape graphs are available in outputs/graphs/"))
        
        print("\n" + c_bold("Available actions:"))
        print(f"  * Run {c_cyan('python run_all.py --mode update')} to check PubMed for newly published articles.")
        print(f"  * Run {c_cyan('python run_all.py --mode graphs')} to re-export Cytoscape and HTML networks.")
        print(f"  * Run {c_cyan('python run_all.py --mode demo')} to test the pipeline on a 50-article sample.")
        print(f"  * Run {c_cyan('python run_all.py --mode full')} to rebuild everything from scratch.")
        return

    if force_full:
        print(c_yellow("\n[!] Force rebuild requested: recomputing all stages from scratch..."))
    else:
        print(c_yellow("\n[!] Missing artifacts detected. Automatically building required stages..."))

    t0 = time.time()

    # Stage 1: Corpus
    if force_full or not status["corpus"]["exists"]:
        print(c_bold("\n[Stage 1/5] Downloading PubMed Corpus for MDD..."))
        run_corpus_stage(cfg, mode="full" if force_full else "update", api_key=api_key)
    else:
        print(c_green(f"[Stage 1/5] Corpus TSV found ({status['corpus']['size']}). Skipping."))

    # Stage 2: NER
    if force_full or not status["ner"]["exists"]:
        print(c_bold("\n[Stage 2/5] Running Named Entity Recognition (HunFlair2 + miRNA + DeprTerms)..."))
        run_ner_stage(cfg, mode="full" if force_full else "update")
    else:
        print(c_green(f"[Stage 2/5] Entities TSV found ({status['ner']['size']}). Skipping."))

    # Stage 3: Associations
    if force_full or not status["associations"]["exists"]:
        print(c_bold("\n[Stage 3/5] Extracting syntactic sentence-level associations..."))
        run_associations_stage(cfg)
    else:
        print(c_green(f"[Stage 3/5] Associations TSV found ({status['associations']['size']}). Skipping."))

    # Stage 4: Normalize
    if force_full or not status["normalize"]["exists"]:
        print(c_bold("\n[Stage 4/5] Normalizing mentions to primary database IDs..."))
        run_normalize_stage(cfg, api_key=api_key)
    else:
        print(c_green(f"[Stage 4/5] Normalized associations found ({status['normalize']['size']}). Skipping."))

    # Stage 5: Unique associations
    if force_full or not status["unique"]["exists"]:
        print(c_bold("\n[Stage 5/5] Deduplicating and calculating confidence scores..."))
        input_tsv = norm_p if norm_p.is_file() else assoc_p
        run_unique_associations_stage(cfg, associations_tsv=input_tsv, out_tsv=unique_p)
    else:
        print(c_green(f"[Stage 5/5] Unique associations found ({status['unique']['size']}). Skipping."))

    # Graphs
    generate_standard_graphs(cfg)

    elapsed = time.time() - t0
    print(c_green(f"\n============================================================================"))
    print(c_bold(f"  Pipeline execution completed successfully in {elapsed / 60:.1f} minutes!"))
    print(c_green(f"============================================================================"))
    print(f"Deliverable table: {unique_p}")
    print(f"Network visuals:   {resolve_path(cfg, 'graphs_dir')}")


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MDD Text-Mining Master Workflow Runner (Smart auto-recovery, update, & export)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_all.py                  # Default: check status, build any missing stages
  python run_all.py --mode update    # Fetch new PubMed papers published since last run
  python run_all.py --mode demo      # Run fast smoke-test on 50 abstracts (~2 min)
  python run_all.py --mode graphs    # Export interactive HTML and Cytoscape .xgmml graphs
  python run_all.py --mode full      # Recompute all stages from scratch
""",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "update", "demo", "graphs", "full"],
        default="auto",
        help="Execution mode (default: auto)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="NCBI Entrez API key (overrides NCBI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--demo-records",
        type=int,
        default=50,
        help="Number of records to process in demo mode (default: 50)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to custom config.yaml (default: config.yaml in repo root)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_banner()

    cfg = load_config(args.config)
    email, api_key = get_ncbi_credentials(cfg, api_key_cli=args.api_key)
    env_info = check_environment()
    print_environment_status(env_info, email, api_key)

    status = inspect_status(cfg)
    print_status_dashboard(status)

    if args.mode == "demo":
        run_demo_pipeline(cfg, max_records=args.demo_records, api_key=api_key)
        return 0

    if args.mode == "graphs":
        if not status["unique"]["exists"]:
            print(c_red("[!] Cannot export graphs: unique_associations.tsv is missing."))
            print(f"    Run {c_cyan('python run_all.py')} first to build associations.")
            return 1
        generate_standard_graphs(cfg)
        return 0

    if args.mode == "update":
        run_update_workflow(cfg, api_key=api_key)
        return 0

    if args.mode == "full":
        run_auto_or_full(cfg, force_full=True, api_key=api_key)
        return 0

    # Default auto mode
    run_auto_or_full(cfg, force_full=False, api_key=api_key)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
