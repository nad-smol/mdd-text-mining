#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load and resolve pipeline configuration."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PIPELINE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PIPELINE_DIR / "config.yaml"


def load_config(path: Optional[Path | str] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path.resolve())
    cfg["_pipeline_dir"] = str(PIPELINE_DIR)
    return cfg


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve a paths.* entry relative to pipeline/ unless absolute."""
    raw = cfg["paths"][key]
    p = Path(raw)
    if p.is_absolute():
        return p
    return (PIPELINE_DIR / p).resolve()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_ncbi_credentials(cfg: Dict[str, Any], api_key_cli: Optional[str] = None) -> tuple[str, Optional[str]]:
    """
    Priority for API key: CLI flag > env NCBI_API_KEY > config.yaml.
    """
    email = (cfg.get("ncbi") or {}).get("email") or ""
    if not email:
        raise ValueError("ncbi.email must be set in config.yaml (NCBI Entrez requirement).")

    key = api_key_cli or os.environ.get("NCBI_API_KEY") or (cfg.get("ncbi") or {}).get("api_key") or None
    if isinstance(key, str) and not key.strip():
        key = None
    return email, key


def related_artifact_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Configured path strings for pipeline artifacts (portable across installs)."""
    keys = [
        "corpus_tsv",
        "corpus_metadata",
        "entities_tsv",
        "associations_tsv",
        "associations_by_pmid_dir",
        "stats_dir",
        "sif_dir",
        "graphs_dir",
        "env_factors_json",
        "rules_txt",
        "normalize_dir",
        "unique_associations_tsv",
    ]
    return {k: str(cfg["paths"][k]) for k in keys}


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
