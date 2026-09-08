#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load DeprTerms / Env_factors dictionary patterns for NER."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def load_deprterms_patterns(json_path: Path) -> List[Dict[str, Any]]:
    """
    Compile regex patterns from Env_factors.json.

    Each item: canonical, env_group, env_subgroup, variant, regex
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    compiled: List[Dict[str, Any]] = []
    for canonical, info in data.items():
        env_group = info.get("env_group", "") or ""
        env_subgroup = info.get("env_subgroup", "") or ""
        for p in info.get("patterns", []) or []:
            pattern_text = p.get("pattern")
            if not pattern_text:
                continue
            compiled.append(
                {
                    "canonical": canonical,
                    "env_group": env_group,
                    "env_subgroup": env_subgroup,
                    "variant": p.get("variant", "") or "",
                    "regex": re.compile(pattern_text, flags=re.IGNORECASE),
                }
            )
    return compiled
