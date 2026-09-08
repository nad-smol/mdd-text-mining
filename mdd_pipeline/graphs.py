#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build association graphs: XGMML (Cytoscape) and interactive HTML.

Optional filters: predicates, max nodes, min confidence, normalized endpoints only.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .config_utils import ensure_parent, resolve_path

csv.field_size_limit(10_000_000)

# Node fill: (normalized, not-normalized) per entity type
NODE_COLORS: Dict[str, Tuple[str, str]] = {
    "Chemical": ("#7EB6D9", "#B8D4E8"),
    "Disease": ("#E8A0A0", "#E8C8C8"),
    "Gene": ("#8FCB8F", "#C5DEC5"),
    "Species": ("#C4A8E0", "#D9CBE8"),
    "miRNA": ("#7EC8C3", "#B8DEDB"),
    "DeprTerms": ("#E0C07A", "#E8D9B0"),
}
NODE_COLOR_DEFAULT = ("#A8A8B8", "#D0D0D8")

# Edge styles (color, line_style, arrow_shape), cycled per predicate
EDGE_STYLE_PALETTE: List[Tuple[str, str, str]] = [
    ("#4A6FA5", "solid", "triangle"),
    ("#C45C26", "solid", "diamond"),
    ("#2E8B57", "dashed", "triangle"),
    ("#8B4513", "solid", "circle"),
    ("#6A5ACD", "dotted", "triangle"),
    ("#CD5C5C", "dashed", "diamond"),
    ("#20B2AA", "solid", "tee"),
    ("#DAA520", "dashed", "circle"),
    ("#708090", "solid", "triangle"),
    ("#D2691E", "dotted", "diamond"),
    ("#4682B4", "dashed", "tee"),
    ("#9932CC", "solid", "diamond"),
    ("#556B2F", "dotted", "triangle"),
    ("#B22222", "solid", "circle"),
    ("#5F9EA0", "dashed", "triangle"),
]


@dataclass
class GraphNode:
    id: str
    label: str
    etype: str
    db_id: str
    normalized: bool
    text_synonyms: str
    pref_name: str
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def fill(self) -> str:
        bright, dull = NODE_COLORS.get(self.etype, NODE_COLOR_DEFAULT)
        return bright if self.normalized else dull

    def width_height(self) -> Tuple[float, float]:
        # Soft rectangle sized from label length
        n = max(len(self.label), 4)
        w = min(42.0 + n * 7.2, 320.0)
        h = 36.0 if n <= 28 else 48.0
        if n > 45:
            h = 56.0
        return w, h


@dataclass
class GraphEdge:
    id: str
    source: str
    target: str
    predicate: str
    direction: str
    confidence: float
    n_sentences: int
    n_pmids: int
    support_pmids: str
    alt_predicates: str = ""

    @property
    def directed(self) -> bool:
        return self.direction in {"->", "<-"}

    @property
    def label(self) -> str:
        return f"{self.predicate}, {self.confidence:.3f}"

    def width(self, cs_min: float, cs_max: float) -> float:
        if cs_max <= cs_min:
            return 2.5
        t = (self.confidence - cs_min) / (cs_max - cs_min)
        return 1.2 + t * 7.0  # 1.2 .. 8.2


def _node_id(etype: str, db_id: str, label: str) -> str:
    if db_id:
        return f"{etype}|{db_id}"
    # stable id from type + preferred display label
    raw = f"{etype}|{label}".encode("utf-8")
    return f"{etype}|txt:{hashlib.md5(raw).hexdigest()[:12]}"


def _display_label(pref: str, synonyms: str, normalized: bool) -> str:
    if normalized and pref:
        return pref.strip()
    # First text synonym (original casing), else preferred name
    parts = [p.strip() for p in (synonyms or "").split(";") if p.strip()]
    if parts:
        return parts[0]
    return (pref or "").strip() or "?"


def _edge_style(predicate: str) -> Tuple[str, str, str]:
    h = int(hashlib.md5(predicate.encode("utf-8")).hexdigest(), 16)
    return EDGE_STYLE_PALETTE[h % len(EDGE_STYLE_PALETTE)]


def _load_catalog_extra(catalog_tsv: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """(EntityType, DbId) -> Extra dict + PreferredName/Synonyms/DbSource."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not catalog_tsv.is_file():
        return out
    with catalog_tsv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            etype = (row.get("EntityType") or "").strip()
            dbid = (row.get("DbId") or "").strip()
            if not etype or not dbid:
                continue
            try:
                extra = json.loads(row.get("Extra") or "{}")
            except Exception:
                extra = {}
            if not isinstance(extra, dict):
                extra = {}
            extra = dict(extra)
            extra["_DbSource"] = (row.get("DbSource") or "").strip()
            extra["_PreferredName"] = (row.get("PreferredName") or "").strip()
            extra["_DbSynonyms"] = (row.get("Synonyms") or "").strip()
            out[(etype, dbid)] = extra
    return out


def _load_and_filter_associations(
    unique_tsv: Path,
    *,
    predicates: Optional[Set[str]],
    normalized_only: bool,
    max_nodes: Optional[int],
    min_confidence: Optional[float] = None,
    keep_multi_edges: bool = False,
) -> Tuple[Dict[str, GraphNode], List[GraphEdge]]:
    rows: List[Dict[str, Any]] = []
    with unique_tsv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pred = (row.get("Predicate") or "").strip()
            if predicates and pred not in predicates:
                continue
            id1 = (row.get("Object1ID") or "").strip()
            id2 = (row.get("Object2ID") or "").strip()
            if normalized_only and (not id1 or not id2):
                continue
            try:
                cs = float(row.get("ConfidenceScore") or 0)
            except ValueError:
                cs = 0.0
            if min_confidence is not None and cs < min_confidence:
                continue
            row["_cs"] = cs
            rows.append(row)

    rows.sort(key=lambda r: -float(r["_cs"]))

    nodes: Dict[str, GraphNode] = {}
    selected: Set[str] = set()

    def try_add_node(etype: str, db_id: str, pref: str, syns: str) -> Optional[str]:
        normalized = bool(db_id)
        label = _display_label(pref, syns, normalized)
        nid = _node_id(etype, db_id, label)
        if nid in nodes:
            if syns and len(syns) > len(nodes[nid].text_synonyms):
                nodes[nid].text_synonyms = syns
            return nid
        if max_nodes is not None and len(selected) >= max_nodes:
            return None
        nodes[nid] = GraphNode(
            id=nid,
            label=label,
            etype=etype,
            db_id=db_id,
            normalized=normalized,
            text_synonyms=syns,
            pref_name=pref,
        )
        selected.add(nid)
        return nid

    accepted_rows: List[Dict[str, Any]] = []
    for row in rows:
        t1 = (row.get("Object1Type") or "").strip()
        t2 = (row.get("Object2Type") or "").strip()
        id1 = (row.get("Object1ID") or "").strip()
        id2 = (row.get("Object2ID") or "").strip()
        pref1 = row.get("Object1PrefName") or ""
        pref2 = row.get("Object2PrefName") or ""
        syn1 = row.get("Object1TextSynonyms") or ""
        syn2 = row.get("Object2TextSynonyms") or ""
        lab1 = _display_label(pref1, syn1, bool(id1))
        lab2 = _display_label(pref2, syn2, bool(id2))
        nid1 = _node_id(t1, id1, lab1)
        nid2 = _node_id(t2, id2, lab2)
        # Skip self-loops
        if nid1 == nid2:
            continue

        in1 = nid1 in selected
        in2 = nid2 in selected
        need = (0 if in1 else 1) + (0 if in2 else 1)
        if max_nodes is not None and len(selected) + need > max_nodes:
            if not (in1 and in2):
                continue

        s1 = try_add_node(t1, id1, pref1, syn1)
        s2 = try_add_node(t2, id2, pref2, syn2)
        if s1 is None or s2 is None:
            continue
        accepted_rows.append(row)

    edges: List[GraphEdge] = []
    edge_keys: Set[Tuple[str, str, str, str]] = set()
    # best edge per directed (src,tgt) when collapsing multi-predicate parallels
    pair_best: Dict[Tuple[str, str], GraphEdge] = {}
    pair_alts: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for row in accepted_rows:
        t1 = (row.get("Object1Type") or "").strip()
        t2 = (row.get("Object2Type") or "").strip()
        id1 = (row.get("Object1ID") or "").strip()
        id2 = (row.get("Object2ID") or "").strip()
        lab1 = _display_label(row.get("Object1PrefName") or "", row.get("Object1TextSynonyms") or "", bool(id1))
        lab2 = _display_label(row.get("Object2PrefName") or "", row.get("Object2TextSynonyms") or "", bool(id2))
        sid = _node_id(t1, id1, lab1)
        tid = _node_id(t2, id2, lab2)
        if sid not in nodes or tid not in nodes or sid == tid:
            continue
        pred = (row.get("Predicate") or "").strip()
        direction = (row.get("Direction") or "").strip()
        src, tgt = sid, tid
        if direction == "<-":
            src, tgt = tid, sid
        key = (src, tgt, pred, direction)
        if key in edge_keys:
            continue
        edge_keys.add(key)
        edge = GraphEdge(
            id=f"e{len(edge_keys)}",
            source=src,
            target=tgt,
            predicate=pred,
            direction=direction,
            confidence=float(row["_cs"]),
            n_sentences=int(float(row.get("n_sentences") or 0)),
            n_pmids=int(float(row.get("n_pmids") or 0)),
            support_pmids=(row.get("SupportPMIDs") or "")[:2000],
        )
        if keep_multi_edges:
            edges.append(edge)
        else:
            pair = (src, tgt)
            prev = pair_best.get(pair)
            if prev is None or edge.confidence > prev.confidence:
                if prev is not None:
                    pair_alts[pair].append(f"{prev.predicate} ({prev.confidence:.3f})")
                pair_best[pair] = edge
            else:
                pair_alts[pair].append(f"{edge.predicate} ({edge.confidence:.3f})")

    if not keep_multi_edges:
        edges = []
        for i, ((src, tgt), edge) in enumerate(sorted(pair_best.items()), start=1):
            alts = pair_alts.get((src, tgt)) or []
            edges.append(
                GraphEdge(
                    id=f"e{i}",
                    source=edge.source,
                    target=edge.target,
                    predicate=edge.predicate,
                    direction=edge.direction,
                    confidence=edge.confidence,
                    n_sentences=edge.n_sentences,
                    n_pmids=edge.n_pmids,
                    support_pmids=edge.support_pmids,
                    alt_predicates="; ".join(alts[:12]),
                )
            )

    return nodes, edges


def _attach_catalog(
    nodes: Dict[str, GraphNode],
    catalog_extra: Dict[Tuple[str, str], Dict[str, Any]],
) -> None:
    for n in nodes.values():
        if not n.db_id:
            continue
        extra = catalog_extra.get((n.etype, n.db_id))
        if extra:
            n.extra = extra
            if not n.pref_name and extra.get("_PreferredName"):
                n.pref_name = str(extra["_PreferredName"])
                if n.normalized:
                    n.label = n.pref_name


# ---------------------------------------------------------------------------
# XGMML
# ---------------------------------------------------------------------------


def _esc(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _truncate_synonyms(syns: str, max_items: int = 12, max_chars: int = 400) -> str:
    parts = [p.strip() for p in (syns or "").split(";") if p.strip()]
    out = "; ".join(parts[:max_items])
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip("; ") + "…"
    return out


def _layout_positions(
    nodes: Dict[str, GraphNode],
    edges: List[GraphEdge],
) -> Dict[str, Tuple[float, float]]:
    """
    Deterministic circular layout weighted by degree.
    High-degree hubs slightly inward; avoids all nodes at (0,0).
    """
    degree: Dict[str, int] = {nid: 0 for nid in nodes}
    for e in edges:
        if e.source in degree:
            degree[e.source] += 1
        if e.target in degree:
            degree[e.target] += 1

    ordered = sorted(nodes.keys(), key=lambda nid: (-degree.get(nid, 0), nid))
    n = max(len(ordered), 1)
    # Layout radius scales with node count
    radius = max(280.0, 55.0 * math.sqrt(n))
    cx, cy = radius + 80.0, radius + 80.0
    max_deg = max(degree.values()) if degree else 1

    pos: Dict[str, Tuple[float, float]] = {}
    for i, nid in enumerate(ordered):
        angle = 2.0 * math.pi * i / n
        # Higher-degree nodes closer to center
        d = degree.get(nid, 0)
        r = radius * (0.55 + 0.45 * (1.0 - (d / max_deg if max_deg else 0)))
        pos[nid] = (cx + r * math.cos(angle), cy + r * math.sin(angle))
    return pos


def _cy_line_type(line_style: str) -> str:
    return {
        "solid": "SOLID",
        "dashed": "EQUAL_DASH",
        "dotted": "DOTS",
    }.get(line_style, "SOLID")


def _cy_arrow(arrow: str, undirected: bool) -> str:
    if undirected:
        return "NONE"
    return {
        "triangle": "DELTA",
        "diamond": "DIAMOND",
        "circle": "CIRCLE",
        "tee": "T",
    }.get(arrow, "DELTA")


def write_xgmml(
    path: Path,
    nodes: Dict[str, GraphNode],
    edges: List[GraphEdge],
    *,
    title: str = "MDD associations",
) -> None:
    """
    Cytoscape-friendly XGMML:
      - numeric node/edge ids (source/target match)
      - graphics uses w/h/x/y and ROUNDED_RECTANGLE
      - visual props also stored as att for Style mapping
    """
    cs_vals = [e.confidence for e in edges] or [0.0]
    cs_min, cs_max = min(cs_vals), max(cs_vals)
    positions = _layout_positions(nodes, edges)

    # Map logical ids -> numeric string ids required by many Cytoscape importers
    logical_ids = list(nodes.keys())
    id_map = {lid: str(i + 1) for i, lid in enumerate(logical_ids)}

    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<graph label="{_esc(title)}" directed="1"',
        '  xmlns:dc="http://purl.org/dc/elements/1.1/"',
        '  xmlns:xlink="http://www.w3.org/1999/xlink"',
        '  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
        '  xmlns:cy="http://www.cytoscape.org"',
        '  xmlns="http://www.cs.rpi.edu/XGMML">',
    ]

    for lid in logical_ids:
        n = nodes[lid]
        nid = id_map[lid]
        w, h = n.width_height()
        x, y = positions.get(lid, (0.0, 0.0))
        syns = _truncate_synonyms(n.text_synonyms)
        lines.append(f'  <node id="{nid}" label="{_esc(n.label)}" name="base">')
        # shared name column used by Cytoscape labels
        lines.append(f'    <att name="name" value="{_esc(n.label)}" type="string"/>')
        lines.append(f'    <att name="shared name" value="{_esc(n.label)}" type="string"/>')
        lines.append(f'    <att name="canonicalName" value="{_esc(n.label)}" type="string"/>')
        lines.append(f'    <att name="entity_key" value="{_esc(lid)}" type="string"/>')
        lines.append(f'    <att name="entity_type" value="{_esc(n.etype)}" type="string"/>')
        lines.append(f'    <att name="db_id" value="{_esc(n.db_id)}" type="string"/>')
        lines.append(
            f'    <att name="normalized" value="{"true" if n.normalized else "false"}" type="boolean"/>'
        )
        lines.append(f'    <att name="pref_name" value="{_esc(n.pref_name)}" type="string"/>')
        lines.append(f'    <att name="text_synonyms" value="{_esc(syns)}" type="string"/>')
        # Style-mappable visual hints
        lines.append(f'    <att name="node_fill" value="{n.fill}" type="string"/>')
        lines.append(f'    <att name="node_width" value="{w:.1f}" type="real"/>')
        lines.append(f'    <att name="node_height" value="{h:.1f}" type="real"/>')
        src = str((n.extra or {}).get("_DbSource") or "")
        if src:
            lines.append(f'    <att name="db_source" value="{_esc(src)}" type="string"/>')
        for k, v in (n.extra or {}).items():
            if k.startswith("_"):
                continue
            if v is None or v == "":
                continue
            vs = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            # Truncate long attribute values
            if len(str(vs)) > 800:
                vs = str(vs)[:797] + "…"
            lines.append(f'    <att name="{_esc(str(k))}" value="{_esc(str(vs))}" type="string"/>')
        lines.append(
            f'    <graphics type="ROUNDED_RECTANGLE" fill="{n.fill}" outline="#666666"'
            f' w="{w:.1f}" h="{h:.1f}" x="{x:.2f}" y="{y:.2f}" width="1.5"/>'
        )
        lines.append("  </node>")

    for i, e in enumerate(edges, start=1):
        color, line_style, arrow = _edge_style(e.predicate)
        ew = e.width(cs_min, cs_max)
        undirected = e.direction == "<->"
        src = id_map.get(e.source)
        tgt = id_map.get(e.target)
        if src is None or tgt is None:
            continue
        elabel = e.label
        cy_arrow = _cy_arrow(arrow, undirected)
        cy_line = _cy_line_type(line_style)
        lines.append(
            f'  <edge id="{i}" label="{_esc(elabel)}" source="{src}" target="{tgt}"'
            f' cy:directed="{"0" if undirected else "1"}">'
        )
        lines.append(f'    <att name="name" value="{_esc(elabel)}" type="string"/>')
        lines.append(f'    <att name="interaction" value="{_esc(e.predicate)}" type="string"/>')
        lines.append(f'    <att name="predicate" value="{_esc(e.predicate)}" type="string"/>')
        lines.append(f'    <att name="direction" value="{_esc(e.direction)}" type="string"/>')
        lines.append(f'    <att name="confidence" value="{e.confidence:.6f}" type="real"/>')
        lines.append(f'    <att name="n_sentences" value="{e.n_sentences}" type="integer"/>')
        lines.append(f'    <att name="n_pmids" value="{e.n_pmids}" type="integer"/>')
        pmids_short = _truncate_synonyms(
            e.support_pmids.replace("; ", ";"), max_items=20, max_chars=300
        )
        lines.append(f'    <att name="support_pmids" value="{_esc(pmids_short)}" type="string"/>')
        if e.alt_predicates:
            lines.append(
                f'    <att name="alt_predicates" value="{_esc(e.alt_predicates)}" type="string"/>'
            )
        lines.append(f'    <att name="edge_color" value="{color}" type="string"/>')
        lines.append(f'    <att name="edge_width" value="{ew:.2f}" type="real"/>')
        lines.append(f'    <att name="edge_line_style" value="{line_style}" type="string"/>')
        lines.append(f'    <att name="target_arrow_shape" value="{cy_arrow}" type="string"/>')
        # Nested graphics attrs for Cytoscape arrow/line style
        lines.append(f'    <graphics width="{ew:.2f}" fill="{color}">')
        lines.append('      <att name="cytoscapeEdgeGraphicsAttributes">')
        lines.append('        <att name="sourceArrow" value="NONE"/>')
        lines.append(f'        <att name="targetArrow" value="{cy_arrow}"/>')
        lines.append(f'        <att name="sourceArrowColor" value="{color}"/>')
        lines.append(f'        <att name="targetArrowColor" value="{color}"/>')
        lines.append(f'        <att name="edgeLineType" value="{cy_line}"/>')
        lines.append("      </att>")
        lines.append("    </graphics>")
        lines.append("  </edge>")

    lines.append("</graph>")
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML (cytoscape.js)
# ---------------------------------------------------------------------------


def write_html(
    path: Path,
    nodes: Dict[str, GraphNode],
    edges: List[GraphEdge],
    *,
    title: str = "MDD associations",
) -> None:
    cs_vals = [e.confidence for e in edges] or [0.0]
    cs_min, cs_max = min(cs_vals), max(cs_vals)

    cy_nodes = []
    for n in nodes.values():
        w, h = n.width_height()
        tip_parts = [
            f"Type: {n.etype}",
            f"Normalized: {'yes' if n.normalized else 'no'}",
        ]
        if n.db_id:
            tip_parts.append(f"ID: {n.db_id}")
        src = (n.extra or {}).get("_DbSource")
        if src:
            tip_parts.append(f"Source: {src}")
        if n.pref_name:
            tip_parts.append(f"Pref: {n.pref_name}")
        # First matching Extra field for tooltip
        for k in ("formula", "description", "rank", "infectious_group", "function_note"):
            v = (n.extra or {}).get(k)
            if v:
                tip_parts.append(f"{k}: {str(v)[:120]}")
                break
        cy_nodes.append(
            {
                "data": {
                    "id": n.id,
                    "label": n.label,
                    "etype": n.etype,
                    "tooltip": " | ".join(tip_parts),
                    "w": w,
                    "h": h,
                    "fill": n.fill,
                }
            }
        )

    cy_edges = []
    for e in edges:
        color, line_style, arrow = _edge_style(e.predicate)
        cy_line = {"solid": "solid", "dashed": "dashed", "dotted": "dotted"}.get(line_style, "solid")
        undirected = e.direction == "<->"
        cy_edges.append(
            {
                "data": {
                    "id": e.id,
                    "source": e.source,
                    "target": e.target,
                    "label": e.label,
                    "predicate": e.predicate,
                    "confidence": e.confidence,
                    "width": e.width(cs_min, cs_max),
                    "color": color,
                    "lineStyle": cy_line,
                    "arrow": "none" if undirected else "triangle",
                    "tooltip": (
                        f"{e.predicate} ({e.direction}) | CS={e.confidence:.3f} | "
                        f"sent={e.n_sentences} | pmids={e.n_pmids}"
                        + (f" | alts: {e.alt_predicates}" if e.alt_predicates else "")
                    ),
                }
            }
        )

    elements = json.dumps({"nodes": cy_nodes, "edges": cy_edges}, ensure_ascii=False)
    # Legend for entity types
    legend_items = []
    for etype, (bright, dull) in NODE_COLORS.items():
        legend_items.append(
            f'<span class="leg"><i style="background:{bright}"></i>{html.escape(etype)} '
            f'<i class="dull" style="background:{dull}"></i>unnorm</span>'
        )
    legend_html = " ".join(legend_items)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<style>
  html, body {{ margin:0; height:100%; font-family: "Segoe UI", system-ui, sans-serif; background:#f7f6f3; color:#333; }}
  #bar {{ padding:10px 14px; background:#fff; border-bottom:1px solid #ddd; display:flex; flex-wrap:wrap; gap:12px; align-items:center; }}
  #bar h1 {{ font-size:16px; margin:0; font-weight:600; }}
  #cy {{ width:100%; height:calc(100% - 52px); }}
  #tip {{
    display:none; position:fixed; z-index:20; max-width:360px; padding:8px 10px;
    background:#2b2b2b; color:#f5f5f5; font-size:12px; border-radius:6px; pointer-events:none;
    box-shadow:0 4px 14px rgba(0,0,0,.25);
  }}
  .leg {{ display:inline-flex; align-items:center; gap:4px; margin-right:10px; font-size:12px; color:#555; }}
  .leg i {{ width:12px; height:12px; border-radius:3px; display:inline-block; border:1px solid #999; }}
  .leg i.dull {{ opacity:0.85; }}
  #meta {{ font-size:12px; color:#777; }}
</style>
</head>
<body>
<div id="bar">
  <h1>{html.escape(title)}</h1>
  <div>{legend_html}</div>
  <div id="meta">{len(nodes)} nodes, {len(edges)} edges</div>
</div>
<div id="cy"></div>
<div id="tip"></div>
<script>
const elements = {elements};
const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: elements,
  layout: {{
    name: 'cose',
    animate: false,
    padding: 30,
    nodeRepulsion: 6000,
    idealEdgeLength: 90,
    edgeElasticity: 80
  }},
  style: [
    {{
      selector: 'node',
      style: {{
        'shape': 'round-rectangle',
        'background-color': 'data(fill)',
        'label': 'data(label)',
        'color': '#222',
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': 11,
        'font-weight': 500,
        'text-wrap': 'wrap',
        'text-max-width': 140,
        'width': 'data(w)',
        'height': 'data(h)',
        'border-width': 1,
        'border-color': '#666',
        'padding': 4
      }}
    }},
    {{
      selector: 'edge',
      style: {{
        'width': 'data(width)',
        'line-color': 'data(color)',
        'target-arrow-color': 'data(color)',
        'target-arrow-shape': 'data(arrow)',
        'curve-style': 'bezier',
        'line-style': 'data(lineStyle)',
        'label': 'data(label)',
        'font-size': 9,
        'color': '#444',
        'text-rotation': 'autorotate',
        'text-margin-y': -8,
        'arrow-scale': 1.4
      }}
    }},
    {{
      selector: 'node:selected',
      style: {{ 'border-width': 3, 'border-color': '#222' }}
    }}
  ]
}});
const tip = document.getElementById('tip');
function showTip(ev, text) {{
  tip.style.display = 'block';
  tip.textContent = text;
  tip.style.left = (ev.originalEvent.clientX + 12) + 'px';
  tip.style.top = (ev.originalEvent.clientY + 12) + 'px';
}}
cy.on('mouseover', 'node', function(ev) {{ showTip(ev, ev.target.data('tooltip')); }});
cy.on('mouseover', 'edge', function(ev) {{ showTip(ev, ev.target.data('tooltip')); }});
cy.on('mouseout', 'node, edge', function() {{ tip.style.display = 'none'; }});
</script>
</body>
</html>
"""
    ensure_parent(path)
    path.write_text(doc, encoding="utf-8")


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def run_graphs_stage(
    cfg: Dict[str, Any],
    *,
    unique_tsv: Optional[Path] = None,
    predicates: Optional[Sequence[str]] = None,
    max_nodes: Optional[int] = None,
    min_confidence: Optional[float] = None,
    normalized_only: bool = False,
    keep_multi_edges: bool = False,
    out_dir: Optional[Path] = None,
    stem: str = "associations_graph",
) -> Dict[str, Any]:
    uniq_path = (
        Path(unique_tsv)
        if unique_tsv
        else resolve_path(cfg, "unique_associations_tsv")
    )
    if not uniq_path.is_file():
        raise FileNotFoundError(f"unique associations not found: {uniq_path}")

    graphs_dir = Path(out_dir) if out_dir else resolve_path(cfg, "graphs_dir")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    pred_set = {p.strip() for p in (predicates or []) if p and str(p).strip()} or None
    catalog_path = resolve_path(cfg, "normalize_dir") / "entity_catalog.tsv"
    catalog_extra = _load_catalog_extra(catalog_path)

    nodes, edges = _load_and_filter_associations(
        uniq_path,
        predicates=pred_set,
        normalized_only=normalized_only,
        max_nodes=max_nodes,
        min_confidence=min_confidence,
        keep_multi_edges=keep_multi_edges,
    )
    _attach_catalog(nodes, catalog_extra)

    if not nodes:
        raise RuntimeError(
            "No nodes left after filters; relax --predicates / --max-nodes / "
            "--min-confidence / --normalized-only"
        )

    title_bits = ["MDD associations"]
    if pred_set:
        title_bits.append("predicates=" + ",".join(sorted(pred_set)[:5]))
    if max_nodes:
        title_bits.append(f"max_nodes={max_nodes}")
    if min_confidence is not None:
        title_bits.append(f"min_cs={min_confidence:g}")
    if normalized_only:
        title_bits.append("normalized_only")
    if keep_multi_edges:
        title_bits.append("multi_edges")
    else:
        title_bits.append("collapsed_edges")
    title = " | ".join(title_bits)

    xgmml_path = graphs_dir / f"{stem}.xgmml"
    html_path = graphs_dir / f"{stem}.html"
    write_xgmml(xgmml_path, nodes, edges, title=title)
    write_html(html_path, nodes, edges, title=title)

    # Predicate style legend sidecar
    pred_styles = {}
    for p in sorted({e.predicate for e in edges}):
        c, ls, ar = _edge_style(p)
        pred_styles[p] = {"color": c, "line_style": ls, "arrow": ar}
    meta = {
        "stage": "graphs",
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_normalized_nodes": sum(1 for n in nodes.values() if n.normalized),
        "filters": {
            "predicates": sorted(pred_set) if pred_set else None,
            "max_nodes": max_nodes,
            "min_confidence": min_confidence,
            "normalized_only": normalized_only,
            "keep_multi_edges": keep_multi_edges,
        },
        "outputs": {
            "xgmml": str(xgmml_path),
            "html": str(html_path),
        },
        "predicate_edge_styles": pred_styles,
        "input": str(uniq_path),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (graphs_dir / f"{stem}_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta
