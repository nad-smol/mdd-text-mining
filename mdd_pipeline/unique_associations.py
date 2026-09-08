#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate unique associations and compute confidence scores.

Key: (entity_key1, entity_key2, Predicate, Direction); <-> pairs merged by
lexicographic entity-key order. Self-loops dropped.
CS = geometric mean of sentence/doc frequency factors, mean MatchConfidence
(default 0.5), and mean extraction Probability.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config_utils import ensure_parent, resolve_path

csv.field_size_limit(10_000_000)

DEFAULT_NORM_CONF = 0.5

OUTPUT_FIELDS = [
    "Object1ID",
    "Object1Type",
    "Object1PrefName",
    "Object1TextSynonyms",
    "Object2ID",
    "Object2Type",
    "Object2PrefName",
    "Object2TextSynonyms",
    "Predicate",
    "Direction",
    "ConfidenceScore",
    "n_sentences",
    "n_pmids",
    "SupportPMIDs",
]


def _norm_surface(s: str) -> str:
    return (s or "").strip().lower()


def _entity_key(etype: str, eid: str, surface: str) -> str:
    eid = (eid or "").strip()
    etype = (etype or "").strip()
    if eid:
        return f"{etype}|{eid}"
    return f"{etype}|{_norm_surface(surface)}"


@dataclass
class SideAgg:
    eid: str = ""
    etype: str = ""
    surfaces: Counter = field(default_factory=Counter)
    pref_votes: Counter = field(default_factory=Counter)
    conf_sum: float = 0.0
    conf_n: int = 0

    def add(
        self,
        *,
        eid: str,
        etype: str,
        surface: str,
        pref: str,
        conf: float,
    ) -> None:
        if eid and not self.eid:
            self.eid = eid
        if etype and not self.etype:
            self.etype = etype
        surf = (surface or "").strip()
        if surf:
            self.surfaces[surf] += 1
        if pref:
            self.pref_votes[pref] += 1
        self.conf_sum += conf
        self.conf_n += 1

    @property
    def mean_conf(self) -> float:
        if self.conf_n <= 0:
            return DEFAULT_NORM_CONF
        return self.conf_sum / self.conf_n

    def pref_name(self) -> str:
        if self.pref_votes:
            return self.pref_votes.most_common(1)[0][0]
        if self.surfaces:
            return self.surfaces.most_common(1)[0][0]
        return ""

    def text_synonyms(self) -> str:
        # stable: by frequency desc, then alpha
        items = sorted(self.surfaces.items(), key=lambda x: (-x[1], x[0].lower()))
        return "; ".join(s for s, _ in items)


@dataclass
class GroupAgg:
    side1: SideAgg = field(default_factory=SideAgg)
    side2: SideAgg = field(default_factory=SideAgg)
    predicate: str = ""
    direction: str = ""
    pmids: set = field(default_factory=set)
    n_sentences: int = 0
    prob_sum: float = 0.0

    def add_prob(self, p: float) -> None:
        self.prob_sum += p
        self.n_sentences += 1

    @property
    def mean_prob(self) -> float:
        if self.n_sentences <= 0:
            return 0.0
        return self.prob_sum / self.n_sentences

    @property
    def n_pmids(self) -> int:
        return len(self.pmids)


def _load_mention_lookups(
    mentions_tsv: Path,
) -> Tuple[Dict[Tuple[str, str], Dict[str, str]], Dict[Tuple[str, str], Dict[str, str]]]:
    """
    Returns:
      by_surface: (surface_lower, etype) -> {DbId, PreferredName, MatchConfidence, DbSource}
      by_id: (etype, DbId) -> same (first / best confidence wins)
    """
    by_surface: Dict[Tuple[str, str], Dict[str, str]] = {}
    by_id: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not mentions_tsv.is_file():
        return by_surface, by_id
    with mentions_tsv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            mention = _norm_surface(row.get("Mention") or "")
            etype = (row.get("EntityType") or "").strip()
            dbid = (row.get("DbId") or "").strip()
            pref = (row.get("PreferredName") or "").strip()
            status = (row.get("Status") or "").strip()
            try:
                conf = float(row.get("MatchConfidence") or DEFAULT_NORM_CONF)
            except ValueError:
                conf = DEFAULT_NORM_CONF
            if status != "ok":
                # Non-ok status: use default confidence when DbId is absent
                conf = DEFAULT_NORM_CONF if not dbid else conf
            rec = {
                "DbId": dbid,
                "PreferredName": pref,
                "MatchConfidence": f"{conf:.6f}",
                "DbSource": (row.get("DbSource") or "").strip(),
                "Status": status,
            }
            if mention and etype:
                by_surface[(mention, etype)] = rec
            if dbid and etype and status == "ok":
                prev = by_id.get((etype, dbid))
                if prev is None or float(prev["MatchConfidence"]) < conf:
                    by_id[(etype, dbid)] = rec
    return by_surface, by_id


def _load_catalog_pref(catalog_tsv: Path) -> Dict[Tuple[str, str], str]:
    """(EntityType, DbId) -> PreferredName from catalog."""
    out: Dict[Tuple[str, str], str] = {}
    if not catalog_tsv.is_file():
        return out
    with catalog_tsv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            etype = (row.get("EntityType") or "").strip()
            dbid = (row.get("DbId") or "").strip()
            pref = (row.get("PreferredName") or "").strip()
            if etype and dbid and pref:
                out[(etype, dbid)] = pref
    return out


def _lookup_side(
    *,
    surface: str,
    etype: str,
    eid: str,
    by_surface: Dict[Tuple[str, str], Dict[str, str]],
    by_id: Dict[Tuple[str, str], Dict[str, str]],
    catalog_pref: Dict[Tuple[str, str], str],
) -> Tuple[str, float]:
    """Return (preferred_name, match_confidence)."""
    etype = (etype or "").strip()
    eid = (eid or "").strip()
    surf = _norm_surface(surface)

    if eid:
        rec = by_id.get((etype, eid))
        if rec:
            pref = rec.get("PreferredName") or catalog_pref.get((etype, eid), "")
            return pref, float(rec["MatchConfidence"])
        pref = catalog_pref.get((etype, eid), "")
        # ID without mention row: default or mid confidence if pref known
        return pref, DEFAULT_NORM_CONF if not pref else 0.85

    rec = by_surface.get((surf, etype))
    if rec and rec.get("Status") == "ok" and rec.get("DbId"):
        return rec.get("PreferredName") or "", float(rec["MatchConfidence"])
    # No ID: default normalization confidence
    return "", DEFAULT_NORM_CONF


def _parse_prob(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _canonicalize_pair(
    key1: str,
    key2: str,
    direction: str,
    side1: Dict[str, str],
    side2: Dict[str, str],
) -> Tuple[str, str, Dict[str, str], Dict[str, str]]:
    """For <-> swap so key1 <= key2 lexicographically."""
    if direction == "<->" and key1 > key2:
        return key2, key1, side2, side1
    return key1, key2, side1, side2


def run_unique_associations_stage(
    cfg: Dict[str, Any],
    *,
    associations_tsv: Optional[Path] = None,
    mentions_tsv: Optional[Path] = None,
    catalog_tsv: Optional[Path] = None,
    out_tsv: Optional[Path] = None,
) -> Dict[str, Any]:
    norm_dir = resolve_path(cfg, "normalize_dir")
    assoc_path = Path(associations_tsv) if associations_tsv else (norm_dir / "associations_normalized.tsv")
    mentions_path = Path(mentions_tsv) if mentions_tsv else (norm_dir / "normalized_mentions.tsv")
    catalog_path = Path(catalog_tsv) if catalog_tsv else (norm_dir / "entity_catalog.tsv")
    out_path = (
        Path(out_tsv)
        if out_tsv
        else resolve_path(cfg, "unique_associations_tsv")
    )
    meta_path = out_path.with_name("unique_associations_metadata.json")

    if not assoc_path.is_file():
        raise FileNotFoundError(f"associations_normalized not found: {assoc_path}")

    by_surface, by_id = _load_mention_lookups(mentions_path)
    catalog_pref = _load_catalog_pref(catalog_path)

    groups: Dict[Tuple[str, str, str, str], GroupAgg] = {}

    n_in = 0
    n_self_loops = 0
    with assoc_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_in += 1
            o1 = (row.get("Object1") or "").strip()
            o2 = (row.get("Object2") or "").strip()
            t1 = (row.get("Object1Type") or "").strip()
            t2 = (row.get("Object2Type") or "").strip()
            id1 = (row.get("Object1ID") or row.get("Object1Id") or "").strip()
            id2 = (row.get("Object2ID") or row.get("Object2Id") or "").strip()
            predicate = (row.get("Predicate") or "").strip()
            direction = (row.get("Direction") or "").strip()
            pmid = (row.get("PMID") or "").strip()
            prob = _parse_prob(row.get("Probability") or "")

            key1 = _entity_key(t1, id1, o1)
            key2 = _entity_key(t2, id2, o2)
            side1 = {"surface": o1, "etype": t1, "eid": id1}
            side2 = {"surface": o2, "etype": t2, "eid": id2}
            key1, key2, side1, side2 = _canonicalize_pair(
                key1, key2, direction, side1, side2
            )
            # Drop self-loops after keying/canonicalize
            if key1 == key2:
                n_self_loops += 1
                continue

            gkey = (key1, key2, predicate, direction)
            g = groups.get(gkey)
            if g is None:
                g = GroupAgg(predicate=predicate, direction=direction)
                groups[gkey] = g

            pref1, conf1 = _lookup_side(
                surface=side1["surface"],
                etype=side1["etype"],
                eid=side1["eid"],
                by_surface=by_surface,
                by_id=by_id,
                catalog_pref=catalog_pref,
            )
            pref2, conf2 = _lookup_side(
                surface=side2["surface"],
                etype=side2["etype"],
                eid=side2["eid"],
                by_surface=by_surface,
                by_id=by_id,
                catalog_pref=catalog_pref,
            )
            g.side1.add(
                eid=side1["eid"],
                etype=side1["etype"],
                surface=side1["surface"],
                pref=pref1,
                conf=conf1,
            )
            g.side2.add(
                eid=side2["eid"],
                etype=side2["etype"],
                surface=side2["surface"],
                pref=pref2,
                conf=conf2,
            )
            if pmid:
                g.pmids.add(pmid)
            g.add_prob(prob)

            if n_in % 100_000 == 0:
                print(f"  aggregated {n_in} association rows -> {len(groups)} unique")

    if not groups:
        raise RuntimeError("No associations to aggregate")

    max_sent = max(g.n_sentences for g in groups.values())
    max_pmid = max(g.n_pmids for g in groups.values())
    log_sent_den = math.log(1 + max_sent) if max_sent > 0 else 1.0
    log_pmid_den = math.log(1 + max_pmid) if max_pmid > 0 else 1.0

    rows: List[Dict[str, str]] = []
    for g in groups.values():
        f_sent = math.log(1 + g.n_sentences) / log_sent_den
        f_doc = math.log(1 + g.n_pmids) / log_pmid_den
        n_norm = (g.side1.mean_conf + g.side2.mean_conf) / 2.0
        p_extr = g.mean_prob
        # Clamp zeros before geometric mean
        parts = [max(f_sent, 1e-12), max(f_doc, 1e-12), max(n_norm, 1e-12), max(p_extr, 1e-12)]
        cs = math.prod(parts) ** 0.25

        pmids_sorted = sorted(g.pmids, key=lambda x: (len(x), x))
        rows.append(
            {
                "Object1ID": g.side1.eid,
                "Object1Type": g.side1.etype,
                "Object1PrefName": g.side1.pref_name(),
                "Object1TextSynonyms": g.side1.text_synonyms(),
                "Object2ID": g.side2.eid,
                "Object2Type": g.side2.etype,
                "Object2PrefName": g.side2.pref_name(),
                "Object2TextSynonyms": g.side2.text_synonyms(),
                "Predicate": g.predicate,
                "Direction": g.direction,
                "ConfidenceScore": f"{cs:.6f}",
                "n_sentences": str(g.n_sentences),
                "n_pmids": str(g.n_pmids),
                "SupportPMIDs": "; ".join(pmids_sorted),
            }
        )

    rows.sort(key=lambda r: (-float(r["ConfidenceScore"]), -int(r["n_pmids"]), -int(r["n_sentences"])))

    ensure_parent(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    try:
        tmp.replace(out_path)
    except PermissionError:
        fallback = out_path.with_suffix(out_path.suffix + ".rewrite")
        tmp.replace(fallback)
        print(
            f"[unique-associations] WARNING: {out_path.name} locked; wrote {fallback.name}"
        )
        out_path = fallback

    n_both_id = sum(1 for r in rows if r["Object1ID"] and r["Object2ID"])
    n_one_id = sum(
        1
        for r in rows
        if bool(r["Object1ID"]) != bool(r["Object2ID"])
    )
    n_no_id = sum(1 for r in rows if not r["Object1ID"] and not r["Object2ID"])

    summary = {
        "stage": "unique-associations",
        "n_assoc_in": n_in,
        "n_skipped_self_loops": n_self_loops,
        "n_unique": len(rows),
        "n_unique_both_ids": n_both_id,
        "n_unique_one_id": n_one_id,
        "n_unique_no_id": n_no_id,
        "max_n_sentences": max_sent,
        "max_n_pmids": max_pmid,
        "confidence_formula": "geo_mean(F_sent, F_doc, N_norm, P_extr)",
        "norm_confidence_fallback": DEFAULT_NORM_CONF,
        "outputs": {
            "unique_associations": str(out_path),
            "metadata": str(meta_path),
        },
        "inputs": {
            "associations_normalized": str(assoc_path),
            "normalized_mentions": str(mentions_path),
            "entity_catalog": str(catalog_path),
        },
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
