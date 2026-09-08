#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalize association entity mentions to primary database IDs.

One API per type (PubChem, OLS4 DOID/MONDO, NCBI Gene/Taxonomy, local miRNA/DeprTerms);
Species kept only if infectious by NCBI clade lineage. Cache + catalog metadata on disk.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from Bio import Entrez

from .config_utils import ensure_parent, get_ncbi_credentials, resolve_path

csv.field_size_limit(10_000_000)

USER_AGENT = "MDD-Pipeline/0.1 (research; contact via config email)"
INFECTIOUS_LINEAGE_TERMS = [
    # Used only when LineageEx TaxIds are missing
    "virus",
    "viruses",
    "bacteria",
    "bacterium",
    "fungi",
    "fungus",
    "archaea",
    "plasmodium",
    "toxoplasma",
    "trypanosoma",
    "leishmania",
    "apicomplexa",
    "entamoeba",
    "giardia",
    "euglenozoa",
    "kinetoplastid",
]

# NCBI Taxonomy clade roots (ancestor TaxIds). Protozoa has no single NCBI node;
# major eukaryotic pathogen clades are listed under that group label.
INFECTIOUS_CLADE_ROOTS: List[Tuple[str, str]] = [
    # (tax_id, group); first match in lineage wins
    ("10239", "virus"),  # Viruses
    ("12884", "virus"),  # Viroids
    ("2", "bacteria"),
    ("2157", "archaea"),
    ("4751", "fungi"),
    ("6029", "fungi"),  # Microsporidia
    ("5794", "protozoa"),  # Apicomplexa
    ("33682", "protozoa"),  # Euglenozoa
    ("554915", "protozoa"),  # Amoebozoa
    ("207245", "protozoa"),  # Metamonada
    ("5719", "protozoa"),  # Parabasalia
    ("5752", "protozoa"),  # Heterolobosea
]
INFECTIOUS_CLADE_TAXID_SET = {tid for tid, _ in INFECTIOUS_CLADE_ROOTS}


# Expand short acronyms before NCBI Taxonomy search (first-hit is often wrong)
SPECIES_QUERY_ALIASES = {
    "hiv": "Human immunodeficiency virus",
    "hiv-1": "Human immunodeficiency virus 1",
    "hiv-2": "Human immunodeficiency virus 2",
    "hcv": "Hepatitis C virus",
    "hbv": "Hepatitis B virus",
    "hpv": "Human papillomavirus",
    "hsv": "Herpes simplex virus",
    "hsv-1": "Human alphaherpesvirus 1",
    "hsv-2": "Human alphaherpesvirus 2",
    "ebv": "Human gammaherpesvirus 4",
    "cmv": "Human betaherpesvirus 5",
    "covid-19": "Severe acute respiratory syndrome coronavirus 2",
    "covid19": "Severe acute respiratory syndrome coronavirus 2",
    "sars-cov-2": "Severe acute respiratory syndrome coronavirus 2",
    "sars-cov2": "Severe acute respiratory syndrome coronavirus 2",
    "mrsa": "Methicillin-resistant Staphylococcus aureus",
}

CACHE_VERSION = "v5"

# Map short disease mentions to OLS-friendly query labels
DISEASE_QUERY_ALIASES = {
    "depression": "major depressive disorder",
    "depressive": "major depressive disorder",
    "depressed": "major depressive disorder",
    "major depression": "major depressive disorder",
    "depressive disorder": "major depressive disorder",
    "depressive symptoms": "major depressive disorder",
    "depressive symptom": "major depressive disorder",
    "mdd": "major depressive disorder",
    "md": "major depressive disorder",
    "mde": "major depressive episode",
    "gad": "generalized anxiety disorder",
    "ptsd": "post-traumatic stress disorder",
    "bdd": "body dysmorphic disorder",
    "adhd": "attention deficit hyperactivity disorder",
    "bpd": "borderline personality disorder",
    "bd": "bipolar disorder",
    "ad": "alzheimer disease",
    "suicide": "suicide attempt",
    "suicidal ideation": "suicidal ideation",
    "anhedonia": "anhedonia",
    "trauma": "psychological trauma",
    "pain": "pain",
}

ASSOC_NORMALIZED_FIELDS = [
    "PMID",
    "Section",
    "Sentence",
    "Object1",
    "Object1Type",
    "Object1ID",
    "Object2",
    "Object2Type",
    "Object2ID",
    "Trigger",
    "Predicate",
    "EntityPair",
    "Direction",
    "Stem",
    "Distance",
    "Probability",
    "DecisionNote",
]


MENTION_FIELDS = [
    "Mention",
    "EntityType",
    "DbSource",
    "DbId",
    "PreferredName",
    "MatchConfidence",
    "IsInfectious",
    "InfectiousGroup",
    "Status",
]

CATALOG_FIELDS = [
    "DbSource",
    "DbId",
    "EntityType",
    "PreferredName",
    "Synonyms",
    "Extra",
]


@dataclass
class NormHit:
    db_source: str
    db_id: str
    preferred_name: str
    confidence: float
    synonyms: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    is_infectious: str = ""  # yes/no/blank
    infectious_group: str = ""  # virus|bacteria|fungi|protozoa|archaea|""
    status: str = "ok"  # ok | unmatched | non_infectious | skipped


def _norm_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\u2013\u2014\u2212]+", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _http_json(url: str, timeout: float = 25.0) -> Optional[Any]:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


class DiskCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {}
        if path.is_file():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self.data[key] = value

    def save(self) -> None:
        ensure_parent(self.path)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Resolvers (one DB each)
# ---------------------------------------------------------------------------


def _bare_id(db_id: str) -> str:
    """Strip source prefix before first colon (CID:3821 → 3821)."""
    s = (db_id or "").strip()
    if not s:
        return ""
    if ":" in s:
        return s.split(":", 1)[1].strip()
    return s


def resolve_chemical_pubchem(name: str, sleep_s: float) -> Optional[NormHit]:
    q = urllib.parse.quote(name.strip())
    data = _http_json(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{q}/cids/JSON"
    )
    time.sleep(sleep_s)
    cids: List[Any] = []
    if data:
        cids = (data.get("IdentifierList") or {}).get("CID") or []
    # Fallback: PubChem autocomplete → first suggestion → CID
    if not cids:
        ac = _http_json(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/{q}/json?limit=5"
        )
        time.sleep(sleep_s)
        suggestions = (ac or {}).get("dictionary_terms", {}).get("compound") or []
        for sug in suggestions:
            sq = urllib.parse.quote(str(sug))
            data2 = _http_json(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{sq}/cids/JSON"
            )
            time.sleep(sleep_s)
            cids = ((data2 or {}).get("IdentifierList") or {}).get("CID") or []
            if cids:
                break
    if not cids:
        return None
    cid = str(cids[0])
    props = _http_json(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/"
        f"Title,MolecularFormula,MolecularWeight,IUPACName,InChIKey/JSON"
    )
    time.sleep(sleep_s)
    pref = name
    extra: Dict[str, Any] = {}
    if props:
        rows = (props.get("PropertyTable") or {}).get("Properties") or []
        if rows:
            row = rows[0]
            pref = row.get("Title") or row.get("IUPACName") or name
            extra = {
                "formula": row.get("MolecularFormula") or "",
                "mw": row.get("MolecularWeight") or "",
                "iupac_name": row.get("IUPACName") or "",
                "inchikey": row.get("InChIKey") or "",
                "record_kind": "compound",
            }
    return NormHit(
        db_source="PubChem",
        db_id=cid,
        preferred_name=str(pref),
        confidence=0.85,
        synonyms=[],
        extra=extra,
    )


def _ols_docs_to_hit(
    docs: List[Dict[str, Any]],
    *,
    query: str,
    ontology: str,
    id_prefix: str,
) -> Optional[NormHit]:
    """Exact label/synonym match first; else first OLS hit with confidence 0.7."""
    if not docs:
        return None
    qn = _norm_key(query)
    exact: Optional[NormHit] = None
    first: Optional[NormHit] = None
    for doc in docs:
        label = (doc.get("label") or "").strip()
        short = (doc.get("short_form") or doc.get("obo_id") or "").strip()
        if not short:
            iri = doc.get("iri") or ""
            m = re.search(rf"({id_prefix}[_:]\d+)", iri, re.I)
            short = m.group(1).replace("_", ":") if m else ""
        if not short:
            continue
        oid = short.upper().replace("_", ":")
        if not oid.startswith(f"{id_prefix}:"):
            digits = re.sub(r"\D", "", oid)
            if not digits:
                continue
            oid = f"{id_prefix}:{digits}"
        bare = _bare_id(oid)
        syns = doc.get("synonym") or []
        if isinstance(syns, str):
            syns = [syns]
        names = [label] + list(syns)
        is_exact = any(_norm_key(n) == qn for n in names if n)
        desc = doc.get("description")
        if isinstance(desc, list):
            desc = desc[0] if desc else ""
        hit = NormHit(
            db_source=id_prefix,
            db_id=bare,
            preferred_name=label or query,
            confidence=0.9 if is_exact else 0.7,
            synonyms=[s for s in syns if s][:20],
            extra={
                "description": str(desc or "")[:500],
                "iri": doc.get("iri") or "",
                "ontology": ontology,
                "obo_id": oid,
            },
        )
        if first is None:
            # Gate non-exact first hit: fuzzy >= 0.45 or shared prefix (len >= 4)
            if _fuzzy(qn, _norm_key(label)) >= 0.45 or any(
                _norm_key(n).startswith(qn) or qn.startswith(_norm_key(n))
                for n in names
                if n and len(_norm_key(n)) >= 4
            ):
                first = hit
        if is_exact:
            exact = hit
            break
    return exact or first


def resolve_disease_ols(name: str, sleep_s: float) -> Optional[NormHit]:
    """Primary: DOID via OLS4; fallback MONDO."""
    query = DISEASE_QUERY_ALIASES.get(_norm_key(name), name)
    q = urllib.parse.quote(query.strip())
    for ontology, prefix in (("doid", "DOID"), ("mondo", "MONDO")):
        url = (
            "https://www.ebi.ac.uk/ols4/api/search"
            f"?q={q}&ontology={ontology}&rows=10&queryFields=label,synonym"
        )
        data = _http_json(url)
        time.sleep(sleep_s)
        if not data:
            continue
        docs = (data.get("response") or {}).get("docs") or []
        hit = _ols_docs_to_hit(docs, query=query, ontology=ontology, id_prefix=prefix)
        if hit:
            return hit
    return None


def resolve_gene_ncbi(
    name: str,
    *,
    email: str,
    api_key: Optional[str],
    sleep_s: float,
    tax_id: int = 9606,
) -> Optional[NormHit]:
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    term = f"{name}[Gene Name] AND txid{tax_id}[Organism]"
    try:
        h = Entrez.esearch(db="gene", term=term, retmax=5)
        rec = Entrez.read(h)
        h.close()
        time.sleep(sleep_s)
    except Exception:
        return None
    ids = rec.get("IdList") or []
    if not ids:
        # Broader organism-scoped search
        try:
            h = Entrez.esearch(db="gene", term=f"{name} AND txid{tax_id}[Organism]", retmax=5)
            rec = Entrez.read(h)
            h.close()
            time.sleep(sleep_s)
            ids = rec.get("IdList") or []
        except Exception:
            return None
    if not ids:
        return None
    gene_id = str(ids[0])
    try:
        h = Entrez.esummary(db="gene", id=gene_id)
        summaries = Entrez.read(h)
        h.close()
        time.sleep(sleep_s)
    except Exception:
        return NormHit("NCBI_Gene", gene_id, name, 0.7, [], {"tax_id": tax_id})
    doc = None
    if isinstance(summaries, dict) and "DocumentSummarySet" in summaries:
        docs = summaries["DocumentSummarySet"].get("DocumentSummary") or []
        doc = docs[0] if docs else None
    elif isinstance(summaries, list) and summaries:
        doc = summaries[0]
    symbol = name
    desc = ""
    if isinstance(doc, dict):
        symbol = doc.get("Name") or doc.get("NomenclatureSymbol") or name
        desc = doc.get("Description") or doc.get("Summary") or ""
    return NormHit(
        db_source="NCBI_Gene",
        db_id=gene_id,
        preferred_name=str(symbol),
        confidence=0.85,
        synonyms=[],
        extra={
            "tax_id": tax_id,
            "description": str(desc)[:500],
            "function_note": str(desc)[:500],
        },
    )


def _parse_lineage_ex(rec0: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize NCBI LineageEx / Lineage into [{tax_id, name, rank}, ...]."""
    out: List[Dict[str, str]] = []
    lin = rec0.get("LineageEx") or rec0.get("Lineage") or []
    if isinstance(lin, list):
        for item in lin:
            if isinstance(item, dict):
                out.append(
                    {
                        "tax_id": str(item.get("TaxId") or "").strip(),
                        "name": str(item.get("ScientificName") or "").strip(),
                        "rank": str(item.get("Rank") or "").strip(),
                    }
                )
            else:
                out.append({"tax_id": "", "name": str(item).strip(), "rank": ""})
    elif isinstance(lin, str):
        for part in lin.split(";"):
            name = part.strip()
            if name:
                out.append({"tax_id": "", "name": name, "rank": ""})
    # Include the taxon itself
    self_id = str(rec0.get("TaxId") or "").strip()
    self_name = str(rec0.get("ScientificName") or "").strip()
    self_rank = str(rec0.get("Rank") or "").strip()
    if self_id or self_name:
        out.append({"tax_id": self_id, "name": self_name, "rank": self_rank})
    return out


def _classify_infectious(
    lineage: List[Dict[str, str]],
) -> Tuple[bool, str, Optional[str]]:
    """
    Classify via NCBI hierarchy: if any ancestor TaxId is an infectious clade root.

    Returns (is_infectious, group, matched_clade_taxid).
    """
    ancestor_ids = {n["tax_id"] for n in lineage if n.get("tax_id")}
    for clade_tid, group in INFECTIOUS_CLADE_ROOTS:
        if clade_tid in ancestor_ids:
            return True, group, clade_tid

    # Fallback: keyword scan on names (when TaxIds absent from Lineage string)
    if not any(n.get("tax_id") for n in lineage):
        blob = " ".join(n.get("name", "") for n in lineage).lower()
        if any(term in blob for term in INFECTIOUS_LINEAGE_TERMS):
            if "virus" in blob:
                return True, "virus", None
            if "bacter" in blob:
                return True, "bacteria", None
            if "fung" in blob:
                return True, "fungi", None
            if "archaea" in blob:
                return True, "archaea", None
            return True, "protozoa", None
    return False, "", None


def resolve_species_ncbi(
    name: str,
    *,
    email: str,
    api_key: Optional[str],
    sleep_s: float,
) -> Optional[NormHit]:
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
    query_name = SPECIES_QUERY_ALIASES.get(_norm_key(name), name)
    try:
        h = Entrez.esearch(
            db="taxonomy",
            term=f"{query_name}[Scientific Name] OR {query_name}[All Names]",
            retmax=8,
        )
        rec = Entrez.read(h)
        h.close()
        time.sleep(sleep_s)
    except Exception:
        return None
    ids = rec.get("IdList") or []
    if not ids:
        return None

    qn = _norm_key(query_name)
    best_hit: Optional[NormHit] = None
    best_score = -1.0
    # Score top candidates; NCBI first hit is not always the intended taxon
    id_csv = ",".join(str(x) for x in ids[:8])
    try:
        h = Entrez.efetch(db="taxonomy", id=id_csv, retmode="xml")
        records = Entrez.read(h)
        h.close()
        time.sleep(sleep_s)
    except Exception:
        return None
    if not records:
        return None

    for rec0 in records:
        tax_id = str(rec0.get("TaxId") or "")
        sci = str(rec0.get("ScientificName") or query_name)
        rank = str(rec0.get("Rank") or "")
        other = rec0.get("OtherNames") or {}
        aliases: List[str] = [sci]
        if isinstance(other, dict):
            for key in ("GenbankCommonName", "CommonName", "Synonym", "EquivalentName"):
                val = other.get(key)
                if isinstance(val, list):
                    aliases.extend(str(x) for x in val)
                elif val:
                    aliases.append(str(val))

        lineage = _parse_lineage_ex(rec0 if isinstance(rec0, dict) else {})
        lineage_names = [n["name"] for n in lineage if n.get("name")]

        score = 0.0
        for a in aliases:
            an = _norm_key(a)
            if an == qn:
                score = 1.0
                break
            score = max(score, _fuzzy(qn, an))
        score = max(score, _fuzzy(_norm_key(name), _norm_key(sci)))
        if qn.isalpha() and len(qn) <= 6 and qn in _norm_key(sci).replace("-", " ").split():
            score = max(score, 0.9)
        # Down-weight Simian immunodeficiency virus when query targets HIV
        if "immunodeficiency" in qn and "simian" in _norm_key(sci):
            score *= 0.4

        infectious, group, clade_tid = _classify_infectious(lineage)
        hit = NormHit(
            db_source="NCBI_Taxonomy",
            db_id=tax_id,
            preferred_name=sci,
            confidence=round(min(0.95, 0.55 + 0.4 * score), 3),
            synonyms=[a for a in aliases if a and a != sci][:15],
            extra={
                "rank": rank,
                "lineage": "; ".join(lineage_names[:30]),
                "lineage_taxids": ";".join(
                    n["tax_id"] for n in lineage if n.get("tax_id")
                )[:500],
                "infectious_group": group,
                "infectious_clade_taxid": clade_tid or "",
                "query_used": query_name,
            },
            is_infectious="yes" if infectious else "no",
            infectious_group=group,
            status="ok" if infectious else "non_infectious",
        )
        if score > best_score:
            best_score = score
            best_hit = hit

    if best_hit is None or best_score < 0.45:
        return None
    return best_hit


def resolve_mirna_local(name: str) -> NormHit:
    # NER NormalizedName is often already mir-* / let-*
    canon = _norm_key(name).replace(" ", "")
    return NormHit(
        db_source="local_mirna",
        db_id=canon,
        preferred_name=canon,
        confidence=0.9,
        synonyms=[name] if name != canon else [],
        extra={},
    )


def resolve_deprterms_local(name: str) -> NormHit:
    canon = name.strip()
    slug = re.sub(r"[^a-z0-9]+", "_", _norm_key(canon)).strip("_")
    return NormHit(
        db_source="local_deprterms",
        db_id=slug,
        preferred_name=canon,
        confidence=1.0,
        synonyms=[],
        extra={},
    )


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


def _collect_unique_mentions(associations_tsv: Path) -> Dict[Tuple[str, str], int]:
    """(mention_lower, etype) -> count from associations Object1/Object2."""
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    with associations_tsv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            for side in ("1", "2"):
                mention = (row.get(f"Object{side}") or "").strip()
                etype = (row.get(f"Object{side}Type") or "").strip()
                if mention and etype:
                    counts[(_norm_key(mention), etype)] += 1
    return counts


def _hit_from_cache(raw: Dict[str, Any]) -> NormHit:
    return NormHit(
        db_source=raw.get("db_source") or "",
        db_id=raw.get("db_id") or "",
        preferred_name=raw.get("preferred_name") or "",
        confidence=float(raw.get("confidence") or 0),
        synonyms=list(raw.get("synonyms") or []),
        extra=dict(raw.get("extra") or {}),
        is_infectious=raw.get("is_infectious") or "",
        infectious_group=raw.get("infectious_group")
        or (raw.get("extra") or {}).get("infectious_group")
        or "",
        status=raw.get("status") or "ok",
    )


def run_normalize_stage(
    cfg: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    associations_tsv: Optional[Path] = None,
    max_mentions: Optional[int] = None,
    sleep_s: Optional[float] = None,
    entity_types: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    email, key = get_ncbi_credentials(cfg, api_key_cli=api_key)
    key = key or api_key
    assoc_path = (
        Path(associations_tsv)
        if associations_tsv
        else resolve_path(cfg, "associations_tsv")
    )
    if not assoc_path.is_file():
        raise FileNotFoundError(f"associations not found: {assoc_path}")

    out_dir = resolve_path(cfg, "normalize_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    mentions_out = out_dir / "normalized_mentions.tsv"
    catalog_out = out_dir / "entity_catalog.tsv"
    assoc_out = out_dir / "associations_normalized.tsv"
    cache_path = out_dir / "normalize_cache.json"

    sleep = float(sleep_s if sleep_s is not None else (cfg.get("normalize") or {}).get("sleep_s", 0.12))
    cache = DiskCache(cache_path)

    type_filter = {t.strip() for t in (entity_types or []) if str(t).strip()} or None

    unique = _collect_unique_mentions(assoc_path)
    items = sorted(unique.items(), key=lambda x: (-x[1], x[0][1], x[0][0]))
    if type_filter:
        items = [it for it in items if it[0][1] in type_filter]
    if max_mentions is not None:
        items = items[:max_mentions]

    print(f"Unique mentions to normalize: {len(items)} (from {assoc_path})")
    if type_filter:
        print(f"  type filter: {sorted(type_filter)}")

    # Previous mentions for partial re-runs (--types / --max-mentions)
    prev_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
    if mentions_out.is_file():
        with mentions_out.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                prev_by_key[(row["Mention"], row["EntityType"])] = dict(row)

    mention_rows: List[Dict[str, str]] = []
    catalog: Dict[Tuple[str, str], NormHit] = {}
    resolved_keys: Set[Tuple[str, str]] = set()

    for i, ((mention_key, etype), freq) in enumerate(items, 1):
        resolved_keys.add((mention_key, etype))
        cache_key = f"{CACHE_VERSION}|{etype}|{mention_key}"
        cached = cache.get(cache_key)
        if cached:
            hit = _hit_from_cache(cached)
        else:
            hit = None
            surface = mention_key
            try:
                if etype == "Chemical":
                    hit = resolve_chemical_pubchem(surface, sleep)
                elif etype == "Disease":
                    hit = resolve_disease_ols(surface, sleep)
                elif etype == "Gene":
                    hit = resolve_gene_ncbi(surface, email=email, api_key=key, sleep_s=sleep)
                elif etype == "Species":
                    hit = resolve_species_ncbi(surface, email=email, api_key=key, sleep_s=sleep)
                elif etype == "miRNA":
                    hit = resolve_mirna_local(surface)
                elif etype == "DeprTerms":
                    hit = resolve_deprterms_local(surface)
                else:
                    hit = NormHit("", "", surface, 0.0, status="skipped")
            except Exception as exc:
                hit = NormHit("", "", surface, 0.0, status=f"error:{type(exc).__name__}")

            if hit is None:
                hit = NormHit("", "", surface, 0.0, status="unmatched")

            cache.set(
                cache_key,
                {
                    "db_source": hit.db_source,
                    "db_id": hit.db_id,
                    "preferred_name": hit.preferred_name,
                    "confidence": hit.confidence,
                    "synonyms": hit.synonyms,
                    "extra": hit.extra,
                    "is_infectious": hit.is_infectious,
                    "infectious_group": hit.infectious_group,
                    "status": hit.status,
                    "freq": freq,
                },
            )

        mention_rows.append(
            {
                "Mention": mention_key,
                "EntityType": etype,
                "DbSource": hit.db_source,
                "DbId": hit.db_id,
                "PreferredName": hit.preferred_name,
                "MatchConfidence": f"{hit.confidence:.3f}",
                "IsInfectious": hit.is_infectious,
                "InfectiousGroup": hit.infectious_group,
                "Status": hit.status,
            }
        )
        if hit.db_id and hit.status in {"ok", ""}:
            catalog[(hit.db_source, hit.db_id)] = hit

        if i % 25 == 0:
            print(f"  normalized {i}/{len(items)}")
            cache.save()

    cache.save()

    for key, row in prev_by_key.items():
        if key in resolved_keys:
            continue
        mention_rows.append(row)
        if row.get("DbId") and row.get("Status") in {"ok", ""}:
            catalog.setdefault(
                (row.get("DbSource") or "", row["DbId"]),
                NormHit(
                    db_source=row.get("DbSource") or "",
                    db_id=row["DbId"],
                    preferred_name=row.get("PreferredName") or "",
                    confidence=float(row.get("MatchConfidence") or 0),
                    is_infectious=row.get("IsInfectious") or "",
                    infectious_group=row.get("InfectiousGroup") or "",
                    status=row.get("Status") or "ok",
                ),
            )

    # Merge Extra from previous catalog when present
    if catalog_out.is_file():
        with catalog_out.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = (row.get("DbSource") or "", row.get("DbId") or "")
                if not key[1]:
                    continue
                if key not in catalog:
                    try:
                        extra = json.loads(row.get("Extra") or "{}")
                    except Exception:
                        extra = {}
                    catalog[key] = NormHit(
                        db_source=key[0],
                        db_id=key[1],
                        preferred_name=row.get("PreferredName") or "",
                        confidence=0.0,
                        synonyms=[s for s in (row.get("Synonyms") or "").split("; ") if s],
                        extra=extra,
                        status="ok",
                    )
                elif not catalog[key].extra and row.get("Extra"):
                    try:
                        catalog[key].extra = json.loads(row["Extra"])
                    except Exception:
                        pass
                    if row.get("Synonyms") and not catalog[key].synonyms:
                        catalog[key].synonyms = [
                            s for s in row["Synonyms"].split("; ") if s
                        ]

    with mentions_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MENTION_FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in mention_rows:
            out = {k: row.get(k, "") for k in MENTION_FIELDS}
            w.writerow(out)

    with catalog_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_FIELDS, delimiter="\t")
        w.writeheader()
        for (src, dbid), hit in sorted(catalog.items()):
            w.writerow(
                {
                    "DbSource": src,
                    "DbId": _bare_id(dbid),
                    "EntityType": next(
                        (
                            r["EntityType"]
                            for r in mention_rows
                            if _bare_id(r.get("DbId") or "") == _bare_id(dbid)
                            and (r.get("DbSource") or "") == (src or r.get("DbSource") or "")
                        ),
                        next(
                            (
                                r["EntityType"]
                                for r in mention_rows
                                if _bare_id(r.get("DbId") or "") == _bare_id(dbid)
                            ),
                            "",
                        ),
                    ),
                    "PreferredName": hit.preferred_name,
                    "Synonyms": "; ".join(hit.synonyms[:30]),
                    "Extra": json.dumps(hit.extra, ensure_ascii=False),
                }
            )

    # Map mention+type → bare id; Species status for filtering
    id_map: Dict[Tuple[str, str], str] = {}
    species_status: Dict[str, Dict[str, str]] = {}
    n_mentions_ok = 0
    for r in mention_rows:
        key = (r["Mention"], r["EntityType"])
        bare = _bare_id(r.get("DbId") or "")
        if r["Status"] == "ok" and bare:
            id_map[key] = bare
            n_mentions_ok += 1
        if r["EntityType"] == "Species":
            species_status[r["Mention"]] = {
                "status": r["Status"],
                "infectious": r["IsInfectious"],
                "db_id": bare,
            }

    # Rewrite associations: ObjectN / ObjectNType / ObjectNID grouped; bare IDs only
    n_in = n_out = n_drop_species = n_drop_unmatched_species = n_species_pending = 0
    n_id1 = n_id2 = n_both = 0
    # Write to temp then replace (Windows often locks the open TSV in Excel)
    assoc_tmp = assoc_out.with_suffix(assoc_out.suffix + ".tmp")
    with assoc_path.open(encoding="utf-8", newline="") as f_in, assoc_tmp.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in, delimiter="\t")
        writer = csv.DictWriter(f_out, fieldnames=ASSOC_NORMALIZED_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in reader:
            n_in += 1
            o1 = _norm_key(row.get("Object1") or "")
            o2 = _norm_key(row.get("Object2") or "")
            t1 = (row.get("Object1Type") or "").strip()
            t2 = (row.get("Object2Type") or "").strip()
            id1 = id_map.get((o1, t1), "")
            id2 = id_map.get((o2, t2), "")

            drop = False
            for typ, ment in ((t1, o1), (t2, o2)):
                if typ != "Species":
                    continue
                info = species_status.get(ment)
                if info is None:
                    n_species_pending += 1
                    continue
                if info["status"] == "ok" and info["infectious"] == "yes":
                    continue
                if info["status"] in {"unmatched", "error"} or not info["db_id"]:
                    drop = True
                    n_drop_unmatched_species += 1
                    break
                drop = True
                n_drop_species += 1
                break
            if drop:
                continue

            if id1:
                n_id1 += 1
            if id2:
                n_id2 += 1
            if id1 and id2:
                n_both += 1

            writer.writerow(
                {
                    "PMID": row.get("PMID") or "",
                    "Section": row.get("Section") or "",
                    "Sentence": row.get("Sentence") or "",
                    "Object1": row.get("Object1") or "",
                    "Object1Type": t1,
                    "Object1ID": id1,
                    "Object2": row.get("Object2") or "",
                    "Object2Type": t2,
                    "Object2ID": id2,
                    "Trigger": row.get("Trigger") or "",
                    "Predicate": row.get("Predicate") or "",
                    "EntityPair": row.get("EntityPair") or "",
                    "Direction": row.get("Direction") or "",
                    "Stem": row.get("Stem") or "",
                    "Distance": row.get("Distance") or "",
                    "Probability": row.get("Probability") or "",
                    "DecisionNote": row.get("DecisionNote") or "",
                }
            )
            n_out += 1

    try:
        assoc_tmp.replace(assoc_out)
    except PermissionError:
        fallback = assoc_out.with_suffix(assoc_out.suffix + ".rewrite")
        assoc_tmp.replace(fallback)
        print(
            f"[normalize] WARNING: {assoc_out.name} is locked; wrote {fallback.name}. "
            "Close Excel and rename it over the original."
        )
        assoc_out = fallback

    summary = {
        "stage": "normalize",
        "n_unique_mentions_resolved_this_run": len(items),
        "n_mentions_in_lexicon": len(mention_rows),
        "n_mentions_ok": n_mentions_ok,
        "n_catalog_ids": len(catalog),
        "n_assoc_in": n_in,
        "n_assoc_out": n_out,
        "n_assoc_object1_id": n_id1,
        "n_assoc_object2_id": n_id2,
        "n_assoc_both_ids": n_both,
        "n_assoc_dropped_non_infectious_species": n_drop_species,
        "n_assoc_dropped_unmatched_species": n_drop_unmatched_species,
        "n_assoc_species_pending_normalize": n_species_pending,
        "outputs": {
            "normalized_mentions": str(mentions_out),
            "entity_catalog": str(catalog_out),
            "associations_normalized": str(assoc_out),
            "cache": str(cache_path),
        },
        "sources": {
            "Chemical": "PubChem (CID, bare)",
            "Disease": "OLS4 DOID→MONDO (bare numeric id; DbSource in catalog)",
            "Gene": "NCBI Gene (bare GeneID)",
            "Species": "NCBI Taxonomy taxid bare; infectious clade filter",
            "miRNA": "local canonical",
            "DeprTerms": "local Env_factors canonical",
        },
        "note": (
            "Low ID coverage usually means normalize was only run with --max-mentions. "
            "Run full `python run_pipeline.py normalize` to resolve all unique mentions."
        ),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "normalize_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
