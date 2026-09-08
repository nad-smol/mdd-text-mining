#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PubMed corpus stage: titles and abstracts for MDD.

Supports NCBI API key, half-year date splitting (Entrez result-window limits),
and incremental update by skipping PMIDs already in the corpus TSV.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET

from Bio import Entrez

from .config_utils import (
    ensure_parent,
    get_ncbi_credentials,
    related_artifact_paths,
    resolve_path,
)

CORPUS_FIELDNAMES = [
    "PMID",
    "PMCID",
    "Year",
    "Journal",
    "ArticleTitle",
    "DOI",
    "MeshTerms",
    "PublicationTypes",
    "HasFullText",
    "Abstract",
    "Introduction",
    "Methods",
    "Results",
    "Discussion",
    "Conclusions",
    "OtherSections",
    "FullText",
]


def build_mesh_query(
    mesh_terms: Sequence[str],
    exclude_publication_types: Sequence[str] | None = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> str:
    # Do NOT wrap MeSH terms in quotes: Entrez treats
    # "Depressive Disorder, Major"[Mesh] as a missing quoted phrase (0 hits),
    # while Depressive Disorder, Major[Mesh] resolves correctly (~45k).
    parts = [f"{term}[Mesh]" for term in mesh_terms]
    query = " OR ".join(parts)
    if len(parts) > 1:
        query = f"({query})"

    if exclude_publication_types:
        for pt in exclude_publication_types:
            query = f"{query} NOT {pt}[PT]"

    if date_from and date_to:
        date_filter = f'("{date_from}"[PDAT] : "{date_to}"[PDAT])'
        query = f"({query}) AND {date_filter}"

    return query


def apply_date_filter(base_query: str, date_from: str, date_to: str) -> str:
    """Attach a PDAT window to an arbitrary Entrez query string."""
    q = (base_query or "").strip()
    if not q:
        raise ValueError("empty PubMed query")
    date_filter = f'("{date_from}"[PDAT] : "{date_to}"[PDAT])'
    return f"({q}) AND {date_filter}"

def generate_date_intervals(
    start_year: int,
    end_year: int,
    half_year: bool = True,
) -> List[Tuple[str, str, str]]:
    intervals: List[Tuple[str, str, str]] = []
    for year in range(start_year, end_year + 1):
        if half_year:
            intervals.append((f"{year}/01/01", f"{year}/06/30", f"{year} H1"))
            intervals.append((f"{year}/07/01", f"{year}/12/31", f"{year} H2"))
        else:
            intervals.append((f"{year}/01/01", f"{year}/12/31", f"{year}"))
    return intervals


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    t = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return " ".join(t.split())


def extract_pubmed_metadata(article_xml: ET.Element) -> Dict[str, str]:
    medline = article_xml.find("MedlineCitation")
    pmid = medline.findtext("PMID") if medline is not None else None
    article = medline.find("Article") if medline is not None else None

    title = ""
    journal = ""
    year = ""
    abstract = ""

    if article is not None:
        title = article.findtext("ArticleTitle") or ""

        journal_elem = article.find("Journal")
        if journal_elem is not None:
            journal = journal_elem.findtext("Title") or ""
            pub_date = journal_elem.find("JournalIssue/PubDate")
            if pub_date is not None:
                year = pub_date.findtext("Year") or ""
                if not year:
                    medline_date = pub_date.findtext("MedlineDate") or ""
                    if medline_date:
                        year = medline_date.split(" ")[0]

        abstr_elems = article.findall("Abstract/AbstractText")
        parts = []
        for a in abstr_elems:
            label = a.get("Label")
            part_text = "".join(a.itertext()) if a is not None else ""
            if label:
                parts.append(f"{label}: {part_text}")
            else:
                parts.append(part_text)
        abstract = " ".join(parts)

    mesh_terms: List[str] = []
    mesh_list = medline.find("MeshHeadingList") if medline is not None else None
    if mesh_list is not None:
        for mh in mesh_list.findall("MeshHeading"):
            desc = mh.find("DescriptorName")
            if desc is not None and desc.text:
                mesh_terms.append(desc.text)

    pub_types: List[str] = []
    if medline is not None:
        for pt in medline.findall("Article/PublicationTypeList/PublicationType"):
            if pt.text:
                pub_types.append(pt.text)

    doi = ""
    if article is not None:
        for id_e in article.findall("ELocationID"):
            if id_e.get("EIdType", "").lower() == "doi" and id_e.text:
                doi = id_e.text
                break

    return {
        "PMID": pmid or "",
        "Journal": journal or "",
        "Year": year or "",
        "ArticleTitle": title or "",
        "Abstract": abstract or "",
        "MeshTerms": "; ".join(mesh_terms),
        "PublicationTypes": "; ".join(pub_types),
        "DOI": doi,
    }


def chunked(lst: Sequence[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield list(lst[i : i + n])


def load_existing_pmids(corpus_tsv: Path) -> Set[str]:
    if not corpus_tsv.exists():
        return set()
    pmids: Set[str] = set()
    with corpus_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pmid = (row.get("PMID") or "").strip()
            if pmid:
                pmids.add(pmid)
    return pmids


def count_corpus_rows(corpus_tsv: Path) -> int:
    if not corpus_tsv.exists():
        return 0
    with corpus_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        return sum(1 for _ in reader)


def configure_entrez(email: str, api_key: Optional[str]) -> None:
    Entrez.email = email
    Entrez.api_key = api_key
    Entrez.tool = "mdd_text_mining_pipeline"


def esearch_pmids(query: str, sleep_s: float) -> List[str]:
    handle = Entrez.esearch(db="pubmed", term=query, retmax=100000)
    result = Entrez.read(handle)
    handle.close()
    time.sleep(sleep_s)
    return list(result.get("IdList", []))


def efetch_articles(pmids: Sequence[str], sleep_s: float) -> List[ET.Element]:
    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="xml",
        retmode="xml",
    )
    xml_data = handle.read()
    handle.close()
    time.sleep(sleep_s)
    root = ET.fromstring(xml_data)
    return root.findall(".//PubmedArticle")


def row_from_article(art: ET.Element) -> Optional[Dict[str, str]]:
    meta = extract_pubmed_metadata(art)
    pmid = meta.get("PMID") or ""
    if not pmid:
        return None
    abstract = clean_text(meta.get("Abstract", "")) or "-"
    return {
        "PMID": pmid,
        "PMCID": "",
        "Year": meta.get("Year", ""),
        "Journal": meta.get("Journal", ""),
        "ArticleTitle": clean_text(meta.get("ArticleTitle", "")),
        "DOI": meta.get("DOI", ""),
        "MeshTerms": meta.get("MeshTerms", ""),
        "PublicationTypes": meta.get("PublicationTypes", ""),
        "HasFullText": "0",
        "Abstract": abstract,
        "Introduction": "-",
        "Methods": "-",
        "Results": "-",
        "Discussion": "-",
        "Conclusions": "-",
        "OtherSections": "-",
        "FullText": "-",
    }


def write_metadata(
    metadata_path: Path,
    cfg: dict,
    *,
    query_example: str,
    n_records: int,
    n_added: int,
    mode: str,
    start_year: int,
    end_year: int,
) -> None:
    ensure_parent(metadata_path)
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "project": cfg.get("project"),
        "query_example": query_example,
        "mesh_terms": cfg["pubmed"]["mesh_terms"],
        "exclude_publication_types": cfg["pubmed"].get("exclude_publication_types", []),
        "start_year": start_year,
        "end_year": end_year,
        "fetch_pmc_fulltext": bool(cfg["pubmed"].get("fetch_pmc_fulltext", False)),
        "mode_last_run": mode,
        "n_records": n_records,
        "n_added_last_run": n_added,
        "updated_at_utc": now,
        "artifacts": related_artifact_paths(cfg),
    }
    if metadata_path.exists():
        try:
            old = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload["created_at_utc"] = old.get("created_at_utc", now)
        except json.JSONDecodeError:
            payload["created_at_utc"] = now
    else:
        payload["created_at_utc"] = now

    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_corpus_stage(
    cfg: dict,
    *,
    mode: Optional[str] = None,
    api_key: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    max_records: Optional[int] = None,
    corpus_tsv: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    query: Optional[str] = None,
    quiet: bool = False,
) -> dict:
    """
    Download / update PubMed corpus.

    Returns summary dict with paths and counts.
    """
    pubmed = cfg["pubmed"]
    ncbi = cfg["ncbi"]

    mode = (mode or cfg.get("corpus", {}).get("mode") or "update").lower()
    if mode not in {"update", "full"}:
        raise ValueError("corpus mode must be 'update' or 'full'")

    email, resolved_key = get_ncbi_credentials(cfg, api_key_cli=api_key)
    configure_entrez(email, resolved_key)

    sleep_s = float(ncbi.get("sleep_between_requests", 0.34 if not resolved_key else 0.12))
    batch_size = int(ncbi.get("pmid_batch_size", 200))

    sy = int(start_year if start_year is not None else pubmed["start_year"])
    ey_cfg = end_year if end_year is not None else pubmed.get("end_year")
    ey = int(ey_cfg) if ey_cfg is not None else datetime.now().year

    out_tsv = Path(corpus_tsv) if corpus_tsv else resolve_path(cfg, "corpus_tsv")
    out_meta = Path(metadata_path) if metadata_path else resolve_path(cfg, "corpus_metadata")
    ensure_parent(out_tsv)

    mesh_terms = pubmed["mesh_terms"]
    exclude_pts = pubmed.get("exclude_publication_types") or []
    half = bool(pubmed.get("use_half_years", True))

    custom_query = (query or "").strip() or None
    if custom_query:
        query_example = custom_query
    else:
        query_example = build_mesh_query(mesh_terms, exclude_pts)

    existing: Set[str] = set()
    if mode == "update" and out_tsv.exists():
        existing = load_existing_pmids(out_tsv)
        if not quiet:
            print(f"[corpus] update mode: {len(existing)} PMIDs already in {out_tsv}")
    elif mode == "full":
        if out_tsv.exists() and not quiet:
            print(f"[corpus] full mode: rebuilding {out_tsv}")
        existing = set()

    write_header = mode == "full" or not out_tsv.exists()
    file_mode = "w" if write_header else "a"

    intervals = generate_date_intervals(sy, ey, half)
    added = 0
    processed_batches = 0

    if not quiet:
        key_state = "yes" if resolved_key else "no"
        print(f"[corpus] query: {query_example}")
        print(f"[corpus] years {sy}–{ey}, intervals={len(intervals)}, api_key={key_state}")
        print(f"[corpus] output: {out_tsv}")

    with out_tsv.open(file_mode, encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CORPUS_FIELDNAMES, delimiter="\t")
        if write_header:
            writer.writeheader()

        for date_from, date_to, label in intervals:
            if max_records is not None and added >= max_records:
                break

            if custom_query:
                q = apply_date_filter(custom_query, date_from, date_to)
            else:
                q = build_mesh_query(mesh_terms, exclude_pts, date_from, date_to)
            try:
                id_list = esearch_pmids(q, sleep_s)
            except Exception as exc:
                print(f"[corpus] esearch error ({label}): {exc}")
                continue

            new_pmids = [pmid for pmid in id_list if pmid not in existing]
            if not quiet:
                print(
                    f"[corpus] {label}: found={len(id_list)}, "
                    f"new={len(new_pmids)}, kept_so_far={added}"
                )

            if not new_pmids:
                continue

            for pmid_batch in chunked(new_pmids, batch_size):
                if max_records is not None:
                    remaining = max_records - added
                    if remaining <= 0:
                        break
                    pmid_batch = pmid_batch[:remaining]

                try:
                    articles = efetch_articles(pmid_batch, sleep_s)
                except Exception as exc:
                    print(f"[corpus] efetch error (batch {pmid_batch[0]}…): {exc}")
                    continue

                processed_batches += 1
                for art in articles:
                    row = row_from_article(art)
                    if row is None:
                        continue
                    pmid = row["PMID"]
                    if pmid in existing:
                        continue
                    writer.writerow(row)
                    existing.add(pmid)
                    added += 1
                    if max_records is not None and added >= max_records:
                        break

                out_f.flush()
                if not quiet and added and added % 100 == 0:
                    print(f"[corpus] added {added} new records…")

                if max_records is not None and added >= max_records:
                    break

    n_total = count_corpus_rows(out_tsv)
    write_metadata(
        out_meta,
        cfg,
        query_example=query_example,
        n_records=n_total,
        n_added=added,
        mode=mode,
        start_year=sy,
        end_year=ey,
    )

    summary = {
        "mode": mode,
        "corpus_tsv": str(out_tsv),
        "corpus_metadata": str(out_meta),
        "n_added": added,
        "n_records": n_total,
        "api_key_used": bool(resolved_key),
        "query_example": query_example,
        "batches": processed_batches,
    }
    if not quiet:
        print(
            f"[corpus] done: added={added}, total_in_file={n_total}, "
            f"metadata={out_meta}"
        )
    return summary
