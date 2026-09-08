#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NER stage: HunFlair2, miRNA regex, and DeprTerms dictionary.

Uses CUDA when available; otherwise CPU.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .config_utils import ensure_parent, resolve_path
from .deprterms_dict import load_deprterms_patterns
from .mirna import find_mirnas

csv.field_size_limit(10_000_000)

ENTITY_FIELDNAMES = [
    "PMID",
    "Section",
    "Start",
    "End",
    "Text",
    "EntityType",
    "Source",
    "NormalizedID",
    "NormalizedName",
    "EnvGroup",
    "EnvSubgroup",
    "Confidence",
]

# Corpus text columns (title/abstract; extras if full text is present)
DEFAULT_TEXT_SECTIONS = [
    "ArticleTitle",
    "Abstract",
    "Introduction",
    "Methods",
    "Results",
    "Discussion",
    "Conclusions",
    "OtherSections",
    "FullText",
]

# HunFlair2 labels kept (case-insensitive); cell lines excluded
HUNFLAIR_KEEP = {
    "chemical",
    "disease",
    "gene",
    "species",
}
HUNFLAIR_SKIP = {
    "cell_line",
    "cell-line",
    "cellline",
    "cell line",
    "cell lines",
}

# HunFlair label → association entity type
HUNFLAIR_TYPE_MAP = {
    "chemical": "Chemical",
    "disease": "Disease",
    "gene": "Gene",
    "species": "Species",
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")


def _apply_windows_pathlib_patch() -> None:
    """Flair checkpoints sometimes pickle PosixPath; fix load on Windows."""
    if sys.platform.startswith("win"):
        import pathlib

        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]


def split_text_to_sentences_with_offsets(text: str) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    start = 0
    for m in SENTENCE_SPLIT_RE.finditer(text):
        end = m.start()
        sent = text[start:end].strip()
        if sent:
            # Offset of stripped sentence in original text
            local = text[start:end]
            pad = len(local) - len(local.lstrip())
            spans.append({"text": sent, "start": start + pad})
        start = m.end()
    if start < len(text):
        local = text[start:]
        sent = local.strip()
        if sent:
            pad = len(local) - len(local.lstrip())
            spans.append({"text": sent, "start": start + pad})
    return spans


def normalize_hunflair_label(raw: str) -> Optional[str]:
    key = re.sub(r"[\s\-]+", "_", (raw or "").strip().lower())
    key_spaced = (raw or "").strip().lower()
    if key in HUNFLAIR_SKIP or key_spaced in HUNFLAIR_SKIP:
        return None
    if key in HUNFLAIR_KEEP:
        return HUNFLAIR_TYPE_MAP[key]
    # Plural / alternate HunFlair label spellings
    aliases = {
        "chemicals": "Chemical",
        "diseases": "Disease",
        "genes": "Gene",
        "gene/protein": "Gene",
        "gene_protein": "Gene",
        "proteins": "Gene",
        "protein": "Gene",
    }
    if key in aliases:
        return aliases[key]
    return None


def setup_hunflair_device(prefer_gpu: bool = True):
    """Set flair.device to CUDA if possible. Returns (torch, flair, device)."""
    import torch
    import flair

    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        flair.device = device
        print(f"HunFlair2 device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        flair.device = device
        if prefer_gpu:
            print(
                "WARNING: CUDA not available; HunFlair2 falls back to CPU "
                "(much slower). Install a CUDA build of torch for GPU."
            )
        else:
            print("HunFlair2 device: CPU (forced)")
    return torch, flair, device


def load_hunflair_tagger(prefer_gpu: bool = True):
    """Load HunFlair2 NER tagger onto GPU when available."""
    _apply_windows_pathlib_patch()
    torch, flair, device = setup_hunflair_device(prefer_gpu=prefer_gpu)

    print("Loading HunFlair2 model...")
    try:
        from flair.models.prefixed_tagger import PrefixedSequenceTagger

        tagger = PrefixedSequenceTagger.load("hunflair/hunflair2-ner")
    except Exception:
        from flair.nn import Classifier

        tagger = Classifier.load("hunflair2")

    tagger.to(device)
    return tagger, device


def hunflair_ner_on_text(
    tagger,
    text: str,
    pmid: str,
    section: str,
) -> List[Dict[str, str]]:
    from flair.data import Sentence

    entities: List[Dict[str, str]] = []
    if not text or not str(text).strip() or str(text).strip() == "-":
        return entities

    sentence_spans = split_text_to_sentences_with_offsets(text)
    if not sentence_spans:
        return entities

    sentences = [Sentence(s["text"]) for s in sentence_spans]
    tagger.predict(sentences)

    for sent_info, sent in zip(sentence_spans, sentences):
        offset = sent_info["start"]
        spans = sent.get_spans("ner")
        if not spans:
            # Alternate Classifier API: labels instead of spans
            try:
                spans = [lab.data_point for lab in sent.get_labels("ner")]
            except Exception:
                spans = []

        for span in spans:
            label = span.get_label("ner")
            etype = normalize_hunflair_label(label.value)
            if etype is None:
                continue
            score = float(getattr(label, "score", 0.0) or 0.0)
            local_start = span.start_position
            local_end = span.end_position
            start = offset + local_start
            end = offset + local_end
            mention = text[start:end]
            entities.append(
                {
                    "PMID": pmid,
                    "Section": section,
                    "Start": str(start),
                    "End": str(end),
                    "Text": mention,
                    "EntityType": etype,
                    "Source": "HunFlair2",
                    "NormalizedID": "",
                    "NormalizedName": "",
                    "EnvGroup": "",
                    "EnvSubgroup": "",
                    "Confidence": f"{score:.4f}",
                }
            )
    return entities


def deprterms_ner_on_text(
    patterns: List[Dict[str, Any]],
    text: str,
    pmid: str,
    section: str,
) -> List[Dict[str, str]]:
    entities: List[Dict[str, str]] = []
    if not text or not str(text).strip() or str(text).strip() == "-":
        return entities

    for item in patterns:
        for m in item["regex"].finditer(text):
            entities.append(
                {
                    "PMID": pmid,
                    "Section": section,
                    "Start": str(m.start()),
                    "End": str(m.end()),
                    "Text": m.group(0),
                    "EntityType": "DeprTerms",
                    "Source": "Dictionary",
                    "NormalizedID": "",
                    "NormalizedName": item["canonical"],
                    "EnvGroup": item["env_group"],
                    "EnvSubgroup": item["env_subgroup"],
                    "Confidence": "",
                }
            )
    return entities


def mirna_ner_on_text(text: str, pmid: str, section: str) -> List[Dict[str, str]]:
    entities: List[Dict[str, str]] = []
    if not text or not str(text).strip() or str(text).strip() == "-":
        return entities

    for hit in find_mirnas(text):
        entities.append(
            {
                "PMID": pmid,
                "Section": section,
                "Start": str(hit.start),
                "End": str(hit.end),
                "Text": hit.text,
                "EntityType": "miRNA",
                "Source": "Regex",
                "NormalizedID": "",
                "NormalizedName": hit.canonical,
                "EnvGroup": "",
                "EnvSubgroup": "",
                "Confidence": "",
            }
        )
    return entities


def _existing_pmids(entities_tsv: Path) -> Set[str]:
    if not entities_tsv.is_file():
        return set()
    seen: Set[str] = set()
    with entities_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pmid = (row.get("PMID") or "").strip()
            if pmid:
                seen.add(pmid)
    return seen


def run_ner_stage(
    cfg: Dict[str, Any],
    *,
    mode: Optional[str] = None,
    max_docs: Optional[int] = None,
    prefer_gpu: bool = True,
    corpus_tsv: Optional[Path] = None,
    entities_tsv: Optional[Path] = None,
    env_json: Optional[Path] = None,
    text_sections: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run NER over corpus TSV.

    mode full: rewrite entities TSV; update: append PMIDs not yet present.
    """
    ner_cfg = cfg.get("ner") or {}
    mode = (mode or ner_cfg.get("mode") or "full").lower()
    if mode not in {"full", "update"}:
        raise ValueError(f"ner mode must be full|update, got {mode!r}")

    corpus_path = Path(corpus_tsv) if corpus_tsv else resolve_path(cfg, "corpus_tsv")
    out_path = Path(entities_tsv) if entities_tsv else resolve_path(cfg, "entities_tsv")
    env_path = Path(env_json) if env_json else resolve_path(cfg, "env_factors_json")
    sections = text_sections or list(ner_cfg.get("text_sections") or DEFAULT_TEXT_SECTIONS)

    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    if not env_path.is_file():
        raise FileNotFoundError(f"Env_factors JSON not found: {env_path}")

    prefer_gpu = bool(ner_cfg.get("prefer_gpu", prefer_gpu))
    tagger, device = load_hunflair_tagger(prefer_gpu=prefer_gpu)
    patterns = load_deprterms_patterns(env_path)
    print(f"DeprTerms patterns loaded: {len(patterns)}")

    skip_pmids: Set[str] = set()
    write_header = True
    open_mode = "w"
    if mode == "update" and out_path.is_file():
        skip_pmids = _existing_pmids(out_path)
        open_mode = "a"
        write_header = False
        print(f"Update mode: skipping {len(skip_pmids)} PMIDs already in {out_path}")
    else:
        ensure_parent(out_path)

    total_docs = 0
    processed_docs = 0
    skipped_docs = 0
    total_entities = 0
    by_type: Dict[str, int] = {}

    with corpus_path.open("r", encoding="utf-8", newline="") as f_in, out_path.open(
        open_mode, encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in, delimiter="\t")
        writer = csv.DictWriter(f_out, fieldnames=ENTITY_FIELDNAMES, delimiter="\t")
        if write_header:
            writer.writeheader()

        for row in reader:
            pmid = (row.get("PMID") or "").strip()
            if not pmid:
                continue
            total_docs += 1
            if pmid in skip_pmids:
                skipped_docs += 1
                continue

            doc_entities: List[Dict[str, str]] = []
            for section in sections:
                text = row.get(section, "") or ""
                if not str(text).strip() or str(text).strip() == "-":
                    continue
                doc_entities.extend(hunflair_ner_on_text(tagger, text, pmid, section))
                doc_entities.extend(deprterms_ner_on_text(patterns, text, pmid, section))
                doc_entities.extend(mirna_ner_on_text(text, pmid, section))

            for ent in doc_entities:
                writer.writerow(ent)
                et = ent["EntityType"]
                by_type[et] = by_type.get(et, 0) + 1
            total_entities += len(doc_entities)
            processed_docs += 1

            if processed_docs % 25 == 0:
                print(
                    f"NER progress: processed={processed_docs} "
                    f"entities={total_entities} (corpus row {total_docs})"
                )
                f_out.flush()

            if max_docs is not None and processed_docs >= max_docs:
                print(f"Stopped early: max_docs={max_docs}")
                break

    summary = {
        "stage": "ner",
        "mode": mode,
        "device": str(device),
        "corpus_tsv": str(corpus_path),
        "entities_tsv": str(out_path),
        "env_factors_json": str(env_path),
        "n_corpus_rows_seen": total_docs,
        "n_docs_processed": processed_docs,
        "n_docs_skipped": skipped_docs,
        "n_entities": total_entities,
        "entities_by_type": by_type,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_path.with_name("entities_metadata.json")
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
