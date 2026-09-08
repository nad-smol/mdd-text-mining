#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone corpus clustering for study-design exploration.

TF-IDF on title+abstract (stopwords/boilerplate stripped) plus sparse study-type,
PublicationTypes, and MeSH features. Short/empty abstracts are excluded.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import yaml
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from mdd_pipeline.config_utils import load_config, resolve_path

csv.field_size_limit(10_000_000)

# Boilerplate / generic scientific English that rarely separates study designs
SCIENTIFIC_STOPWORDS = {
    "study", "studies", "studied", "result", "results", "method", "methods",
    "objective", "objectives", "aim", "aims", "background", "conclusion",
    "conclusions", "conclude", "concluded", "purpose", "introduction",
    "patient", "patients", "participant", "participants", "subject", "subjects",
    "group", "groups", "compared", "comparison", "significant", "significantly",
    "significance", "associated", "association", "associations", "including",
    "included", "include", "using", "used", "use", "based", "data", "analysis",
    "analyzed", "analysed", "may", "also", "however", "therefore", "thus",
    "among", "within", "between", "across", "related", "respectively",
    "found", "showed", "shown", "suggest", "suggests", "suggested",
    "reported", "report", "present", "presented", "investigate", "investigated",
    "evaluate", "evaluated", "assess", "assessed", "determine", "determined",
    "effect", "effects", "level", "levels", "rate", "rates", "higher", "lower",
    "increased", "decreased", "increase", "decrease", "versus", "vs",
    "year", "years", "month", "months", "week", "weeks", "day", "days",
    "one", "two", "three", "four", "five", "first", "second", "total",
    "new", "novel", "current", "recent", "further", "additional", "well",
    "whether", "role", "evidence", "known", "potential", "important",
}

# Domain terms that dominate MDD abstracts; removing them helps method clusters
DISEASE_STOPWORDS = {
    "depression", "depressive", "depressed", "mdd", "major", "disorder",
    "disorders", "antidepressant", "antidepressants", "mood", "psychiatric",
    "psychiatry", "mental", "symptom", "symptoms", "illness", "unipolar",
    "affective",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster MDD corpus abstracts by study design signals")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Corpus TSV (default: paths.corpus_tsv from config)",
    )
    p.add_argument(
        "--keywords",
        type=Path,
        default=None,
        help="YAML dictionary of study types (default: data/study_type_keywords.yaml)",
    )
    p.add_argument(
        "--out-tsv",
        type=Path,
        default=None,
        help="Output TSV path (default: ../outputs/clustering/corpus_clusters.tsv)",
    )
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--max-docs", type=int, default=0, help="0 = all rows")
    p.add_argument("--min-df", type=int, default=5)
    p.add_argument("--max-features", type=int, default=8000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--keep-disease-terms",
        action="store_true",
        help="Do NOT remove depression/MDD boilerplate from text features",
    )
    p.add_argument(
        "--no-meta-features",
        action="store_true",
        help="Use only cleaned text TF-IDF (no MeSH / PublicationType / rule features)",
    )
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument(
        "--min-abstract-chars",
        type=int,
        default=80,
        help="Skip docs whose Abstract is shorter than this (after strip); "
             "empty/'-' abstracts are always skipped",
    )
    p.add_argument(
        "--include-no-abstract",
        action="store_true",
        help="Do NOT filter out missing/short abstracts (legacy behaviour)",
    )
    return p.parse_args()


def load_study_type_dict(path: Path) -> List[Tuple[str, str, List[str]]]:
    """
    Returns list of (type_id, label, keywords_sorted_longest_first).
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items: List[Tuple[str, str, List[str]]] = []
    for type_id, info in (data.get("study_types") or {}).items():
        label = info.get("label") or type_id
        kws = [k.strip() for k in (info.get("keywords") or []) if k and str(k).strip()]
        kws = sorted(set(kws), key=lambda s: (-len(s), s.lower()))
        items.append((type_id, label, kws))
    return items


def combine_title_abstract(title: str, abstract: str) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if abstract in {"", "-"}:
        abstract = ""
    if title and abstract:
        return f"{title} {abstract}"
    return title or abstract


def match_study_types(
    text: str,
    study_types: Sequence[Tuple[str, str, List[str]]],
) -> Tuple[List[str], List[str]]:
    """
    Return (matched_labels_semicolon_order, matched_keyword_phrases).
    A type is included if any of its keywords occur as a case-insensitive substring.
    """
    text_l = text.lower()
    matched_labels: List[str] = []
    matched_kws: List[str] = []
    seen_kw: Set[str] = set()

    for _type_id, label, keywords in study_types:
        hit_kws = [kw for kw in keywords if kw.lower() in text_l]
        if not hit_kws:
            continue
        matched_labels.append(label)
        for kw in hit_kws:
            key = kw.lower()
            if key not in seen_kw:
                seen_kw.add(key)
                matched_kws.append(kw)

    return matched_labels, matched_kws


_token_re = re.compile(r"[a-z0-9][a-z0-9\-']+")


def build_stop_set(keep_disease_terms: bool) -> Set[str]:
    stops = set(SCIENTIFIC_STOPWORDS)
    # sklearn English stop list is large; fold in a compact core manually + SCIENTIFIC
    # Full ENGLISH_STOP_WORDS imported lazily where available
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stops |= set(ENGLISH_STOP_WORDS)
    except Exception:
        pass
    if not keep_disease_terms:
        stops |= DISEASE_STOPWORDS
    return stops


def normalize_for_tfidf(text: str, stop_set: Set[str]) -> str:
    tokens = _token_re.findall(text.lower())
    kept = [t for t in tokens if t not in stop_set and len(t) > 2]
    return " ".join(kept)


def split_meta_field(value: str) -> List[str]:
    if not value or value.strip() in {"", "-"}:
        return []
    parts = re.split(r"[;|]", value)
    return [p.strip() for p in parts if p.strip()]


def abstract_usable(abstract: str, min_chars: int) -> bool:
    a = (abstract or "").strip()
    if not a or a == "-":
        return False
    return len(a) >= min_chars


def load_corpus_rows(
    corpus_path: Path,
    max_docs: int,
    *,
    min_abstract_chars: int = 80,
    require_abstract: bool = True,
) -> Tuple[List[dict], List[dict]]:
    """
    Returns (rows_for_clustering, excluded_rows_with_reason).
    Excluded rows get an extra key _exclude_reason.
    max_docs applies to clustering-eligible rows only.
    """
    rows: List[dict] = []
    excluded: List[dict] = []
    with corpus_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            title = (row.get("ArticleTitle") or "").strip()
            abstract = (row.get("Abstract") or "").strip()
            text = combine_title_abstract(title, abstract)
            if not text.strip():
                skipped = dict(row)
                skipped["_exclude_reason"] = "empty_title_and_abstract"
                excluded.append(skipped)
                continue

            if require_abstract and not abstract_usable(abstract, min_abstract_chars):
                skipped = dict(row)
                if not abstract or abstract == "-":
                    skipped["_exclude_reason"] = "missing_abstract"
                else:
                    skipped["_exclude_reason"] = f"short_abstract_lt_{min_abstract_chars}"
                excluded.append(skipped)
                continue

            rows.append(row)
            if max_docs and len(rows) >= max_docs:
                break
    return rows, excluded


def write_excluded(path: Path, excluded: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["PMID", "Year", "ArticleTitle", "Abstract", "ExcludeReason"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row in excluded:
            w.writerow(
                {
                    "PMID": row.get("PMID", ""),
                    "Year": row.get("Year", ""),
                    "ArticleTitle": row.get("ArticleTitle", ""),
                    "Abstract": row.get("Abstract", ""),
                    "ExcludeReason": row.get("_exclude_reason", ""),
                }
            )


def build_feature_matrix(
    rows: Sequence[dict],
    study_types: Sequence[Tuple[str, str, List[str]]],
    *,
    keep_disease_terms: bool,
    use_meta_features: bool,
    min_df: int,
    max_features: int,
) -> Tuple[
    sparse.spmatrix,
    sparse.spmatrix,
    TfidfVectorizer,
    List[List[str]],
    List[List[str]],
    List[str],
]:
    stop_set = build_stop_set(keep_disease_terms)

    cleaned_texts: List[str] = []
    rule_labels_per_doc: List[List[str]] = []
    rule_kws_per_doc: List[List[str]] = []
    raw_texts: List[str] = []

    type_ids = [t[0] for t in study_types]
    type_id_to_idx = {t: i for i, t in enumerate(type_ids)}
    label_to_id = {lab: tid for tid, lab, _ in study_types}

    rule_mat = sparse.lil_matrix((len(rows), len(type_ids)), dtype=np.float32)

    pub_docs: List[str] = []
    mesh_docs: List[str] = []

    for i, row in enumerate(rows):
        raw = combine_title_abstract(row.get("ArticleTitle", ""), row.get("Abstract", ""))
        raw_texts.append(raw)
        labels, kws = match_study_types(raw, study_types)
        rule_labels_per_doc.append(labels)
        rule_kws_per_doc.append(kws)

        for lab in labels:
            tid = label_to_id.get(lab)
            if tid is not None:
                rule_mat[i, type_id_to_idx[tid]] = 1.0

        cleaned_texts.append(normalize_for_tfidf(raw, stop_set))
        pubs = split_meta_field(row.get("PublicationTypes", ""))
        meshes = split_meta_field(row.get("MeshTerms", ""))
        pub_docs.append(" ".join(f"pub_{re.sub(r'[^a-z0-9]+', '_', p.lower())}" for p in pubs))
        mesh_docs.append(" ".join(f"mesh_{re.sub(r'[^a-z0-9]+', '_', m.lower())}" for m in meshes))

    text_vec = TfidfVectorizer(
        min_df=min_df,
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_text = text_vec.fit_transform(cleaned_texts)

    blocks = [X_text]
    if use_meta_features:
        blocks.append(rule_mat.tocsr())
        pub_vec = TfidfVectorizer(min_df=2, max_features=200)
        mesh_vec = TfidfVectorizer(min_df=5, max_features=1500)
        blocks.append(pub_vec.fit_transform(pub_docs))
        blocks.append(mesh_vec.fit_transform(mesh_docs))

    X = sparse.hstack(blocks, format="csr")
    X = normalize(X, norm="l2", axis=1)
    return X, X_text, text_vec, rule_labels_per_doc, rule_kws_per_doc, raw_texts


def top_terms_per_cluster(
    X_text: sparse.spmatrix,
    labels: np.ndarray,
    feature_names: np.ndarray,
    n_top: int = 15,
) -> Dict[int, List[str]]:
    """Mean TF-IDF weight per cluster over text features only (first block)."""
    out: Dict[int, List[str]] = {}
    for cid in sorted(set(labels.tolist())):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            out[cid] = []
            continue
        centroid = np.asarray(X_text[idx].mean(axis=0)).ravel()
        top_idx = centroid.argsort()[::-1][:n_top]
        out[cid] = [feature_names[j] for j in top_idx if centroid[j] > 0]
    return out


def write_outputs(
    out_tsv: Path,
    rows: Sequence[dict],
    raw_texts: Sequence[str],
    rule_labels: Sequence[Sequence[str]],
    rule_kws: Sequence[Sequence[str]],
    clusters: Sequence[int],
    cluster_terms: Dict[int, List[str]],
) -> Path:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "PMID",
        "Text",
        "MatchedKeywords",
        "StudyTypes_rules",
        "Cluster",
    ]
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row, text, labs, kws, cid in zip(rows, raw_texts, rule_labels, rule_kws, clusters):
            w.writerow(
                {
                    "PMID": row.get("PMID", ""),
                    "Text": text,
                    "MatchedKeywords": "; ".join(kws),
                    "StudyTypes_rules": "; ".join(labs),
                    "Cluster": str(cid),
                }
            )

    summary_path = out_tsv.with_name(out_tsv.stem + "_summaries.tsv")
    counts = Counter(clusters)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["Cluster", "N_docs", "TopTerms", "ExamplePMIDs"],
            delimiter="\t",
        )
        w.writeheader()
        examples: Dict[int, List[str]] = defaultdict(list)
        for row, cid in zip(rows, clusters):
            if len(examples[cid]) < 5:
                examples[cid].append(row.get("PMID", ""))
        for cid in sorted(counts):
            w.writerow(
                {
                    "Cluster": cid,
                    "N_docs": counts[cid],
                    "TopTerms": "; ".join(cluster_terms.get(cid, [])),
                    "ExamplePMIDs": "; ".join(examples[cid]),
                }
            )
    return summary_path


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    corpus_path = Path(args.corpus) if args.corpus else resolve_path(cfg, "corpus_tsv")
    keywords_path = (
        Path(args.keywords)
        if args.keywords
        else (PIPELINE_DIR / "data" / "study_type_keywords.yaml")
    )
    out_tsv = (
        Path(args.out_tsv)
        if args.out_tsv
        else (resolve_path(cfg, "outputs_dir") / "clustering" / "corpus_clusters.tsv")
    )

    if not corpus_path.exists():
        print(f"[cluster] corpus not found: {corpus_path}", file=sys.stderr)
        return 1
    if not keywords_path.exists():
        print(f"[cluster] keywords file not found: {keywords_path}", file=sys.stderr)
        return 1

    study_types = load_study_type_dict(keywords_path)
    print(f"[cluster] loaded {len(study_types)} study-type groups from {keywords_path}")

    require_abstract = not args.include_no_abstract
    rows, excluded = load_corpus_rows(
        corpus_path,
        args.max_docs,
        min_abstract_chars=args.min_abstract_chars,
        require_abstract=require_abstract,
    )
    excl_path = out_tsv.with_name(out_tsv.stem + "_excluded.tsv")
    if excluded:
        write_excluded(excl_path, excluded)

    print(
        f"[cluster] documents for clustering: {len(rows)} "
        f"(excluded no/short abstract: {len(excluded)}) from {corpus_path}"
    )
    if excluded:
        reasons = Counter(r.get("_exclude_reason", "") for r in excluded)
        for reason, n in reasons.most_common():
            print(f"[cluster]   excluded {n}: {reason}")
        print(f"[cluster] wrote {excl_path}")

    if len(rows) < args.n_clusters:
        print(
            f"[cluster] not enough docs ({len(rows)}) for n_clusters={args.n_clusters}",
            file=sys.stderr,
        )
        return 1

    X, X_text, text_vec, rule_labels, rule_kws, raw_texts = build_feature_matrix(
        rows,
        study_types,
        keep_disease_terms=args.keep_disease_terms,
        use_meta_features=not args.no_meta_features,
        min_df=args.min_df,
        max_features=args.max_features,
    )
    print(f"[cluster] feature matrix: {X.shape}")

    km = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        batch_size=min(args.batch_size, len(rows)),
        n_init=10,
        reassignment_ratio=0.01,
    )
    labels = km.fit_predict(X)
    print(f"[cluster] finished MiniBatchKMeans k={args.n_clusters}")

    cluster_terms = top_terms_per_cluster(
        X_text, labels, np.array(text_vec.get_feature_names_out())
    )

    summary_path = write_outputs(
        out_tsv, rows, raw_texts, rule_labels, rule_kws, labels.tolist(), cluster_terms
    )
    n_typed = sum(1 for labs in rule_labels if labs)
    print(f"[cluster] wrote {out_tsv}")
    print(f"[cluster] wrote {summary_path}")
    print(f"[cluster] docs with >=1 rule study-type: {n_typed}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
