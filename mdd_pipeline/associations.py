#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Association extraction: sentence-level entity-trigger-entity triples.

Uses character-span distances, trigger-between bonus, rule Direction for
Object1/Object2 order, and longest-trigger preference on overlaps.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config_utils import ensure_parent, resolve_path
from .rules_loader import Rule, index_rules_by_pair, load_rules

csv.field_size_limit(10_000_000)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")

# Species surface forms that are not infectious agents
SPECIES_STOPWORDS = {
    "patient",
    "patients",
    "man",
    "men",
    "woman",
    "women",
    "child",
    "children",
    "adult",
    "adults",
    "individual",
    "individuals",
    "volunteer",
    "volunteers",
    "person",
    "people",
    "human",
    "humans",
    "homo sapiens",
    "subject",
    "subjects",
    "participant",
    "participants",
    "donor",
    "donors",
    "control",
    "controls",
    "case",
    "cases",
    "outpatient",
    "outpatients",
    "inpatient",
    "inpatients",
    "respondent",
    "respondents",
    "veteran",
    "veterans",
    "persons",
    "people",
    "mouse",
    "mice",
    "rat",
    "rats",
    "rodent",
    "rodents",
}

ASSOC_FIELDNAMES = [
    "PMID",
    "Section",
    "Sentence",
    "Object1",
    "Object1Type",
    "Object2",
    "Object2Type",
    "Trigger",
    "Predicate",
    "EntityPair",
    "Direction",
    "Stem",
    "Distance",
    "Probability",
    "DecisionNote",
]


@dataclass
class EntityMention:
    text: str
    etype: str
    start: int  # absolute offset in section
    end: int
    normalized_name: str = ""
    env_group: str = ""
    env_subgroup: str = ""

    @property
    def display(self) -> str:
        return self.normalized_name or self.text


@dataclass
class TriggerHit:
    rule: Rule
    start: int  # offset in sentence
    end: int
    surface: str


@dataclass
class Candidate:
    e1: EntityMention  # Object1 after direction apply
    e2: EntityMention
    trigger: TriggerHit
    distance: float
    trigger_between: bool
    probability: float
    n_rivals: int = 0


def split_sentences_with_offsets(text: str) -> List[Tuple[str, int]]:
    """Return list of (sentence_text, start_offset_in_section)."""
    if not text or not text.strip() or text.strip() == "-":
        return []
    spans: List[Tuple[str, int]] = []
    start = 0
    for m in SENTENCE_SPLIT_RE.finditer(text):
        end = m.start()
        chunk = text[start:end]
        sent = chunk.strip()
        if sent:
            pad = len(chunk) - len(chunk.lstrip())
            spans.append((sent, start + pad))
        start = m.end()
    if start < len(text):
        chunk = text[start:]
        sent = chunk.strip()
        if sent:
            pad = len(chunk) - len(chunk.lstrip())
            spans.append((sent, start + pad))
    return spans


def _span_edge_distance(a0: int, a1: int, b0: int, b1: int) -> int:
    """Min distance between two closed-open [start, end) spans."""
    if a1 <= b0:
        return b0 - a1
    if b1 <= a0:
        return a0 - b1
    return 0  # overlap


def _is_between(trig0: int, trig1: int, e10: int, e11: int, e20: int, e21: int) -> bool:
    mid = (trig0 + trig1) // 2
    if e11 <= e20:
        return e11 <= mid <= e20
    if e21 <= e10:
        return e21 <= mid <= e10
    return False


def score_triple(
    e1: EntityMention,
    e2: EntityMention,
    trig: TriggerHit,
    sent_abs_start: int,
) -> Tuple[float, bool, float]:
    """
    Returns (distance, trigger_between, probability).

    Distance = d(e1,e2) + d(e1,trig) + d(e2,trig) in characters (sentence-local).
    """
    # Convert absolute entity offsets to sentence-local
    e10, e11 = e1.start - sent_abs_start, e1.end - sent_abs_start
    e20, e21 = e2.start - sent_abs_start, e2.end - sent_abs_start
    t0, t1 = trig.start, trig.end

    d12 = _span_edge_distance(e10, e11, e20, e21)
    d1t = _span_edge_distance(e10, e11, t0, t1)
    d2t = _span_edge_distance(e20, e21, t0, t1)
    distance = float(d12 + d1t + d2t)

    between = _is_between(t0, t1, e10, e11, e20, e21)
    adj = distance * 0.5 if between else distance
    # Soft probability in (0, 1]; scaled by adjusted character distance
    probability = 1.0 / (1.0 + adj / 20.0)
    return distance, between, probability


def order_by_direction(
    left: EntityMention,
    right: EntityMention,
    rule: Rule,
) -> Tuple[EntityMention, EntityMention]:
    """
    Map two mentions to Object1/Object2 using rule.direction and EntityPair A-B.
    `left`/`right` are in text order.
    """
    a, b = rule.type_a, rule.type_b
    if rule.direction == "<->":
        return left, right
    if rule.direction == "->":
        # Object1 = type A, Object2 = type B
        if left.etype == a and right.etype == b:
            return left, right
        if left.etype == b and right.etype == a:
            return right, left
        return left, right
    # <- : Object1 = type B, Object2 = type A
    if left.etype == b and right.etype == a:
        return left, right
    if left.etype == a and right.etype == b:
        return right, left
    return left, right


def find_triggers(sentence: str, rules: Sequence[Rule]) -> List[TriggerHit]:
    """Find trigger hits; drop shorter triggers fully covered by a longer one."""
    hits: List[TriggerHit] = []
    for rule in rules:
        for m in rule.trigger_re.finditer(sentence):
            hits.append(
                TriggerHit(
                    rule=rule,
                    start=m.start(),
                    end=m.end(),
                    surface=m.group(0),
                )
            )
    if not hits:
        return []
    # Longer spans first; drop shorter overlaps
    hits.sort(key=lambda h: (-(h.end - h.start), h.start, h.rule.predicate))
    kept: List[TriggerHit] = []
    occupied: List[Tuple[int, int]] = []
    for h in hits:
        if any(not (h.end <= a or h.start >= b) for a, b in occupied):
            continue
        kept.append(h)
        occupied.append((h.start, h.end))
    return kept


def filter_entity(ent: EntityMention) -> bool:
    if ent.etype == "Species" and ent.text.strip().lower() in SPECIES_STOPWORDS:
        return False
    if not ent.text or not ent.text.strip():
        return False
    return True


def extract_from_sentence(
    sentence: str,
    sent_abs_start: int,
    entities: Sequence[EntityMention],
    rules_by_pair: Dict[str, List[Rule]],
) -> List[Candidate]:
    ents = [e for e in entities if filter_entity(e)]
    if len(ents) < 2:
        return []

    # Collect relevant rules for type pairs present
    type_set = {e.etype for e in ents}
    relevant_rules: List[Rule] = []
    seen_rule_ids: set[int] = set()
    for t1 in type_set:
        for t2 in type_set:
            for r in rules_by_pair.get(f"{t1}-{t2}", []):
                rid = id(r)
                if rid not in seen_rule_ids:
                    seen_rule_ids.add(rid)
                    relevant_rules.append(r)
    if not relevant_rules:
        return []

    triggers = find_triggers(sentence, relevant_rules)
    if not triggers:
        return []

    # Build candidates for every unordered entity pair + compatible trigger
    bucket: Dict[Tuple[int, int, str], List[Candidate]] = defaultdict(list)
    # key: (id(e_left), id(e_right), predicate) after direction → compete on distance

    for i, ea in enumerate(ents):
        for eb in ents[i + 1 :]:
            if ea.text.lower() == eb.text.lower() and ea.etype == eb.etype:
                continue
            pair_key = f"{ea.etype}-{eb.etype}"
            pair_rules = rules_by_pair.get(pair_key, [])
            if not pair_rules:
                continue
            # Text order
            if ea.start <= eb.start:
                left, right = ea, eb
            else:
                left, right = eb, ea

            for th in triggers:
                rule = th.rule
                # Rule must match this type pair (either orientation)
                if {rule.type_a, rule.type_b} != {ea.etype, eb.etype}:
                    continue
                if rule.type_a == rule.type_b and ea.etype != rule.type_a:
                    continue
                o1, o2 = order_by_direction(left, right, rule)
                # Type match after ordering
                if rule.direction == "->":
                    if o1.etype != rule.type_a or o2.etype != rule.type_b:
                        continue
                elif rule.direction == "<-":
                    if o1.etype != rule.type_b or o2.etype != rule.type_a:
                        continue
                else:  # <->
                    if {o1.etype, o2.etype} != {rule.type_a, rule.type_b}:
                        continue

                dist, between, prob = score_triple(o1, o2, th, sent_abs_start)
                # Same entity-span pair: keep best by probability
                key = (id(left), id(right))
                bucket[key].append(
                    Candidate(
                        e1=o1,
                        e2=o2,
                        trigger=TriggerHit(rule, th.start, th.end, th.surface),
                        distance=dist,
                        trigger_between=between,
                        probability=prob,
                    )
                )

    winners: List[Candidate] = []
    for cands in bucket.values():
        cands.sort(key=lambda c: (-c.probability, c.distance, -len(c.trigger.surface)))
        best = cands[0]
        best.n_rivals = len(cands) - 1
        winners.append(best)
    return winners


def _load_entities_by_doc_section(entities_tsv: Path) -> Dict[Tuple[str, str], List[EntityMention]]:
    out: Dict[Tuple[str, str], List[EntityMention]] = defaultdict(list)
    with entities_tsv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pmid = (row.get("PMID") or "").strip()
            section = (row.get("Section") or "").strip()
            if not pmid or not section:
                continue
            try:
                start = int(row.get("Start") or 0)
                end = int(row.get("End") or 0)
            except ValueError:
                continue
            etype = (row.get("EntityType") or "").strip()
            text = row.get("Text") or ""
            out[(pmid, section)].append(
                EntityMention(
                    text=text,
                    etype=etype,
                    start=start,
                    end=end,
                    normalized_name=(row.get("NormalizedName") or "").strip(),
                    env_group=(row.get("EnvGroup") or "").strip(),
                    env_subgroup=(row.get("EnvSubgroup") or "").strip(),
                )
            )
    return out


def _candidate_to_row(
    pmid: str,
    section: str,
    sentence: str,
    cand: Candidate,
) -> Dict[str, str]:
    note = (
        f"distance={cand.distance:.0f}; "
        f"trigger_between={int(cand.trigger_between)}; "
        f"rivals={cand.n_rivals}; "
        f"score=1/(1+adj/20)"
    )
    return {
        "PMID": pmid,
        "Section": section,
        "Sentence": sentence,
        "Object1": cand.e1.display,
        "Object1Type": cand.e1.etype,
        "Object2": cand.e2.display,
        "Object2Type": cand.e2.etype,
        "Trigger": cand.trigger.surface,
        "Predicate": cand.trigger.rule.predicate,
        "EntityPair": cand.trigger.rule.entity_pair,
        "Direction": cand.trigger.rule.direction,
        "Stem": cand.trigger.rule.stem,
        "Distance": f"{cand.distance:.0f}",
        "Probability": f"{cand.probability:.4f}",
        "DecisionNote": note,
    }


def run_associations_stage(
    cfg: Dict[str, Any],
    *,
    max_docs: Optional[int] = None,
    corpus_tsv: Optional[Path] = None,
    entities_tsv: Optional[Path] = None,
    rules_txt: Optional[Path] = None,
    associations_tsv: Optional[Path] = None,
) -> Dict[str, Any]:
    corpus_path = Path(corpus_tsv) if corpus_tsv else resolve_path(cfg, "corpus_tsv")
    entities_path = Path(entities_tsv) if entities_tsv else resolve_path(cfg, "entities_tsv")
    rules_path = Path(rules_txt) if rules_txt else resolve_path(cfg, "rules_txt")
    out_path = (
        Path(associations_tsv)
        if associations_tsv
        else resolve_path(cfg, "associations_tsv")
    )

    for p, label in [
        (corpus_path, "corpus"),
        (entities_path, "entities"),
        (rules_path, "rules"),
    ]:
        if not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")

    print(f"Loading rules: {rules_path}")
    rules = load_rules(rules_path)
    rules_by_pair = index_rules_by_pair(rules)
    print(f"  triggers: {len(rules)}; pair keys: {len(rules_by_pair)}")

    print(f"Loading entities: {entities_path}")
    ents_index = _load_entities_by_doc_section(entities_path)
    print(f"  doc-sections with entities: {len(ents_index)}")

    ensure_parent(out_path)
    text_sections = list(
        (cfg.get("ner") or {}).get("text_sections")
        or ["ArticleTitle", "Abstract"]
    )

    n_docs = 0
    n_sentences = 0
    n_assoc = 0
    by_pair: Dict[str, int] = defaultdict(int)

    with corpus_path.open(encoding="utf-8", newline="") as f_in, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in, delimiter="\t")
        writer = csv.DictWriter(f_out, fieldnames=ASSOC_FIELDNAMES, delimiter="\t")
        writer.writeheader()

        for row in reader:
            pmid = (row.get("PMID") or "").strip()
            if not pmid:
                continue
            n_docs += 1
            if max_docs is not None and n_docs > max_docs:
                break

            for section in text_sections:
                text = row.get(section, "") or ""
                if not str(text).strip() or str(text).strip() == "-":
                    continue
                section_ents = ents_index.get((pmid, section), [])
                if len(section_ents) < 2:
                    continue

                for sent, abs_start in split_sentences_with_offsets(text):
                    n_sentences += 1
                    sent_end = abs_start + len(sent)
                    local_ents = [
                        e
                        for e in section_ents
                        if e.start >= abs_start and e.end <= sent_end
                    ]
                    if len(local_ents) < 2:
                        continue
                    winners = extract_from_sentence(
                        sent, abs_start, local_ents, rules_by_pair
                    )
                    for cand in winners:
                        writer.writerow(_candidate_to_row(pmid, section, sent, cand))
                        n_assoc += 1
                        by_pair[cand.trigger.rule.entity_pair] += 1

            if n_docs % 500 == 0:
                print(f"associations progress: docs={n_docs} assoc={n_assoc}")
                f_out.flush()

    summary = {
        "stage": "associations",
        "corpus_tsv": str(corpus_path),
        "entities_tsv": str(entities_path),
        "rules_txt": str(rules_path),
        "associations_tsv": str(out_path),
        "n_docs": n_docs if max_docs is None else min(n_docs, max_docs or n_docs),
        "n_sentences_scanned": n_sentences,
        "n_associations": n_assoc,
        "associations_by_pair": dict(sorted(by_pair.items(), key=lambda x: -x[1])),
        "n_rules_triggers": len(rules),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_path.with_name("associations_metadata.json")
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
