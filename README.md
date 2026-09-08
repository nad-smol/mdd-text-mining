# MDD Text-Mining Pipeline & Knowledge Graph

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PubMed](https://img.shields.io/badge/Data-PubMed%20%2F%20Entrez-green.svg)](https://pubmed.ncbi.nlm.nih.gov/)
[![Cytoscape Compatible](https://img.shields.io/badge/Cytoscape-XGMML%20%26%20HTML-orange.svg)](https://cytoscape.org/)

An automated, reproducible, literature-scale text-mining pipeline and knowledge graph extraction framework for **Major Depressive Disorder (MDD)**. 

This repository contains the complete open-source software pipeline used to mine biomedical associations across 38,000+ PubMed publications, extract multi-domain relationships (molecular hubs, pharmacotherapies, comorbidities, psychosocial risk factors, infectious agents, and microRNAs), compute evidence-backed confidence scores, and export interactive networks for Cytoscape and web browsers.

---

## Key Features

1. **Automated PubMed Retrieval & Incremental Updates:**
   - Retrieves MeSH-indexed MDD titles and abstracts (`Depressive Disorder, Major[Mesh] NOT Review[PT]`).
   - Supports incremental updating: checks for new PubMed articles published since the last run and appends them without rebuilding the entire database.

2. **Multi-Domain Named Entity Recognition (NER):**
   - **Chemicals, Diseases, Genes/Proteins, Species:** state-of-the-art biomedical NER via HunFlair2.
   - **MicroRNAs:** specialized regular expressions with variant canonicalization (e.g., `miR-146a`, `hsa-let-7a-5p`).
   - **Psychosocial & Environmental Risk Factors:** curated `DeprTerms` dictionary covering childhood adversity, social isolation, life events, sleep disruption, and economic strain.

3. **Sentence-Level Rule-Based Association Extraction:**
   - 43 typed biomedical predicates with defined syntax, trigger wordforms, and directional semantics (e.g., *is used in therapy of*, *is comorbid with*, *increases risk of*, *alters level of*, *is a biomarker of*).
   - Proximity scoring with trigger-between bonus and distance penalties.

4. **Multi-Ontology Entity Normalization:**
   - **Chemicals:** PubChem CID.
   - **Diseases:** Disease Ontology (DOID) with MONDO fallback via OLS4.
   - **Genes:** NCBI Gene (Entrez GeneID, default human).
   - **Species:** NCBI Taxonomy ID with strict lineage filtering for infectious/microbial pathogens.
   - Persistent local cache (`normalize_cache.json`) eliminates redundant API requests.

5. **Multi-Factorial Confidence Scoring:**
   - Confidence score calculated as the geometric mean of sentence occurrence frequency, document (PMID) recurrence, extraction probability, and entity normalization confidence.

6. **Interactive Visualization & Cytoscape Export:**
   - Exports `.xgmml` networks ready for desktop **Cytoscape**.
   - Exports standalone `.html` interactive graphs powered by **Cytoscape.js** for immediate web browser inspection.

---

## Repository Structure

```text
mdd-depression-text-mining/
├── README.md                      # Comprehensive user guide (English)
├── README_RU.md                   # User guide in Russian
├── LICENSE                        # Open-source MIT License
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore rules
├── config.yaml                    # Default pipeline configuration
├── run_pipeline.py                # Modular CLI for granular stage execution
├── run_all.py                     # Master runner (auto-recovery, updates, & demo)
├── data/                          # Dictionaries & extraction rules
│   ├── Env_factors.json           # DeprTerms dictionary patterns
│   ├── Env_factors.tsv            # DeprTerms table format
│   ├── rules.txt                  # Syntactic association rules & predicates
│   └── study_type_keywords.yaml   # Publication study-type keywords
├── mdd_pipeline/                  # Core Python package
│   ├── __init__.py
│   ├── config_utils.py            # Path resolution & configuration loader
│   ├── corpus.py                  # PubMed/Entrez retrieval & incremental updates
│   ├── ner.py                     # Multi-domain NER (HunFlair2 + miRNA + DeprTerms)
│   ├── associations.py            # Sentence-level relation extraction
│   ├── normalize.py               # Ontology normalization (PubChem, OLS4, NCBI)
│   ├── unique_associations.py     # Deduplication & confidence scoring
│   ├── graphs.py                  # XGMML & Cytoscape.js HTML graph export
│   ├── deprterms_dict.py          # Dictionary compiler
│   ├── mirna.py                   # Regex & canonicalization for microRNAs
│   └── rules_loader.py            # Rule parser & predicate indexer
├── tests/                         # Unit tests and smoke tests
│   ├── test_mirna.py
│   ├── test_associations.py
│   └── test_corpus_stage.py
├── scripts/                       # Helper scripts (clustering, dictionary builds)
│   ├── cluster_corpus.py
│   ├── expand_deprterms.py
│   └── build_rules.py
└── outputs/                       # Baseline deliverables & precomputed data
    ├── associations/
    │   ├── unique_associations.tsv          # Final deliverable: 146,802 scored associations
    │   └── unique_associations_metadata.json
    ├── normalization/
    │   ├── entity_catalog.tsv               # Preferred names & database xrefs
    │   ├── normalize_cache.json             # Precomputed API normalization cache
    │   └── normalized_mentions.tsv
    └── graphs/                              # Pre-built research subgraphs (.html & .xgmml)
        ├── fig05_therapy.html / .xgmml
        ├── fig06_risk_factors.html / .xgmml
        ├── fig07_comorbid.html / .xgmml
        ├── fig08_molecular.html / .xgmml
        ├── fig09_infections.html / .xgmml
        ├── fig10_bdnf.html / .xgmml
        └── overview_high_confidence.html / .xgmml
```

---

## System Requirements

- **Operating System:** Linux, macOS, or Windows 10/11 (64-bit).
- **Python:** Version 3.9 or higher (tested on Python 3.10, 3.11, 3.12).
- **Memory (RAM):** 8 GB minimum; 16 GB recommended for full corpus processing.
- **Hardware Acceleration:**
  - **GPU (Recommended):** NVIDIA GPU with CUDA support for accelerated HunFlair2 NER.
  - **CPU:** Fully supported (runs with `--cpu` flag).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mdd-depression-text-mining.git
cd mdd-depression-text-mining
```

### 2. Create and activate a virtual environment

```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional GPU acceleration)*: If you have an NVIDIA GPU, install PyTorch with CUDA support first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 4. Configure NCBI Entrez API Key (Recommended)

NCBI allows 3 requests/sec without an API key, and up to 10 requests/sec with a free API key.

1. Get a free API key at [NCBI Account Settings](https://www.ncbi.nlm.nih.gov/account/settings/).
2. Set the environment variable:
   ```bash
   # Linux / macOS
   export NCBI_API_KEY="your_ncbi_api_key"

   # Windows PowerShell
   $env:NCBI_API_KEY = "your_ncbi_api_key"
   ```
3. Update `ncbi.email` in `config.yaml` with your contact email (required by NCBI policy).

---

## Quick Start: Running the Pipeline

The repository includes a smart workflow manager: `run_all.py`.

### 1. Instant Status Check & Auto-Recovery

If you clone the repository as-is (with precomputed data included), running `python run_all.py` will inspect the files and report the system status:

```bash
python run_all.py
```

**Auto-Recovery ("Fool-Proofing"):** If you or a colleague accidentally delete `unique_associations.tsv` or the entire `outputs/` folder, running `python run_all.py` will detect the missing files and automatically run the required stages to rebuild them!

### 2. Fast Smoke-Test Demo (for Reviewers)

To test the entire pipeline end-to-end on your machine in 1–2 minutes without downloading tens of thousands of abstracts:

```bash
python run_all.py --mode demo
```

This downloads 50 recent MDD articles, performs NER, extracts syntactic associations, computes confidence scores, and builds an interactive HTML graph in `outputs/demo/graphs/demo_graph.html`.

### 3. Incremental Update (Adding New Research)

To update the dataset with newly published literature:

```bash
python run_all.py --mode update
```

This will:
1. Query PubMed for any articles published after the latest date in `outputs/corpus/mdd_corpus.tsv`.
2. Run NER only on the newly downloaded records.
3. Extract new associations and append them.
4. Normalize new mentions using the local cache.
5. Recompute unique associations and confidence scores.
6. Regenerate all Cytoscape and HTML graphs.

### 4. Regenerating Graphs Only

To re-export the standard research subgraphs from the precomputed associations:

```bash
python run_all.py --mode graphs
```

---

## Modular CLI: `run_pipeline.py`

For granular control, you can invoke individual pipeline stages directly via `run_pipeline.py`:

```bash
# 1. Corpus Download / Update
python run_pipeline.py corpus --mode update
python run_pipeline.py corpus --mode full --start-year 2015 --end-year 2024

# 2. Named Entity Recognition
python run_pipeline.py ner --mode full
python run_pipeline.py ner --mode update      # Process only new PMIDs
python run_pipeline.py ner --cpu              # Force CPU execution

# 3. Association Extraction
python run_pipeline.py associations
python run_pipeline.py associations --max-docs 1000

# 4. Entity Normalization
python run_pipeline.py normalize
python run_pipeline.py normalize --types Chemical,Disease

# 5. Deduplication & Confidence Scoring
python run_pipeline.py unique-associations

# 6. Graph Export
python run_pipeline.py graphs --max-nodes 80 --normalized-only --min-confidence 0.25
python run_pipeline.py graphs --predicates "is comorbid with" --stem mdd_comorbidities
python run_pipeline.py graphs --predicates "is used in therapy of" --stem mdd_pharmacotherapy
```

---

## Precomputed Deliverables & Output Formats

The primary deliverable of this research is stored in `outputs/associations/unique_associations.tsv` (146,802 unique associations).

### Structure of `unique_associations.tsv`

| Column | Example | Description |
|---|---|---|
| `Object1ID` | `0000412` | Database ID of the first entity (e.g. DOID for Disease, CID for Chemical, GeneID for Gene) |
| `Object1Type` | `Disease` | Entity type (`Chemical`, `Disease`, `Gene`, `Species`, `miRNA`, `DeprTerms`) |
| `Object1PrefName` | `anxiety` | Canonical / preferred name |
| `Object1TextSynonyms` | `anxiety; anxiety symptoms` | Text mentions observed in abstracts |
| `Object2ID` | `1470` | Database ID of the second entity |
| `Object2Type` | `Disease` | Entity type of the second entity |
| `Object2PrefName` | `major depressive disorder` | Canonical name of the second entity |
| `Object2TextSynonyms` | `depression; MDD; major depression` | Text mentions of the second entity |
| `Predicate` | `is comorbid with` | Extracted biomedical predicate |
| `Direction` | `<->` | Semantic relation direction (`->`, `<-`, or `<->`) |
| `ConfidenceScore` | `0.6482` | Geometric mean confidence score (0.0 to 1.0) |
| `n_sentences` | `842` | Number of supporting sentences in the corpus |
| `n_pmids` | `519` | Number of distinct PubMed articles supporting the edge |
| `SupportPMIDs` | `10572320; 11407273; ...` | Semicolon-separated list of supporting PMIDs |

---

## Visualization in Cytoscape & Browsers

### 1. Web Browser (Cytoscape.js)
Open any `.html` file from `outputs/graphs/` directly in Google Chrome, Firefox, Safari, or Edge:
- **`fig05_therapy.html`**: Antidepressant classes, augmentation strategies (lithium, esketamine).
- **`fig06_risk_factors.html`**: Early adversity, childhood trauma, loneliness, sleep disruption.
- **`fig07_comorbid.html`**: Psychiatric & somatic comorbidity network.
- **`fig08_molecular.html`**: BDNF, SLC6A4, COMT, MAOA, FKBP5, inflammatory cytokines.
- **`fig09_infections.html`**: HIV, HCV, SARS-CoV-2, microbiota.
- **`fig10_bdnf.html`**: Multi-predicate BDNF neurotrophic neighborhood.

### 2. Desktop Cytoscape
1. Launch [Cytoscape](https://cytoscape.org/) (v3.9+).
2. Go to **File -> Import -> Network from File...**
3. Select any `.xgmml` file from `outputs/graphs/`.
4. Nodes are color-coded by biological category:
   - **Chemicals:** Blue (`#7EB6D9`)
   - **Diseases:** Red / Pink (`#E8A0A0`)
   - **Genes / Proteins:** Green (`#8FCB8F`)
   - **Species:** Purple (`#C4A8E0`)
   - **microRNAs:** Teal (`#7EC8C3`)
   - **DeprTerms (Psychosocial):** Yellow / Ochre (`#E0C07A`)

---

## Adapting to Other Diseases / Biomedical Topics

The pipeline is topic-agnostic. To apply it to another biomedical domain (e.g., Alzheimer's disease, Schizophrenia, Parkinson's disease):

1. Open `config.yaml`.
2. Change `pubmed.mesh_terms`:
   ```yaml
   pubmed:
     mesh_terms:
       - "Alzheimer Disease"
   ```
3. Run `python run_all.py --mode full`.

---
