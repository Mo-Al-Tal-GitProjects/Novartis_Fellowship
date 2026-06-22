# Project background

## Why CIPHER exists

CIPHER—**Content-based Image Pathology: Histology Embedding Retrieval**—is a standalone retrospective reconstruction of a computational pathology research idea first explored during Fall 2024.

The original work took place through the Break Through Tech AI Fall AI/ML Challenge Project in connection with Novartis. The working research title was **“Investigating Pre-Trained Image Models for Content-Based Image Retrieval in Histopathology.”** Challenge guidance was provided by Dr. Walter Georgescu and Dr. Xiaoyan Zhang.

The original objective was to compare pre-trained image encoders for histopathology content-based image retrieval. The investigated model families were PLIP, UNI, CLIP, and DINO. The source data came from the PubMed portion of the ARCH dataset, and an advisor supplied a small set of pathologist-curated good matches for selected queries.

## Relationship to the 2024 prototype

The Fall 2024 prototype was notebook-driven and did not reach the reproducibility or scientific-validity standard required for a durable research project. Its data flow, model provenance, and evaluation protocol contained material problems. In particular, the prototype did not consistently issue the intended ARCH queries, did not reliably separate self-retrieval from relevant retrieval, and did not preserve enough provenance to support model-comparison claims. Its nominal PLIP embedding path also executed the OpenAI CLIP checkpoint rather than the intended pathology-adapted PLIP weights; CIPHER treats that output as invalid for PLIP comparison and does not reuse it.

CIPHER is therefore a new implementation of the research objective, not a polished release of the old code. Legacy notebooks, generated arrays, and retrospective audit drafts are deliberately absent from the active repository. Git history may preserve historical files, but they are not part of the CIPHER workflow and should not be cited as CIPHER results.

## Research objective

CIPHER asks:

> What properties of image embeddings from pre-trained vision and pathology models are associated with correct pathologist-curated matches in histopathology image retrieval?

The project will measure retrieval quality, embedding-space behavior, sensitivity to preprocessing and similarity metrics, cross-model agreement, robustness, and failure modes. It will also support an optional explanatory annotation layer for stain, morphology, tissue context, magnification, and artifacts. New annotations will remain explicitly separate from source metadata and pathologist-curated relevance judgments.

## Dataset and ground truth

The benchmark uses 11 ARCH PubMed query images. For each query, a pathologist selected a small set of good match images. The supplied folder contains incomplete rank sequences, repeated labels, alternate encodings, and one sixth match. CIPHER uses a documented adjudication policy:

- visually duplicate alternate encodings for `1024_05` and `2012_01` are represented once;
- the two files labeled `2012_02` are visually distinct and remain separate positives;
- `3216_06` remains a positive because it is present in the curated source folder;
- missing ranks are not invented;
- all other valid observed curated files remain positives.

This yields 49 distinct positive gallery images across 11 queries. Relevance is binary: an image is positive if it belongs to the adjudicated pathologist-curated set for that query. Unjudged gallery images are operational negatives, not proof of biological dissimilarity.

The downloadable ARCH PubMed subset also contains exact duplicate images under different UUIDs. CIPHER retains one deterministic canonical record per exact image hash and excludes the duplicate record from the retrieval gallery. If an exact duplicate ever involves a benchmark query, the query is retained and its gallery duplicate is excluded.

## Attribution and non-endorsement

This repository is an unofficial personal research reconstruction. It is not:

- an official Novartis or NIBR project release;
- an official Break Through Tech deliverable;
- an approved publication or clinical system;
- endorsed by the named advisors, former teammates, institutions, or partner organizations.

Names and historical affiliations appear only to explain the origin of the research question. All reconstruction choices, new code, analyses, interpretations, errors, and omissions belong to the repository author.

## Time boundary

- **Fall 2024:** original challenge exploration.
- **2026 reconstruction:** clean implementation, corrected benchmark definition, reproducible model evaluation, new scientific analyses, documentation, and an eventual unofficial manuscript.

Results produced by the reconstruction must be labeled as new CIPHER analyses rather than attributed retroactively to the 2024 challenge team.
