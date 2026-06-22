# Project overview

CIPHER—Content-based Image Pathology: Histology Embedding Retrieval—is a standalone
research workflow for studying whether frozen image embeddings retrieve
pathologist-curated histopathology matches. It compares CLIP, original DINO ViT-B/16,
PLIP, and original UNI ViT-L/16.

The implemented workflow covers data preparation, ground-truth adjudication, pinned
model inference, aligned embedding generation, exact retrieval, ranking metrics,
query-bootstrap uncertainty, model agreement, norm diagnostics, preprocessing
ablations, figures, reports, and an unofficial manuscript. Every terminal command is
available through `cipher`.

This repository is an independent retrospective reconstruction. Historical context
is documented in `BACKGROUND.md`; it is not an official institutional release or
endorsed publication.

## Scientific position

CIPHER treats biological relevance as an observed judgment, not an inference from
visual proximity. The current benchmark has 11 queries and 49 curated positives.
Unjudged images remain unknown rather than confirmed false biological matches.

The current evidence supports analysis of query heterogeneity, metric sensitivity,
model disagreement, vector-norm effects, grayscale sensitivity, and framing
sensitivity. It does not support general clinical performance claims or causal
attribution to tissue, stain, morphology, disease, or magnification.
