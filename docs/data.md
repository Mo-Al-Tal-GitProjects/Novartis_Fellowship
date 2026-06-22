# Data and relevance judgments

CIPHER uses the PubMed portion of ARCH plus a separately supplied folder of curated
good matches. Dataset and curated-image licenses are independent of CIPHER's MIT
License; local data are excluded from Git.

## Prepared inventory

- 3,360 validated manifest records;
- 3,332 active records after exact-duplicate handling;
- 11 ARCH query images;
- 3,321 gallery images;
- 49 retained pathologist-curated positives;
- three to six positives per query.

The image manifest records source, relative path, UUID or curated filename, caption,
role, format, dimensions, cryptographic hash, perceptual hash, validation status, and
duplicate decision. The relevance manifest maps each query image ID to its retained
curated gallery image IDs.

## Biological-match definition

A correct match is a gallery image retained in the supplied pathologist-curated set
for that query after documented anomaly adjudication. Relevance is binary. Filename
rank suffixes are provenance rather than graded relevance. Missing ranks are not
invented, visually duplicate alternate encodings are represented once, visually
distinct repeated labels remain separate, and the observed sixth match remains a
positive.

Unjudged gallery images are operational negatives for scoring but are not evidence of
biological dissimilarity. ARCH captions are preserved as source text and are not
converted into tissue, diagnosis, stain, morphology, or magnification labels.

## Split and leakage boundary

Query and gallery roles are disjoint. Exact-content duplicates receive one
deterministic canonical gallery record; a query would take priority over its gallery
duplicate. The available source does not contain patient, slide, or specimen IDs, so
leakage at those levels cannot be tested.

See `configs/ground_truth.yaml` for adjudication decisions and
`configs/annotations.yaml` for the optional, currently unpopulated explanatory
annotation schema.
