# Baseline results and interpretation

## Scope

These are new CIPHER reconstruction results, not results of the Fall 2024 prototype.
They describe retrieval against 49 binary, pathologist-curated judgments for 11
queries and a 3,321-image gallery. They are a small-benchmark baseline, not evidence
of general clinical or biological validity.

The validated baseline contains 16 model–metric combinations. Every full ranking
contains all 11 queries, every gallery item, and all 49 positives. Reviewed numeric
results are distributed in `paper/results/`.

## Primary results

| Metric | Model | MAP | MRR | P@5 | R@5 | Top-5 accuracy | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| cosine | UNI | 0.1240 | 0.3154 | 0.1636 | 0.1742 | 0.5455 | 0.1689 |
| cosine | DINO | 0.1218 | 0.2589 | 0.1273 | 0.1242 | 0.3636 | 0.1391 |
| cosine | PLIP | 0.0705 | 0.1143 | 0.0909 | 0.1000 | 0.3636 | 0.0739 |
| cosine | CLIP | 0.0356 | 0.1144 | 0.0182 | 0.0182 | 0.0909 | 0.0308 |
| raw Euclidean | DINO | 0.1251 | 0.2570 | 0.1273 | 0.1242 | 0.3636 | 0.1412 |
| raw Euclidean | UNI | 0.1242 | 0.3171 | 0.1636 | 0.1742 | 0.5455 | 0.1689 |
| raw Euclidean | PLIP | 0.0738 | 0.1159 | 0.0909 | 0.1000 | 0.3636 | 0.0739 |
| raw Euclidean | CLIP | 0.0339 | 0.1054 | 0.0182 | 0.0182 | 0.0909 | 0.0308 |
| raw inner product | UNI | 0.1183 | 0.3000 | 0.1455 | 0.1561 | 0.5455 | 0.1530 |
| raw inner product | DINO | 0.1144 | 0.2139 | 0.1455 | 0.1394 | 0.3636 | 0.1396 |
| raw inner product | PLIP | 0.0513 | 0.1027 | 0.0727 | 0.0848 | 0.2727 | 0.0647 |
| raw inner product | CLIP | 0.0251 | 0.1005 | 0.0182 | 0.0182 | 0.0909 | 0.0308 |

`euclidean_l2` produced exactly the same rankings and scores as cosine for every
model and query, as required by the unit-vector identity. It is omitted from the
table to avoid duplicating the first four rows.

## What the baseline supports

On these 11 queries, UNI has the highest MAP under cosine/normalized retrieval and
raw inner product. DINO has a narrowly higher MAP under raw Euclidean distance
(0.1251 versus 0.1242 for UNI), while UNI retains the higher MRR and top-5 measures.
PLIP is intermediate and CLIP is lowest on most reported measures.

Performance is strongly query-dependent. Under cosine, query 387 has much higher AP
than most queries for all four models, while queries 1024 and 2012 are difficult for
all four. UNI's median first-positive rank is 5, compared with 74 for DINO, 22 for
PLIP, and 396 for CLIP; the corresponding means are much larger, showing that a few
difficult queries create long-tailed failures.

The raw inner-product decrease relative to normalized retrieval for every model
suggests that vector magnitude changes rankings and is not uniformly helpful on this
benchmark. That is an observation about this embedding set, not yet an explanation
of what the norms encode.

## What the baseline does not support

The benchmark is too small for a definitive model hierarchy. The measurements do
not reveal whether a match was driven by tissue, morphology, stain, texture,
magnification, captions, or artifacts. They also cannot establish that an unjudged
result is a true biological false positive. The current data provide no patient or
specimen grouping with which to test leakage at those levels.

Model differences are descriptive until uncertainty, paired query-level comparisons,
retrieval-panel review, cross-model agreement, and explanatory annotations are added.
Paired analysis shows that UNI and DINO are not distinguishable on this
11-query cosine benchmark: the DINO-minus-UNI AP difference is −0.0023 with a
query-bootstrap interval of [−0.0603, 0.0646]. Top-10 neighbor overlap ranges from
0.087 to 0.168 across model pairs, so similar aggregate scores often arise from
different retrieved images.

Grayscale reduced PLIP and UNI MAP by 0.0307 and 0.0266, while center-square cropping
increased DINO MAP by 0.0147. Every paired change interval crossed zero. These results
indicate model-specific sensitivity but do not identify a causal stain, morphology,
or field-of-view mechanism. Full interpretation is recorded in
`paper/manuscript.md` and its tracked quantitative figures and result snapshots.
