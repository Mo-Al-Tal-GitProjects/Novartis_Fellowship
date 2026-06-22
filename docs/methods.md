# Retrieval and evaluation methods

This document defines CIPHER's benchmark contract. It is an implementation
specification as well as a methodological record: changing any rule below creates a
different experiment rather than silently altering an existing result.

## Benchmark population

The benchmark contains 11 ARCH PubMed query images and a gallery of 3,321 images.
The gallery combines the deduplicated ARCH collection with 49 adjudicated,
pathologist-curated positive images. Query and gallery roles are disjoint, and the
retrieval implementation also excludes any gallery row whose image ID equals the
query image ID.

Relevance is binary. A gallery image is relevant to a query only when it is one of
that query's retained curated matches in `data/manifests/relevance.parquet`. The
source filenames' `_01`, `_02`, and similar suffixes are provenance labels, not
graded relevance levels. Unjudged gallery images are treated as operational
negatives for scoring; this does not assert that they are biologically unrelated.

## Exact retrieval

CIPHER uses exhaustive NumPy ranking rather than an approximate index. For each
query, all 3,321 gallery vectors are scored and retained. This is fast at the current
benchmark size, avoids approximation as a confounder, and permits full-rank metrics
and error analysis.

The configured contracts are:

| Name | Stored vectors | Score | Ranking direction |
|---|---|---|---|
| `cosine` | row-wise L2-normalized | dot product | descending |
| `euclidean_l2` | row-wise L2-normalized | Euclidean distance | ascending |
| `inner_product` | raw | dot product | descending |
| `euclidean` | raw | Euclidean distance | ascending |

Cosine and Euclidean distance on unit vectors must induce the same ranking because
`||x-y||² = 2 - 2(x·y)`. CIPHER derives `euclidean_l2` from this identity so the two
rankings remain exactly equivalent despite floating-point evaluation order. Both are
retained as an explicit implementation check.
Raw inner product and raw Euclidean distance can differ because embedding norms are
preserved and may affect ranking.

Every ranking is deterministic. Exact score ties are resolved by ascending gallery
`image_id`, and ranks are contiguous from 1 through 3,321. The full configuration is
stored in `configs/retrieval/base.yaml`.

## Metrics

Metrics are computed independently for each query and then macro-averaged so every
query has equal weight despite having between three and six curated positives.
Configured cutoffs are 1, 3, 5, 10, 20, and 50.

- `hits@k`: number of curated positives among the first `k` results.
- top-k accuracy: proportion of queries with at least one positive among the first
  `k` results.
- precision@k: `hits@k / k`.
- recall@k: `hits@k / number of curated positives for the query`.
- AP@k: sum of precision at relevant ranks up to `k`, divided by the query's total
  number of curated positives. A missed positive therefore remains penalized.
- nDCG@k: binary discounted cumulative gain normalized by an ideal ordering with all
  available positives first.
- average precision: AP over the complete gallery; MAP is its macro mean.
- reciprocal rank: inverse rank of the first positive; MRR is its macro mean.
- mean and median first-relevant rank: descriptive location of the first positive.

The benchmark does not currently calculate graded relevance, patient-level metrics,
or label-stratified performance because those labels are not provided by the source
metadata. Explanatory annotations, when added, will remain separate from the curated
relevance judgments.

## Provenance and integrity

Retrieval run identity includes the immutable embedding-set identity and manifest,
the relevance-manifest checksum, pinned model revision, metric contract, role and
self-exclusion policy, gallery tie-breaker, CIPHER version, and source-tree hash.
Rankings are written to content-addressed directories with a checksum-bearing run
record. The benchmark similarly records its retrieval configuration, all 16 run IDs,
output checksums, and runtime provenance.

Run the baseline with:

```bash
EMBEDDING_SET_ID="<set-id printed by cipher embeddings assemble>"
cipher retrieval run --model uni --metric cosine \
  --embedding-set "$EMBEDDING_SET_ID"

cipher evaluate benchmark --embedding-set "$EMBEDDING_SET_ID"
```

The explicit set ID prevents later source changes from being mistaken for a request
to regenerate already verified embeddings.

## Diagnostic and ablation analyses

CIPHER resamples the 11 query rows 10,000 times with seed 42 to estimate descriptive
MAP intervals and paired model differences. It also measures top-k neighbor-set
Jaccard agreement, per-query difficulty, raw embedding norms, rank–norm Spearman
correlations, and per-query AP changes across metric contracts.

The preprocessing analysis applies two transformations before each model's native processor. `grayscale`
converts RGB to luminance and replicates the result into three channels. It probes
color dependence but is not stain normalization. `center_square` crops the image to a
centered square using its shorter dimension. It probes framing and field-of-view
sensitivity but does not isolate magnification. Both variants regenerate and align
all four full-dataset embedding sets before exact retrieval.
