# Reproducibility

CIPHER treats embeddings as derived research artifacts, not source data. Every run is content-addressed and excluded from Git. An artifact is reusable only when its recorded files pass checksum, shape, and dtype validation and its identity exactly matches the current inputs and execution contract.

## Validated environment

CIPHER 0.1.0 was validated with CPython 3.13.9 on macOS arm64. Embeddings used an Apple M2 GPU through MPS; exact retrieval, scoring, diagnostics, and figure generation used CPU execution. Exact Python package versions are recorded in `requirements.lock`. CPU and CUDA are supported embedding choices, but outputs generated on another device receive a different identity because floating-point results can differ by backend.

## Embedding identity

An embedding artifact identity includes:

- the CIPHER version and Python source-tree hash;
- the image-manifest checksum;
- the complete pinned model configuration and immutable upstream revision;
- the selected role, optional limit, and ordered image IDs;
- the execution device and batch size.

Changing any of these fields creates a new artifact directory rather than overwriting or silently reusing an incompatible result.

## Artifact layout

Each model run is stored beneath `artifacts/embeddings/<model>/<artifact-id>/`:

```text
embeddings.raw.npy   Unnormalized float32 representations
embeddings.l2.npy    Row-wise L2-normalized float32 representations
rows.parquet         Ordered image and embedding-index mapping
run.json             Identity, provenance, shape, and file checksums
```

Raw vectors are retained for norm and embedding-structure analyses. Normalized vectors support cosine-equivalent inner-product retrieval. Pickle is disabled when NumPy arrays are loaded.

## Full aligned model set

The validated full run used the active `all` selection and produced:

| Model | Rows | Dimension | Output |
|---|---:|---:|---|
| CLIP | 3,332 | 512 | vision projection |
| DINO | 3,332 | 768 | CLS token |
| PLIP | 3,332 | 512 | vision projection |
| UNI | 3,332 | 1,024 | CLS token |

The active selection contains 11 query records and 3,321 gallery records. Excluded duplicate rows are not embedded.

The command below validates every referenced file, confirms finite raw and normalized values, verifies unit L2 norms, and compares the ordered row metadata across models:

```bash
cipher embeddings assemble --models clip,dino,plip,uni --role all
```

The resulting `artifacts/embedding_sets/<set-id>/manifest.json` records the four artifact IDs, model revisions, dimensions, checksums, row counts, role counts, and a digest of the ordered image IDs. It copies no embeddings and grants no rights to the underlying images or model weights.

## Gated UNI weights

Each researcher must separately accept the UNI terms. CIPHER reads credentials only through standard Hugging Face authentication and checks the immutable local cache before making an authenticated request. Tokens are never included in model configs, run records, embedding-set manifests, or Git. Once the pinned weights are cached, offline generation works without authentication.

## Reproduction commands

```bash
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps

cipher data prepare
cipher ground-truth validate

for model in clip dino plip uni; do
  cipher embeddings generate --model "$model" --role all --device auto
done

cipher embeddings assemble --models clip,dino,plip,uni --role all --device auto
```

Generated artifacts and locally cached model weights must not be committed or redistributed. Successful embedding generation establishes reproducible inputs for retrieval analysis; it does not establish retrieval quality or biological validity.

## Exact retrieval and benchmark artifacts

Retrieval consumes an explicit immutable embedding-set manifest rather than deriving a
new embedding request from later source code. Retrieval identity still records the
current retrieval source, relevance checksum, model revision, metric mathematics,
self-exclusion rule, and tie-breaker.

Each run stores a complete `rankings.parquet` and checksum-bearing `run.json` beneath
`artifacts/retrieval/<set-id>/<model>/<metric>/<run-id>/`. The benchmark stores
per-query metrics, an aggregate table, a JSON summary, the 16 retrieval run IDs, and
checksums beneath `artifacts/benchmarks/<benchmark-id>/`.

```bash
cipher embeddings assemble --models clip,dino,plip,uni --role all
EMBEDDING_SET_ID="<set-id printed above>"
cipher evaluate benchmark --embedding-set "$EMBEDDING_SET_ID"
```

Re-running the command returns a cache hit only after validating the benchmark
identity and output checksums. See [methods.md](methods.md) for metric definitions.

## Diagnostic and ablation records

The analysis command records 10,000 paired query-bootstrap resamples, cross-model
agreement, norm summaries and correlations, metric sensitivity, and query
diagnostics. Grayscale and center-square variants each receive independent embedding
sets and benchmarks. The preprocessing comparison records query-level changes and
the exact input benchmark-manifest hashes.

Reviewed aggregate values are distributed as CSV files in `paper/results/`, and
reviewed quantitative figures are distributed in `paper/figures/`. Full rankings,
embeddings, locally generated reports, licensed images, and model weights remain
excluded from Git.
