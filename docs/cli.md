# Command-line workflow

Run commands from the repository root after activating `.venv`. The installed
research command is always `cipher`.

## Data and benchmark validation

```bash
cipher data prepare
cipher data inspect
cipher ground-truth validate
```

## Model and embedding workflow

```bash
cipher models inspect --model uni
cipher models access --model uni

for model in clip dino plip uni; do
  cipher embeddings generate --model "$model" --role all --device auto
done

cipher embeddings assemble --models clip,dino,plip,uni --role all --device auto
```

UNI requires separately approved upstream access and local Hugging Face
authentication. Credentials are never accepted as CIPHER command options.

## Retrieval and evaluation

Run one exact retrieval contract:

```bash
EMBEDDING_SET_ID="<set-id printed by cipher embeddings assemble>"
cipher retrieval run \
  --model uni \
  --metric cosine \
  --embedding-set "$EMBEDDING_SET_ID"
```

Valid metrics are `cosine`, `euclidean_l2`, `inner_product`, and `euclidean`.

Run every model and metric in `configs/retrieval/base.yaml`:

```bash
cipher evaluate benchmark --embedding-set "$EMBEDDING_SET_ID"
```

Run uncertainty, agreement, norm, and query diagnostics, then generate figures:

```bash
BENCHMARK_ID="<benchmark-id printed by cipher evaluate benchmark>"
cipher analysis run --benchmark-id "$BENCHMARK_ID"

ANALYSIS_ID="<analysis-id printed by cipher analysis run>"
cipher make-figures --analysis-id "$ANALYSIS_ID" \
  --benchmark-id "$BENCHMARK_ID"
cipher build-paper --analysis-id "$ANALYSIS_ID"
```

Generate a controlled preprocessing embedding set by adding either
`--preprocessing-variant grayscale` or
`--preprocessing-variant center_square` to both `embeddings generate` and
`embeddings assemble`. Compare completed benchmark IDs with:

```bash
BASELINE_BENCHMARK_ID="<model-native benchmark ID>"
GRAYSCALE_BENCHMARK_ID="<grayscale benchmark ID>"
CENTER_SQUARE_BENCHMARK_ID="<center-square benchmark ID>"
cipher analysis compare-preprocessing \
  --baseline-benchmark "$BASELINE_BENCHMARK_ID" \
  --grayscale-benchmark "$GRAYSCALE_BENCHMARK_ID" \
  --center-square-benchmark "$CENTER_SQUARE_BENCHMARK_ID"
```

The assembly, benchmark, and analysis commands print their generated IDs as JSON.
Assign each printed value to the corresponding shell variable before running the
next command. IDs are content-addressed and therefore differ when code, configuration,
data, or execution settings change.

The evaluator writes checksum-verified, content-addressed outputs beneath
`artifacts/benchmarks/<benchmark-id>/`. Generated data and artifacts are ignored by
Git. Use `--force` only when intentionally regenerating an otherwise valid output.

Every command supports `--help`; for example:

```bash
cipher evaluate benchmark --help
```
