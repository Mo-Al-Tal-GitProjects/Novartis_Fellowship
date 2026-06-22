# CIPHER

**Content-based Image Pathology: Histology Embedding Retrieval**

> A reproducible study of histopathology image embeddings for biological-match retrieval.

[![CI](https://github.com/MohammedAlTal/CIPHER/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammedAlTal/CIPHER/actions/workflows/ci.yml)

**CIPHER is an independent revisit of a past Fall 2024 AI in Pathology fellowship
project.** The original challenge explored pre-trained image models for
content-based retrieval in histopathology; this repository returns to that research
question as a standalone study with a new implementation, corrected benchmark
contract, reproducible experiments, and more cautious interpretation. It is not the
original fellowship deliverable or an official institutional release. The full
project history and non-endorsement statement are in [background.md](background.md).

CIPHER asks whether frozen image embeddings retrieve pathologist-curated biological
matches—not merely images with similar color, texture, or composition. It compares
CLIP, DINO, PLIP, and UNI while examining retrieval quality, metric sensitivity,
embedding norms, model agreement, preprocessing effects, and query-level failures.

## Study at a glance

| Component | Current study |
|---|---|
| Research question | What properties of pre-trained image embedding spaces are associated with correct pathologist-curated retrievals? |
| Data | ARCH PubMed images plus a separately supplied curated-match set |
| Benchmark | 11 queries, 3,321 gallery images, and 49 retained curated positives |
| Encoders | CLIP ViT-B/32, original DINO ViT-B/16, PLIP, and original UNI ViT-L/16 |
| Retrieval | Exact full-gallery cosine, normalized Euclidean, raw inner product, and raw Euclidean ranking |
| Evaluation | MAP, MRR, precision/recall/AP/nDCG at k, top-k accuracy, bootstrap intervals, agreement, and failure diagnostics |
| Ablations | Grayscale conversion and center-square cropping |
| Interface | Reproducible terminal workflow through the `cipher` command |

The benchmark is intentionally narrow. Its 11 queries support a controlled case
study of embedding behavior, not a general model leaderboard, clinical claim, or
causal account of biological similarity. Unjudged gallery images are operational
negatives for scoring; they are not confirmed biological mismatches.

## Current findings

For model-native preprocessing and cosine retrieval:

| Model | MAP | 95% query-bootstrap interval | MRR | Top-5 accuracy |
|---|---:|---:|---:|---:|
| UNI | 0.1240 | [0.0514, 0.2165] | 0.3154 | 0.5455 |
| DINO | 0.1218 | [0.0247, 0.2605] | 0.2589 | 0.3636 |
| PLIP | 0.0705 | [0.0226, 0.1234] | 0.1143 | 0.3636 |
| CLIP | 0.0356 | [0.0024, 0.0883] | 0.1144 | 0.0909 |

![Baseline retrieval performance](paper/figures/figure_1_baseline_map.png)

UNI and DINO form the strongest baseline group under cosine retrieval, but their MAP
difference is inconclusive on this query set. The models also retrieve substantially
different neighbors: mean pairwise top-10 Jaccard overlap ranges from 0.087 to 0.168.
Raw vector magnitude materially changes CLIP and PLIP inner-product rankings, and the
preprocessing ablations show model-specific sensitivity with wide uncertainty.

These are descriptive findings from the present benchmark. See
[results_interpretation.md](docs/results_interpretation.md) for the complete reading
and [unofficial_manuscript.md](paper/unofficial_manuscript.md) for the unofficial,
non-peer-reviewed study manuscript.

## How the workflow fits together

```text
ARCH + curated matches
        │
        ▼
validated manifests ──► adjudicated relevance judgments
        │
        ▼
pinned model adapters ──► aligned raw and normalized embeddings
        │
        ▼
exact full-gallery retrieval ──► query metrics and benchmark summaries
        │
        ▼
uncertainty, agreement, norm, and ablation analyses ──► figures and manuscript
```

Every generated embedding set, retrieval run, benchmark, and analysis receives a
content-derived identity. CIPHER validates row order, shape, dtype, checksums, model
revision, and relevant configuration before reusing cached work. Licensed images,
model weights, embeddings, rankings, and local reports are excluded from Git.

## Installation

CIPHER requires Python 3.11 or newer. The committed lock file records the exact
macOS arm64 environment used for the completed study.

```bash
git clone https://github.com/MohammedAlTal/CIPHER.git
cd CIPHER
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps

cipher --help
```

UNI is gated. Each researcher must independently accept the upstream terms and
authenticate through the local Hugging Face credential store:

```bash
hf auth login
cipher models access --model uni
```

Use a read-only token. Never place credentials in project files or command
arguments. CIPHER does not write Hugging Face credentials into artifacts.

## Reproduce the study

The default data command downloads the official ARCH PubMed archive and imports the
locally supplied curated-match folder from `../matches`. Both paths can be overridden
through the CLI.

```bash
# 1. Prepare and validate the benchmark.
cipher data prepare
cipher data inspect
cipher ground-truth validate

# 2. Generate one aligned full-dataset embedding artifact per model.
for model in clip dino plip uni; do
  cipher embeddings generate --model "$model" --role all --device auto
done
cipher embeddings assemble --models clip,dino,plip,uni --role all --device auto

# 3. Use the set ID printed by the assembly command.
EMBEDDING_SET_ID="<embedding-set-id>"
cipher evaluate benchmark --embedding-set "$EMBEDDING_SET_ID"

# 4. Use the benchmark ID printed by the evaluator.
BENCHMARK_ID="<benchmark-id>"
cipher analysis run --benchmark-id "$BENCHMARK_ID"

# 5. Use the analysis ID printed by the analysis command.
ANALYSIS_ID="<analysis-id>"
cipher make-figures \
  --analysis-id "$ANALYSIS_ID" \
  --benchmark-id "$BENCHMARK_ID"
cipher build-paper --analysis-id "$ANALYSIS_ID"
```

The IDs are content-addressed and may change when code, configuration, data, or
execution settings change. Controlled grayscale and center-square workflows are
documented in [cli.md](docs/cli.md).

## Repository guide

| Path | Purpose |
|---|---|
| [`src/cipher/`](src/cipher/) | Data preparation, model adapters, embeddings, retrieval, evaluation, analysis, and CLI |
| [`configs/`](configs/) | Model pins, preprocessing, ground-truth adjudication, and retrieval settings |
| [`tests/`](tests/) | Unit and integration tests using synthetic fixtures |
| [`docs/`](docs/) | Data, models, methods, CLI, reproducibility, interpretation, and limitations |
| [`paper/`](paper/) | Unofficial manuscript, reviewed quantitative figures, and result snapshots |
| [`data/`](data/) | Ignored local data areas and data-boundary documentation |
| [`artifacts/`](artifacts/) | Ignored content-addressed outputs created by study runs |

Start with:

- [project_overview.md](docs/project_overview.md) for the scientific scope;
- [data.md](docs/data.md) for the benchmark and biological-match definition;
- [models.md](docs/models.md) for checkpoints and feature contracts;
- [methods.md](docs/methods.md) for retrieval and evaluation mathematics;
- [reproducibility.md](docs/reproducibility.md) for artifact identity and validation;
- [limitations.md](docs/limitations.md) before extending or citing the results.

## Development checks

```bash
ruff check .
pytest
python -m build
```

Continuous integration runs linting, tests, and package builds on every push and
pull request to `main`.

## Provenance and research boundary

The Fall 2024 fellowship project supplied the motivating question and historical
setting. CIPHER's code, corrected protocol, experiments, figures, and interpretation
are a later independent reconstruction. This repository is not an official Novartis
or NIBR release, an official Break Through Tech deliverable, an approved publication,
or a clinical system. Historical attribution does not imply endorsement by advisors,
mentors, teammates, institutions, or partner organizations.

## License and external assets

Original CIPHER code and documentation are licensed under the
[MIT License](license). The repository license does **not** cover ARCH, the curated
match images, model weights, or third-party software. Their upstream licenses and
access conditions remain controlling; review [third_party_licenses.md](third_party_licenses.md)
before downloading, using, or redistributing any external asset.

If you use the software, cite CIPHER together with the datasets and model checkpoints
used in your run. Repository citation metadata are provided in [citation.cff](citation.cff).
