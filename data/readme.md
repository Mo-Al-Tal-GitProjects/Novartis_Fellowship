# Local data layout

All image data and generated manifests are excluded from Git.

```text
data/
├── external/          # downloaded source archives
├── raw/
│   ├── arch/          # extracted ARCH PubMed files
│   └── matches/       # locally imported curated images
├── interim/           # future derived image data
└── manifests/         # generated image/relevance Parquet files and reports
```

Prepare and validate the data with:

```bash
cipher data prepare
cipher data inspect
cipher ground-truth validate
```

ARCH and curated images are not licensed under the repository MIT License. See `third_party_licenses.md`.
