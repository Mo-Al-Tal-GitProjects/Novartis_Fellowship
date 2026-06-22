# Limitations

- Only 11 queries and 49 positive judgments are available.
- Relevance judgments are positive-only and not exhaustive; unjudged images may be
  valid biological matches.
- Relevance is binary, with no graded match quality.
- No patient, slide, specimen, or acquisition grouping is available for leakage
  analysis.
- ARCH captions are descriptive text, not validated structured biological labels.
- Query-bootstrap intervals measure sensitivity to the observed queries and do not
  establish external or clinical validity.
- Grayscale removes all hue and is not equivalent to stain normalization.
- Center-square cropping changes field of view and composition simultaneously.
- Model-native processors may change effective magnification differently.
- Retrieval panels have not received new pathologist adjudication.
- The optional explanatory annotation schema is intentionally unpopulated; no tissue,
  morphology, stain, disease, magnification, or artifact label is invented.
- CIPHER is research software, not a medical device or clinical decision system.

These constraints should travel with every result, figure, report, and manuscript
derived from the current benchmark.
