# Third-party data, models, and licenses

The repository-level MIT License applies only to original CIPHER code and documentation. It does not relicense third-party assets. This file is an operational summary, not legal advice; upstream terms control.

## ARCH dataset

- Source: [University of Warwick Tissue Image Analytics Centre](https://warwick.ac.uk/fac/cross_fac/tia/data/arch/)
- License stated by the provider: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Use: non-commercial research, with the source paper cited as required by the provider.
- Repository policy: archives and extracted images are ignored and are not distributed under MIT.

## Curated match images

- Source: supplied during the original challenge as pathologist-curated retrieval judgments.
- License/redistribution status: not established in the available records.
- Repository policy: local research use only; images are ignored and must not be published or redistributed without permission. Generated manifests may record local filenames and checksums, but do not grant rights to the images.

## CLIP

- Upstream software: [OpenAI CLIP](https://github.com/openai/CLIP)
- Upstream software license: MIT.
- Model: `openai/clip-vit-base-patch32`.
- Pinned model revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- Inference path: Hugging Face `CLIPVisionModelWithProjection` and the checkpoint's native `CLIPImageProcessor` configuration.
- Repository policy: download through the upstream provider; do not vendor weights. Review the model card before use.

## DINO

- Upstream software/model: [Facebook Research DINO](https://github.com/facebookresearch/dino)
- Upstream license: Apache License 2.0.
- Model: `facebook/dino-vitb16`, corresponding to the original DINO ViT-B/16 backbone rather than DINOv2 or DINOv3.
- Pinned model revision: `f205d5d8e640a89a2b8ef0369670dfc37cc07fc2`.
- Inference path: Hugging Face `ViTModel`, the checkpoint's native `ViTImageProcessor`, and the final-layer CLS token.
- Repository policy: download through the upstream provider; preserve required notices; do not vendor weights.

## PLIP

- Upstream software/model: [PathologyFoundation PLIP](https://github.com/PathologyFoundation/plip) and `vinid/plip`.
- Pinned model revision: `67ade53ddd32195868f422585f72698ef5d15094`.
- Inference path: Hugging Face `CLIPVisionModelWithProjection` and the checkpoint's native `CLIPImageProcessor` configuration.
- License status observed during reconstruction: the upstream Python package metadata labels the software MIT, but the GitHub repository does not contain a license file and the Hugging Face model card does not declare a machine-readable license for the weights.
- Repository policy: research use only unless upstream terms are clarified; do not redistribute weights or derived model packages. Recheck the upstream license before any public release.

## UNI

- Upstream model: [`MahmoodLab/UNI`](https://huggingface.co/MahmoodLab/UNI)
- Access: gated and subject to provider approval/terms.
- Model: original UNI ViT-L/16, not UNI2-h.
- Pinned model revision: `b55a5ec6cade1a39edfe6534189a9b8ca7a022f0`.
- Inference path: `timm` ViT-L/16 with strict state-dictionary loading and the final-layer CLS token.
- Model-card license and terms: CC BY-NC-ND 4.0; non-commercial academic research only, with attribution and the additional provider conditions accepted during gated access.
- Repository policy: each user must obtain access directly; credentials and weights must never enter Git; CIPHER must not redistribute or modify the weights.

## Python dependencies

Python dependencies retain their own licenses. `requirements.lock` records the
validated environment for reproducibility; it does not relicense those packages. No
third-party source code is vendored by CIPHER.
