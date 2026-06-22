# Model definitions

CIPHER compares frozen image representations rather than fine-tuning models. Every supported checkpoint is pinned to an immutable upstream revision, loaded in evaluation mode with remote code disabled, and run in float32. The baseline uses each checkpoint's stored image-processor configuration. Grayscale and center-square-crop alternatives are explicit ablations and never silently replace the baseline.

## Supported models

| CIPHER name | Upstream checkpoint | Revision | Image representation | Dimension |
|---|---|---|---|---:|
| `clip` | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | projected vision embedding | 512 |
| `dino` | `facebook/dino-vitb16` | `f205d5d8e640a89a2b8ef0369670dfc37cc07fc2` | final-layer CLS token | 768 |
| `plip` | `vinid/plip` | `67ade53ddd32195868f422585f72698ef5d15094` | projected vision embedding | 512 |
| `uni` | `MahmoodLab/UNI` | `b55a5ec6cade1a39edfe6534189a9b8ca7a022f0` | final-layer CLS token | 1024 |

`dino` means the original self-supervised DINO ViT-B/16 model used by the 2024 exploration, not DINOv2 or DINOv3. The CLS token is used directly; the optional Hugging Face pooler is disabled because it is not the original retrieval feature.

PLIP is CLIP-shaped but uses pathology-adapted weights from `vinid/plip`. It shares CIPHER's CLIP-compatible vision adapter while retaining an independent model identity, revision, preprocessing configuration, cache namespace, and provenance record.

`uni` means the original gated UNI ViT-L/16 model, not UNI2-h. CIPHER constructs the documented `vit_large_patch16_224` architecture with LayerScale initialization, loads the pinned state dictionary strictly, and extracts the CLS token directly. Its baseline preprocessing resizes the shorter image side to 224, center-crops to 224×224 for rectangular ARCH figures, converts to a tensor, and applies ImageNet mean and standard deviation. This matches the current upstream encoder helper's evaluation default while making the rectangular-image decision explicit for later ablation.

## Gated access

UNI access is individual and must be granted by the provider. CIPHER does not accept tokens through project configuration or CLI arguments. Authenticate through the standard local Hugging Face store:

```bash
hf auth login
cipher models access --model uni
```

The access command reports whether a credential exists, whether it came from the ephemeral `HF_TOKEN` environment variable or local credential store, whether authentication succeeds, whether the pinned gated configuration is readable, and whether the exact configuration and weights are already cached. It does not display the credential or Hugging Face account identity, and it exits nonzero unless the model is accessible. Model weights remain in the upstream cache and are excluded from Git. After a successful download, a researcher may run `hf auth logout`; CIPHER checks the immutable local cache before requesting authentication again.

## Embedding contract

For every model, CIPHER:

1. selects active manifest records in deterministic `image_id` order;
2. loads RGB images in batches;
3. applies the pinned checkpoint's native image processor;
4. extracts unnormalized float32 image features;
5. validates shape and finiteness;
6. stores both raw and row-wise L2-normalized arrays;
7. stores the ordered image rows, file checksums, model revision, device, batch size, manifest hash, and CIPHER source hash.

Raw vectors are preserved because their norms may contain model-specific structure worth analyzing. Normalized vectors support cosine similarity and equivalent inner-product retrieval without destroying the raw representation.

## Current scientific boundary

All four model adapters were validated against their intended immutable checkpoints for the 3,332 active query and gallery records. CIPHER verified row alignment for the model-native, grayscale, and center-square sets. These embeddings feed exact retrieval under four explicit metric contracts. The resulting 11-query measurements and ablations are descriptive and do not establish which model is generally best or which biological property drives retrieval.

Model weights are downloaded from their upstream hosts and excluded from Git. CIPHER's MIT License does not apply to those weights; see [the third-party license summary](../THIRD_PARTY_LICENSES.md).
