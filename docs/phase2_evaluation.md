# Phase II Evaluation

Extracted 5x5x5 overlapping patches from the Phase I augmented structural arrays, excluding windows with more than 70% air.
Eligible patches found: 310460. Sampled patches clustered: 60000.
Sampling is source-balanced: True. Actual source sample counts: {'medieval_1a': 7664, 'medieval_2a': 7664, 'medieval_2b': 7664, 'medieval_2c': 7664, 'medieval_2d': 7664, 'medieval_2e': 7664, 'medieval_2f': 6352, 'medieval_3a': 7664}.
Feature encoding: sparse one-hot per voxel -> 64 SVD components plus relative height; explained variance ratio sum 0.5531.

## K Selection

| k | inertia | silhouette |
|---:|---:|---:|
| 30 | 1125344.62 | 0.2001 |
| 40 | 1074643.75 | 0.2009 |
| 50 | 1025259.00 | 0.2053 |

Chosen k: 50. Prototype tiles retained after pruning clusters smaller than 3: 50.

## Held-Out Source Diagnostics

| Held-out source | Train patches | Held-out patches | Held-out/train distance ratio |
|---|---:|---:|---:|
| medieval_1a | 52336 | 7664 | 1.173 |
| medieval_2a | 52336 | 7664 | 1.176 |
| medieval_2b | 52336 | 7664 | 1.259 |
| medieval_2c | 52336 | 7664 | 1.243 |
| medieval_2d | 52336 | 7664 | 2.243 |
| medieval_2e | 52336 | 7664 | 2.558 |
| medieval_2f | 53648 | 6352 | 1.152 |
| medieval_3a | 52336 | 7664 | 1.254 |

## Observations

- The palette reduction is doing important denoising work: dark masonry variants are visually inconsistent at the block level but structurally equivalent for patch learning, so they are grouped before clustering.
- One-hot/SVD features avoid treating arbitrary palette IDs as ordinal distances.
- Source-balanced patch sampling prevents the largest structure from dominating the training sample.
- The 70% air filter keeps roof edges, wall faces, and openings while dropping mostly empty context patches.
- A sampled patch cap is used for tractable clustering. The script still counts all eligible windows before balanced sampling, so future runs can raise `--max-patches` when more memory or time is available.
- The selected medoids are actual observed patches, not averaged centroids, so every exported prototype is a valid block arrangement from the source corpus.

Preview text files and `prototypes.svg` are written to `datasets/phase2/previews`; use `legend.txt` there to decode compact category labels.
