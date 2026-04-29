# Phase II Evaluation

Extracted 5x5x5 overlapping patches from the Phase I augmented structural arrays, excluding windows with more than 70% air.
Density candidates found: 310460. Eligible architectural patches after filtering: 169004. Sampled patches clustered: 60000.
Rejected dense material chunks: 128456. Rejected low-salience chunks: 0. Rejected low-information single-kind chunks: 13000.
Sampling is source-balanced: True. Actual source sample counts: {'medieval_1a': 7665, 'medieval_2a': 7665, 'medieval_2b': 7665, 'medieval_2c': 7665, 'medieval_2d': 7664, 'medieval_2e': 7664, 'medieval_2f': 6348, 'medieval_3a': 7664}.
Sampling is semantic-bucket-balanced: True. Actual bucket sample counts: {'field': 3557, 'frame': 2032, 'opening': 19127, 'roof': 12785, 'wall': 2248, 'window': 20251}.
Feature encoding: sparse one-hot per voxel -> 64 SVD components plus architectural descriptors and relative height; explained variance ratio sum 0.4460.

## K Selection

| k | inertia | silhouette |
|---:|---:|---:|
| 70 | 1084890.38 | 0.1170 |

Chosen k: 70. Prototype tiles retained after pruning clusters smaller than 3: 65.
Socket-useful simple field prototypes retained by kind: {'log': 2, 'slab': 2}. Field prototypes pruned by cap: 5.

## Held-Out Source Diagnostics

| Held-out source | Train patches | Held-out patches | Held-out/train distance ratio |
|---|---:|---:|---:|
| medieval_1a | 52335 | 7665 | 1.051 |
| medieval_2a | 52335 | 7665 | 1.058 |
| medieval_2b | 52335 | 7665 | 1.095 |
| medieval_2c | 52335 | 7665 | 1.096 |
| medieval_2d | 52336 | 7664 | 1.377 |
| medieval_2e | 52336 | 7664 | 1.419 |
| medieval_2f | 53652 | 6348 | 1.037 |
| medieval_3a | 52336 | 7664 | 1.115 |

## Observations

- The palette reduction is doing important denoising work: dark masonry variants are visually inconsistent at the block level but structurally equivalent for patch learning, so they are grouped before clustering.
- One-hot/SVD features avoid treating arbitrary palette IDs as ordinal distances, while architectural descriptor features keep windows, openings, roof pieces, frames, and walls visible to clustering.
- Source-balanced patch sampling prevents the largest structure from dominating the training sample. Semantic-bucket balancing prevents common solid material chunks from crowding out rare building features while keeping socket-useful simple field tiles.
- The 70% air filter keeps roof edges, wall faces, and openings while dropping mostly empty context patches. Dense material chunks, low-salience local fragments, and simple single-kind sheets/columns without useful sockets are filtered before clustering.
- A sampled patch cap is used for tractable clustering. The script still counts all eligible windows before balanced sampling, so future runs can raise `--max-patches` when more memory or time is available.
- The selected medoids are actual observed patches, not averaged centroids, so every exported prototype is a valid block arrangement from the source corpus.

Preview text files and `prototypes.svg` are written to `datasets/phase2/previews`; use `legend.txt` there to decode compact category labels.
