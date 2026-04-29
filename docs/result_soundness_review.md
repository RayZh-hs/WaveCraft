# Result Soundness Review

## Verdict

The corrected result is methodologically much sounder than the first pass: the two largest defects, ordinal category distances and invalid directional augmentation, have been addressed.
It is now reasonable to treat the exported medoids as a Phase II candidate tile library for qualitative inspection and early adjacency experiments.
It is still not strong enough to treat as a final kitbash library without manual/visual review, because cluster separation is modest and held-out diagnostics show two structures are poorly represented by the others.

## What Looks Sound

- The schematic conversion is reproducible and Phase I produced 32 arrays from 8 source schematics.
- The output structural palette has 385 categories including air, and ornaments are masked out of the training arrays.
- Direction-aware augmentation is enabled: True.
- Feature encoding is `onehot_per_voxel_truncated_svd_plus_height`, so arbitrary palette IDs are no longer treated as ordinal distances.
- Source-balanced sampling is enabled: True.
- Phase II found 310460 eligible 5x5x5 patches and clustered a sampled 60000 patches.
- The selected prototypes are medoids, so every exported tile is an actual observed patch rather than an averaged block soup.
- The transform sample counts are balanced: {'original': 15000, 'mirror_x': 15000, 'mirror_z': 15000, 'rot90_y': 15000}.

## Remaining Soundness Risks

1. Cluster separation is still modest.
   The best tested silhouette is 0.2053 at k=50. This is usable for exploration, but not strong evidence that the learned tiles form crisp architectural parts.

2. Some sources remain structurally distinct.
   Held-out source distances above 1.5x indicate that a source contains patch patterns not well represented by the rest of the corpus.

3. Some prototypes are still homogeneous material fields.
   These may be valid floor/wall/foundation pieces, but they should be manually reviewed so they do not crowd out more informative transition pieces.

4. The held-out evaluation is based on the shared SVD embedding.
   It is useful as a diagnostic, but a stricter validation would fit the dimensionality reduction on training sources only for each hold-out fold.

## Diagnostics

- Prototype tensor shape: `(50, 5, 5, 5)`.
- Feature center shape: `(50, 65)`.
- Feature info: `{'encoding': 'onehot_per_voxel_truncated_svd_plus_height', 'onehot_shape': [60000, 48125], 'svd_components': 64, 'svd_explained_variance_ratio_sum': 0.5531390905380249}`.
- Largest cluster sizes: `[(1, 5019), (2, 3584), (35, 1947), (3, 1884), (9, 1848), (20, 1811), (16, 1760), (8, 1734), (22, 1630), (12, 1619)]`.
- Prototype air fractions: `[0.664, 0.0, 0.0, 0.664, 0.624, 0.664, 0.544, 0.6, 0.672, 0.672, 0.568, 0.68, 0.672, 0.552, 0.56, 0.688, 0.608, 0.4, 0.2, 0.648, 0.672, 0.448, 0.568, 0.584, 0.448, 0.576, 0.592, 0.656, 0.664, 0.544, 0.64, 0.6, 0.64, 0.616, 0.68, 0.68, 0.68, 0.584, 0.536, 0.528, 0.6, 0.64, 0.4, 0.616, 0.2, 0.672, 0.648, 0.496, 0.552, 0.52]`.
- Prototype non-air voxel counts: `[42, 125, 125, 42, 47, 42, 57, 50, 41, 41, 54, 40, 41, 56, 55, 39, 49, 75, 100, 44, 41, 69, 54, 52, 69, 53, 51, 43, 42, 57, 45, 50, 45, 48, 40, 40, 40, 52, 58, 59, 50, 45, 75, 48, 100, 41, 44, 63, 56, 60]`.
- Source sample counts: `{'medieval_1a': 7664, 'medieval_2a': 7664, 'medieval_2b': 7664, 'medieval_2c': 7664, 'medieval_2d': 7664, 'medieval_2e': 7664, 'medieval_2f': 6352, 'medieval_3a': 7664}`.

Held-out source distance ratios:
- medieval_1a: 1.173
- medieval_2a: 1.176
- medieval_2b: 1.259
- medieval_2c: 1.243
- medieval_2d: 2.243
- medieval_2e: 2.558
- medieval_2f: 1.152
- medieval_3a: 1.254

Held-out sources needing inspection:
- medieval_2d: 2.243x train distance
- medieval_2e: 2.558x train distance

Clusters with at least 80% of samples from a single source:
- cluster 1: size 5019, medieval_2e share 100.0%
- cluster 2: size 3584, medieval_2d share 100.0%
- cluster 18: size 529, medieval_2d share 100.0%
- cluster 21: size 481, medieval_2e share 100.0%
- cluster 40: size 304, medieval_2d share 100.0%
- cluster 42: size 313, medieval_2d share 100.0%
- cluster 44: size 213, medieval_2e share 100.0%

Solid single-material prototype tiles:
- tile 1: 125/125 voxels of `wood_medium:log[axis=y,stripped=true]`
- tile 2: 125/125 voxels of `stone:slab[type=bottom]`

## Recommended Next Steps Before Phase III

1. Inspect `datasets/phase2/previews/prototypes.svg` and prune or merge homogeneous duplicate tiles.
2. Run a stricter leave-one-source-out validation where SVD is fit only on training sources for each fold.
3. Consider increasing the tested k range around 35-60 and adding a duplicate-patch cap so transition pieces are not underrepresented.
4. Use the current candidate tiles for a small adjacency-mining dry run, but keep the rule set marked experimental until visual and held-out checks pass.

Bottom line: the corrected result is sound enough for Phase II exploration and limited Phase III prototyping, but not yet a final WFC-ready tile catalog.
