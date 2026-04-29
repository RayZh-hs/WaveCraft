# Result Soundness Review

## Verdict

The corrected result is methodologically much sounder than the first pass: the two largest defects, ordinal category distances and invalid directional augmentation, have been addressed.
It is now reasonable to treat the exported medoids as a Phase II candidate tile library for qualitative inspection and early adjacency experiments.
It is still not strong enough to treat as a final kitbash library without manual/visual review, because cluster separation is modest and held-out diagnostics show two structures are poorly represented by the others.

## What Looks Sound

- The schematic conversion is reproducible and Phase I produced 32 arrays from 8 source schematics.
- The output structural palette has 384 categories including air, and ornaments are masked out of the training arrays.
- Direction-aware augmentation is enabled: True.
- Feature encoding is `onehot_per_voxel_truncated_svd_plus_height`, so arbitrary palette IDs are no longer treated as ordinal distances.
- Source-balanced sampling is enabled: True.
- Phase II found 310460 eligible 5x5x5 patches and clustered a sampled 60000 patches.
- The selected prototypes are medoids, so every exported tile is an actual observed patch rather than an averaged block soup.
- The transform sample counts are balanced: {'original': 15000, 'mirror_x': 15000, 'mirror_z': 15000, 'rot90_y': 15000}.

## Remaining Soundness Risks

1. Cluster separation is still modest.
   The best tested silhouette is 0.2078 at k=40. This is usable for exploration, but not strong evidence that the learned tiles form crisp architectural parts.

2. Some sources remain structurally distinct.
   Held-out source distances above 1.5x indicate that a source contains patch patterns not well represented by the rest of the corpus.

3. Some prototypes are still homogeneous material fields.
   These may be valid floor/wall/foundation pieces, but they should be manually reviewed so they do not crowd out more informative transition pieces.

4. The held-out evaluation is based on the shared SVD embedding.
   It is useful as a diagnostic, but a stricter validation would fit the dimensionality reduction on training sources only for each hold-out fold.

## Diagnostics

- Prototype tensor shape: `(40, 5, 5, 5)`.
- Feature center shape: `(40, 65)`.
- Feature info: `{'encoding': 'onehot_per_voxel_truncated_svd_plus_height', 'onehot_shape': [60000, 48000], 'svd_components': 64, 'svd_explained_variance_ratio_sum': 0.5532481670379639}`.
- Largest cluster sizes: `[(4, 4728), (2, 3591), (22, 3197), (7, 2553), (18, 2302), (16, 2208), (12, 2091), (24, 1893), (34, 1831), (9, 1778)]`.
- Prototype air fractions: `[0.64, 0.632, 0.0, 0.456, 0.0, 0.68, 0.608, 0.616, 0.6, 0.672, 0.672, 0.456, 0.632, 0.6, 0.664, 0.52, 0.64, 0.672, 0.672, 0.456, 0.584, 0.656, 0.672, 0.56, 0.568, 0.568, 0.16, 0.376, 0.608, 0.544, 0.56, 0.48, 0.648, 0.4, 0.648, 0.584, 0.568, 0.64, 0.568, 0.4]`.
- Prototype non-air voxel counts: `[45, 46, 125, 68, 125, 40, 49, 48, 50, 41, 41, 68, 46, 50, 42, 60, 45, 41, 41, 68, 52, 43, 41, 55, 54, 54, 105, 78, 49, 57, 55, 65, 44, 75, 44, 52, 54, 45, 54, 75]`.
- Source sample counts: `{'medieval_1a': 7664, 'medieval_2a': 7664, 'medieval_2b': 7664, 'medieval_2c': 7664, 'medieval_2d': 7664, 'medieval_2e': 7664, 'medieval_2f': 6352, 'medieval_3a': 7664}`.

Held-out source distance ratios:
- medieval_1a: 1.159
- medieval_2a: 1.153
- medieval_2b: 1.242
- medieval_2c: 1.227
- medieval_2d: 2.200
- medieval_2e: 2.467
- medieval_2f: 1.154
- medieval_3a: 1.225

Held-out sources needing inspection:
- medieval_2d: 2.200x train distance
- medieval_2e: 2.467x train distance

Clusters with at least 80% of samples from a single source:
- cluster 2: size 3591, medieval_2d share 100.0%
- cluster 4: size 4728, medieval_2e share 100.0%
- cluster 8: size 302, medieval_2d share 100.0%
- cluster 13: size 391, medieval_2e share 100.0%
- cluster 26: size 656, medieval_2e share 100.0%
- cluster 33: size 270, medieval_2d share 100.0%
- cluster 39: size 629, medieval_2d share 100.0%

Solid single-material prototype tiles:
- tile 2: 125/125 voxels of `stone:slab[type=bottom]`
- tile 4: 125/125 voxels of `wood_medium:log[axis=y]`

## Recommended Next Steps Before Phase III

1. Inspect `datasets/phase2/previews/prototypes.svg` and prune or merge homogeneous duplicate tiles.
2. Run a stricter leave-one-source-out validation where SVD is fit only on training sources for each fold.
3. Consider increasing the tested k range around 35-60 and adding a duplicate-patch cap so transition pieces are not underrepresented.
4. Use the current candidate tiles for a small adjacency-mining dry run, but keep the rule set marked experimental until visual and held-out checks pass.

Bottom line: the corrected result is sound enough for Phase II exploration and limited Phase III prototyping, but not yet a final WFC-ready tile catalog.
