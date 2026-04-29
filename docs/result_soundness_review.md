# Result Soundness Review

## Verdict

The corrected result is better aligned with building generation than the previous density-only extraction.
Phase II now filters dense material chunks, balances the sample across architectural buckets, and appends semantic descriptors so windows, openings, roof pieces, frames, and walls can influence clustering instead of being treated as rare voxel noise.
It is reasonable to treat the exported medoids as a Phase II candidate tile library for qualitative inspection and early adjacency experiments.
It is still not a final kitbash library without manual/visual review, because the extractor still learns local 5x5x5 neighborhoods rather than explicitly labeled complete building modules.

## What Looks Sound

- The schematic conversion is reproducible and Phase I produced 32 arrays from 8 source schematics.
- The output structural palette has 385 categories including air, and ornaments are masked out of the training arrays.
- Direction-aware augmentation is enabled: True.
- Feature encoding is `onehot_per_voxel_truncated_svd_plus_architectural_descriptors_plus_height`, so arbitrary palette IDs are no longer treated as ordinal distances and architectural descriptors now affect clustering.
- Source-balanced sampling is enabled: True.
- Semantic-bucket-balanced sampling is enabled: True.
- Phase II found 310460 density candidates, retained 169004 architectural 5x5x5 patches, and clustered a sampled 60000 patches.
- Dense material chunks rejected before clustering: 128456.
- Low-information single-kind sheets/columns rejected before clustering: 13000.
- Socket-useful simple field prototypes are capped by kind: {'log': 2, 'slab': 2}; pruned by cap: 5.
- The selected prototypes are medoids, so every exported tile is an actual observed patch rather than an averaged block soup.
- The transform sample counts are balanced: {'original': 14999, 'mirror_x': 14997, 'mirror_z': 15002, 'rot90_y': 15002}.
- Semantic bucket sample counts are balanced where available: {'frame': 2032, 'opening': 19127, 'roof': 12785, 'wall': 2248, 'window': 20251, 'field': 3557}.

## Remaining Soundness Risks

1. Cluster separation is still modest.
   The best tested silhouette is 0.1170 at k=70. This is usable for exploration, but not strong evidence that the learned tiles form crisp architectural parts.

2. Some sources remain structurally distinct.
   Held-out source distances above 1.25x indicate that a source contains patch patterns not well represented by the rest of the corpus.

3. Some prototypes may still be local cut-throughs rather than complete semantic objects.
   The new buckets reduce random-looking chunks, but a 5x5x5 sliding window can still intersect only part of a window, roof ridge, or wall segment.

4. The held-out evaluation is based on the shared SVD embedding.
   It is useful as a diagnostic, but a stricter validation would fit the dimensionality reduction on training sources only for each hold-out fold.

## Diagnostics

- Prototype tensor shape: `(65, 5, 5, 5)`.
- Feature center shape: `(70, 74)`.
- Feature info: `{'encoding': 'onehot_per_voxel_truncated_svd_plus_architectural_descriptors_plus_height', 'onehot_shape': [60000, 48125], 'semantic_descriptor_columns': ['air_fraction', 'boundary_contact_fraction', 'air_nonair_interface_fraction', 'window_fraction', 'opening_fraction', 'roof_fraction', 'frame_fraction', 'wall_fraction', 'architectural_score'], 'semantic_feature_weight': 4.0, 'svd_components': 64, 'svd_explained_variance_ratio_sum': 0.44600555300712585}`.
- Largest cluster sizes: `[(44, 2140), (5, 1806), (28, 1805), (55, 1618), (33, 1614), (17, 1613), (48, 1501), (0, 1415), (42, 1398), (40, 1310)]`.
- Prototype air fractions: `[0.656, 0.408, 0.528, 0.64, 0.664, 0.608, 0.6, 0.688, 0.696, 0.4, 0.656, 0.528, 0.4, 0.392, 0.504, 0.56, 0.536, 0.608, 0.56, 0.648, 0.4, 0.664, 0.4, 0.616, 0.384, 0.656, 0.672, 0.648, 0.688, 0.464, 0.608, 0.584, 0.496, 0.504, 0.688, 0.64, 0.688, 0.68, 0.504, 0.584, 0.68, 0.648, 0.6, 0.664, 0.68, 0.608, 0.696, 0.544, 0.616, 0.544, 0.664, 0.672, 0.632, 0.656, 0.592, 0.632, 0.64, 0.464, 0.696, 0.568, 0.512, 0.544, 0.632, 0.616, 0.52]`.
- Prototype non-air voxel counts: `[43, 74, 59, 45, 42, 49, 50, 39, 38, 75, 43, 59, 75, 76, 62, 55, 58, 49, 55, 44, 75, 42, 75, 48, 77, 43, 41, 44, 39, 67, 49, 52, 63, 62, 39, 45, 39, 40, 62, 52, 40, 44, 50, 42, 40, 49, 38, 57, 48, 57, 42, 41, 46, 43, 51, 46, 45, 67, 38, 54, 61, 57, 46, 48, 60]`.
- Source sample counts: `{'medieval_1a': 7665, 'medieval_2a': 7665, 'medieval_2b': 7665, 'medieval_2c': 7665, 'medieval_2d': 7664, 'medieval_2e': 7664, 'medieval_2f': 6348, 'medieval_3a': 7664}`.
- Semantic bucket sample counts: `{'frame': 2032, 'opening': 19127, 'roof': 12785, 'wall': 2248, 'window': 20251, 'field': 3557}`.
- Candidate bucket counts: `{'field': 11948, 'frame': 2032, 'opening': 80656, 'roof': 17736, 'wall': 2248, 'window': 54384}`.
- Prototype pruning: `{'field_prototype_kind_counts': {'log': 2, 'slab': 2}, 'pruned_field_prototypes': 5}`.

Held-out source distance ratios:
- medieval_1a: 1.051
- medieval_2a: 1.058
- medieval_2b: 1.095
- medieval_2c: 1.096
- medieval_2d: 1.377
- medieval_2e: 1.419
- medieval_2f: 1.037
- medieval_3a: 1.115

Held-out sources needing inspection:
- medieval_2d: 1.377x train distance
- medieval_2e: 1.419x train distance

Clusters with at least 80% of samples from a single source:
- cluster 9: size 767, medieval_2e share 100.0%
- cluster 11: size 266, medieval_2e share 100.0%
- cluster 12: size 402, medieval_2e share 100.0%
- cluster 20: size 1197, medieval_2d share 100.0%
- cluster 22: size 295, medieval_2d share 100.0%
- cluster 23: size 328, medieval_2e share 100.0%
- cluster 25: size 321, medieval_2e share 100.0%
- cluster 30: size 198, medieval_2d share 100.0%
- cluster 35: size 221, medieval_2d share 100.0%
- cluster 41: size 740, medieval_3a share 80.5%
- cluster 56: size 80, medieval_2d share 100.0%

## Recommended Next Steps Before Phase III

1. Inspect `datasets/phase2/previews/prototypes.svg` and `datasets/phase2/inspection/index.html` to prune or merge local cut-throughs that still do not read as usable building parts.
2. Run a stricter leave-one-source-out validation where SVD is fit only on training sources for each fold.
3. Consider increasing the tested k range around 50-90 now that buckets preserve rarer feature classes.
4. Use the current candidate tiles for a small adjacency-mining dry run, but keep the rule set marked experimental until visual and held-out checks pass.

Bottom line: the corrected result should generate less material noise and more recognizable architectural fragments, but it still needs visual pruning before becoming a final WFC-ready tile catalog.
