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
- Phase II found 310460 density candidates, retained 157736 architectural 5x5x5 patches, and clustered a sampled 60000 patches.
- Dense material chunks rejected before clustering: 128456.
- Low-information single-kind sheets/columns rejected before clustering: 24268.
- The selected prototypes are medoids, so every exported tile is an actual observed patch rather than an averaged block soup.
- The transform sample counts are balanced: {'original': 15001, 'mirror_x': 14998, 'mirror_z': 15001, 'rot90_y': 15000}.
- Semantic bucket sample counts are balanced where available: {'frame': 2128, 'opening': 20638, 'roof': 13181, 'wall': 2292, 'window': 21761}.

## Remaining Soundness Risks

1. Cluster separation is still modest.
   The best tested silhouette is 0.0769 at k=90. This is usable for exploration, but not strong evidence that the learned tiles form crisp architectural parts.

2. Some sources remain structurally distinct.
   Held-out source distances above 1.25x indicate that a source contains patch patterns not well represented by the rest of the corpus.

3. Some prototypes may still be local cut-throughs rather than complete semantic objects.
   The new buckets reduce random-looking chunks, but a 5x5x5 sliding window can still intersect only part of a window, roof ridge, or wall segment.

4. The held-out evaluation is based on the shared SVD embedding.
   It is useful as a diagnostic, but a stricter validation would fit the dimensionality reduction on training sources only for each hold-out fold.

## Diagnostics

- Prototype tensor shape: `(90, 5, 5, 5)`.
- Feature center shape: `(90, 74)`.
- Feature info: `{'encoding': 'onehot_per_voxel_truncated_svd_plus_architectural_descriptors_plus_height', 'onehot_shape': [60000, 48125], 'semantic_descriptor_columns': ['air_fraction', 'boundary_contact_fraction', 'air_nonair_interface_fraction', 'window_fraction', 'opening_fraction', 'roof_fraction', 'frame_fraction', 'wall_fraction', 'architectural_score'], 'semantic_feature_weight': 4.0, 'svd_components': 64, 'svd_explained_variance_ratio_sum': 0.41391098499298096}`.
- Largest cluster sizes: `[(59, 1479), (21, 1403), (58, 1235), (8, 1201), (30, 1119), (61, 1075), (48, 1072), (53, 1072), (25, 1056), (80, 1004)]`.
- Prototype air fractions: `[0.616, 0.624, 0.648, 0.648, 0.544, 0.616, 0.536, 0.6, 0.664, 0.672, 0.656, 0.632, 0.552, 0.688, 0.688, 0.624, 0.44, 0.536, 0.576, 0.64, 0.648, 0.696, 0.584, 0.576, 0.656, 0.664, 0.512, 0.624, 0.528, 0.696, 0.672, 0.536, 0.672, 0.52, 0.632, 0.392, 0.584, 0.672, 0.6, 0.4, 0.656, 0.504, 0.528, 0.6, 0.584, 0.68, 0.48, 0.672, 0.64, 0.688, 0.584, 0.52, 0.688, 0.648, 0.424, 0.616, 0.672, 0.616, 0.664, 0.648, 0.64, 0.648, 0.456, 0.56, 0.56, 0.496, 0.664, 0.6, 0.608, 0.68, 0.584, 0.648, 0.496, 0.584, 0.672, 0.616, 0.664, 0.616, 0.568, 0.576, 0.616, 0.496, 0.616, 0.496, 0.6, 0.544, 0.448, 0.656, 0.672, 0.688]`.
- Prototype non-air voxel counts: `[48, 47, 44, 44, 57, 48, 58, 50, 42, 41, 43, 46, 56, 39, 39, 47, 70, 58, 53, 45, 44, 38, 52, 53, 43, 42, 61, 47, 59, 38, 41, 58, 41, 60, 46, 76, 52, 41, 50, 75, 43, 62, 59, 50, 52, 40, 65, 41, 45, 39, 52, 60, 39, 44, 72, 48, 41, 48, 42, 44, 45, 44, 68, 55, 55, 63, 42, 50, 49, 40, 52, 44, 63, 52, 41, 48, 42, 48, 54, 53, 48, 63, 48, 63, 50, 57, 69, 43, 41, 39]`.
- Source sample counts: `{'medieval_1a': 7664, 'medieval_2a': 7664, 'medieval_2b': 7664, 'medieval_2c': 7664, 'medieval_2d': 7664, 'medieval_2e': 7664, 'medieval_2f': 6352, 'medieval_3a': 7664}`.
- Semantic bucket sample counts: `{'frame': 2128, 'opening': 20638, 'roof': 13181, 'wall': 2292, 'window': 21761}`.
- Candidate bucket counts: `{'frame': 2128, 'opening': 81064, 'roof': 17740, 'wall': 2292, 'window': 54512}`.

Held-out source distance ratios:
- medieval_1a: 1.000
- medieval_2a: 1.011
- medieval_2b: 1.055
- medieval_2c: 1.048
- medieval_2d: 1.033
- medieval_2e: 1.138
- medieval_2f: 0.994
- medieval_3a: 1.077

Clusters with at least 80% of samples from a single source:
- cluster 68: size 535, medieval_3a share 100.0%
- cluster 78: size 296, medieval_2e share 100.0%

## Recommended Next Steps Before Phase III

1. Inspect `datasets/phase2/previews/prototypes.svg` and `datasets/phase2/inspection/index.html` to prune or merge local cut-throughs that still do not read as usable building parts.
2. Run a stricter leave-one-source-out validation where SVD is fit only on training sources for each fold.
3. Consider increasing the tested k range around 50-90 now that buckets preserve rarer feature classes.
4. Use the current candidate tiles for a small adjacency-mining dry run, but keep the rule set marked experimental until visual and held-out checks pass.

Bottom line: the corrected result should generate less material noise and more recognizable architectural fragments, but it still needs visual pruning before becoming a final WFC-ready tile catalog.
