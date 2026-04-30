# Phase III Ruleset

Mined observed one-voxel adjacencies from 8 original, non-augmented Phase I arrays.
Assigned candidate patches to 48 Phase II medoid tiles with max assignment distance 0.52.
Observed adjacent patch pairs: 212640. Allowed directional tile pairs after overlap validation: 58.
Active tiles with nonzero observed assignments: 48/48.

## Constraint Policy

- A pair is allowed only when it was observed at least 1 time(s) in the original houses.
- Air is treated as empty space in shifted overlaps; air may overlap any category.
- Two different non-air categories may overlap only when their collision fraction <= 0.00.
- Opposite-direction rules are mirrored after mining so the ruleset is symmetric for propagation.
- Dead-end pruning enabled: False.

## Source Assignment Diagnostics

| Source | Candidates | Assigned | Assignment rate | Mean distance | p90 distance |
|---|---:|---:|---:|---:|---:|
| medieval_1a | 3281 | 3254 | 99.2% | 0.377 | 0.464 |
| medieval_2a | 6441 | 6415 | 99.6% | 0.367 | 0.452 |
| medieval_2b | 4250 | 4190 | 98.6% | 0.380 | 0.472 |
| medieval_2c | 7917 | 7749 | 97.9% | 0.384 | 0.472 |
| medieval_2d | 5915 | 5706 | 96.5% | 0.352 | 0.488 |
| medieval_2e | 10378 | 10157 | 97.9% | 0.313 | 0.472 |
| medieval_2f | 1587 | 1577 | 99.4% | 0.373 | 0.460 |
| medieval_3a | 2482 | 2431 | 97.9% | 0.393 | 0.480 |

## Dead-End Validation

Tiles with at least one empty critical face: 45.
- tile 0: -y, +y, -z, +z, -x, +x
- tile 1: +x
- tile 2: -y, +y, -z, +z, -x, +x
- tile 3: -y, +y, -z, +z, -x, +x
- tile 6: -y, +y, -z, +z, -x, +x
- tile 7: -y, +y, -z, +z, -x, +x
- tile 8: -y, +y, -z, +z, -x, +x
- tile 10: -x
- tile 11: -y, +y, -z, +z, -x, +x
- tile 12: -y, +y, -z, +z, -x, +x
- tile 13: -y, +y, -z, +z, -x, +x
- tile 14: -y, +y, -z, +z, -x, +x
- tile 15: -y, +y, -z, +z, -x, +x
- tile 16: -y, +y, -z, +z, -x, +x
- tile 17: -y, +y, -z, +z, -x, +x
- tile 18: -y, +y, -z
- tile 19: -y, +y, -z, +z, -x, +x
- tile 20: -y, +y, -z, +z, -x, +x
- tile 21: -y, +y, -z, +z, -x, +x
- tile 22: -y, -z, +z, -x, +x
- tile 23: -y, +y, -z, +z, -x, +x
- tile 24: -y, +y, -z, +z, -x, +x
- tile 25: -y, +y, -z, +z, -x, +x
- tile 26: -y, +y, +z
- tile 27: -y, +y, -z, +z, -x, +x
- tile 28: -y, +y, -z, +z, -x, +x
- tile 29: -y, +y, -z, +z, -x, +x
- tile 30: -y, +y, -z, +z, -x, +x
- tile 31: -y, +y, -z, +z, -x, +x
- tile 32: -y, +y, -z, +z, -x, +x
- ... 15 more

## Outputs

- Binary ruleset: `datasets/phase3/ruleset.npz`
- JSON adjacency lists: `datasets/phase3/ruleset.json`
- Tile catalog: `datasets/phase3/tile_catalog.json` and `datasets/phase3/tile_catalog.csv`
