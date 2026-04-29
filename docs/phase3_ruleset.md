# Phase III Ruleset

Mined observed one-voxel adjacencies from 8 original, non-augmented Phase I arrays.
Assigned candidate patches to 65 Phase II medoid tiles with max assignment distance 0.52.
Observed adjacent patch pairs: 216134. Allowed directional tile pairs after overlap validation: 462.
Active tiles with nonzero observed assignments: 65/65.

## Constraint Policy

- A pair is allowed only when it was observed at least 1 time(s) in the original houses.
- The corresponding medoid prototypes must also agree across the shifted overlap with mismatch fraction <= 0.42.
- Opposite-direction rules are mirrored after mining so the ruleset is symmetric for propagation.
- Dead-end pruning enabled: False.

## Source Assignment Diagnostics

| Source | Candidates | Assigned | Assignment rate | Mean distance | p90 distance |
|---|---:|---:|---:|---:|---:|
| medieval_1a | 3281 | 3273 | 99.8% | 0.364 | 0.452 |
| medieval_2a | 6441 | 6432 | 99.9% | 0.353 | 0.436 |
| medieval_2b | 4250 | 4240 | 99.8% | 0.370 | 0.460 |
| medieval_2c | 7917 | 7817 | 98.7% | 0.366 | 0.456 |
| medieval_2d | 5915 | 5788 | 97.9% | 0.338 | 0.476 |
| medieval_2e | 10378 | 10250 | 98.8% | 0.301 | 0.456 |
| medieval_2f | 1587 | 1577 | 99.4% | 0.366 | 0.448 |
| medieval_3a | 2482 | 2464 | 99.3% | 0.379 | 0.468 |

## Dead-End Validation

Tiles with at least one empty critical face: 44.
- tile 0: -x, +x
- tile 1: -y, +y, -x, +x
- tile 2: -z, +z, -x, +x
- tile 3: -y, +y, -z, +z
- tile 4: -y, +y
- tile 5: -z, +z
- tile 6: -x
- tile 8: +x
- tile 10: -y, -x
- tile 13: -y, +y, -z, +z
- tile 14: -y, +y, -z, +z, -x, +x
- tile 15: -y, +y, -z
- tile 17: -x, +x
- tile 18: -z, +z, -x, +x
- tile 19: -z
- tile 20: -y, +y
- tile 21: -y, +y, -z, +z
- tile 22: -x, +x
- tile 23: -y, +x
- tile 24: -y, +y, -z, +z, -x, +x
- tile 25: -y, +y, -z, +z
- tile 26: +z
- tile 29: -y, +y, -x, +x
- tile 30: -z, +z, -x, +x
- tile 32: -y, +y, -z, +z, -x, +x
- tile 33: +y, -x, +x
- tile 35: +y, -x, +x
- tile 36: -x
- tile 37: -y
- tile 38: -y, +y, +z
- ... 14 more

## Outputs

- Binary ruleset: `datasets/phase3/ruleset.npz`
- JSON adjacency lists: `datasets/phase3/ruleset.json`
- Tile catalog: `datasets/phase3/tile_catalog.json` and `datasets/phase3/tile_catalog.csv`
