# Phase IV Generation

Generated a 10x12x10 WFC cell grid from the Phase III ruleset.
The overlapping tile reconstruction produced a block volume of 41x49x41 (Y,Z,X).
Output schematic: `datasets/phase4/generated_house.schem`.

## Solver

- Seed: 47
- Successful attempt: 0
- Decisions: 757
- Backtracks: 9
- Propagations: 7261
- Active tiles after preparation: 48
- Allowed directional adjacencies after preparation: 3350
- Source ruleset allowed adjacencies: 58
- Tile stride: 4
- Stride-overlap allowed adjacencies: 3350
- Dead-end tile avoidance: False
- Dead-end repair enabled: False
- Scaffold: inferred

## Scaffold

- Mined training patch assignments: 41479
- Target role cells: {'foundation': 120, 'exterior_wall': 241, 'interior': 560, 'opening': 39, 'roof': 240}
- Roof start y norm: 0.82
- Foundation end y norm: 0.14

## Reconstruction

- Tile stride: 4
- Non-air union threshold: 0.50
- Raw majority non-air voxels: 12636
- Union-poll non-air voxels: 12636
- Union-poll detail voxels: 8016
- Union-poll air fraction: 84.7%
- Mean non-air probability: 33.5%

## Post-Processing

- Palette variation rate: 0.080
- Ornament rate: 0.004
- Lanterns added: 14

## Most Used Tiles

- tile 1: 390 cells
- tile 10: 379 cells
- tile 39: 173 cells
- tile 22: 113 cells
- tile 42: 92 cells
- tile 9: 31 cells
- tile 20: 13 cells
- tile 23: 8 cells
- tile 27: 1 cells

## Files

- Schematic: `datasets/phase4/generated_house.schem`
- Chosen tile grid: `datasets/phase4/chosen_tiles.npy`
- Generated category volume: `datasets/phase4/generated_categories.npy`
- Generation report: `datasets/phase4/generation_report.json`
