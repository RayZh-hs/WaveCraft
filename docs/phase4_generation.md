# Phase IV Generation

Generated a 10x12x10 WFC cell grid from the Phase III ruleset.
The overlapping tile reconstruction produced a block volume of 14x16x14 (Y,Z,X).
Output schematic: `datasets/phase4/generated_house.schem`.

## Solver

- Seed: 47
- Successful attempt: 0
- Decisions: 49
- Backtracks: 0
- Propagations: 6037
- Active tiles after preparation: 19
- Allowed directional adjacencies after preparation: 198
- Dead-end tile avoidance: True
- Dead-end repair enabled: False

## Reconstruction

- Solid vote weight: 5.00
- Raw majority non-air voxels: 631
- Weighted non-air voxels: 1724
- Weighted air fraction: 45.0%

## Post-Processing

- Palette variation rate: 0.080
- Ornament rate: 0.004
- Lanterns added: 1

## Most Used Tiles

- tile 42: 390 cells
- tile 56: 240 cells
- tile 7: 120 cells
- tile 31: 120 cells
- tile 62: 120 cells
- tile 16: 90 cells
- tile 28: 73 cells
- tile 61: 47 cells

## Files

- Schematic: `datasets/phase4/generated_house.schem`
- Chosen tile grid: `datasets/phase4/chosen_tiles.npy`
- Generated category volume: `datasets/phase4/generated_categories.npy`
- Generation report: `datasets/phase4/generation_report.json`
