# Phase I Observations

Loaded 8 source schematics and wrote 32 augmented structural arrays.

## Source Volumes

| Source | Shape (Y,Z,X) | Structural Voxels | Structural Fraction | Raw Palette |
|---|---:|---:|---:|---:|
| medieval_1a | (32, 32, 32) | 2232 | 0.068 | 145 |
| medieval_2a | (32, 32, 32) | 4068 | 0.124 | 229 |
| medieval_2b | (32, 32, 48) | 2795 | 0.057 | 240 |
| medieval_2c | (32, 32, 64) | 5101 | 0.078 | 274 |
| medieval_2d | (64, 32, 32) | 15453 | 0.236 | 294 |
| medieval_2e | (64, 32, 48) | 41694 | 0.424 | 298 |
| medieval_2f | (32, 32, 32) | 1121 | 0.034 | 135 |
| medieval_3a | (32, 32, 32) | 1651 | 0.050 | 170 |

## Palette Reduction

Reduced 767 raw block states to 384 output structural categories including air.
An internal `ornament` marker is used during masking, but ornamental voxels are written as air in the cleaned arrays and are not part of the output palette.
Material variants are collapsed before learning: dark stones such as deepslate, blackstone, basalt, and tuff map to `dark_stone`, while wood species map into broad tonal families and keep geometry/orientation where it affects structure.
Augmentations remap directional categories after spatial transforms, including `facing` values, directional side keys, stair handedness, door hinges, and x/z axes.

Most common raw reductions before masking:

- `air`: 331242
- `wood_medium:log[axis=y]`: 37297
- `stone:slab[type=bottom]`: 12305
- `ornament`: 4243
- `wood_dark:planks`: 3381
- `stone:full`: 2287
- `wood_dark:log[axis=y]`: 2075
- `stone:bricks`: 1883
- `wood_light:planks`: 914
- `wood_medium:planks`: 779
- `dark_stone:tiles`: 713
- `wood_dark:slab[type=top]`: 607
- `wood_dark:log[axis=x]`: 590
- `wood_dark:log[axis=z]`: 527
- `dark_stone:bricks`: 489
- `dark_stone:stairs[facing=south,half=bottom,shape=straight]`: 390
- `wood_dark:stairs[facing=east,half=top,shape=straight]`: 389
- `wood_dark:stairs[facing=south,half=top,shape=straight]`: 382
- `dark_stone:stairs[facing=north,half=bottom,shape=straight]`: 372
- `wood_dark:stairs[facing=west,half=top,shape=straight]`: 350

Noise note: the source data intentionally swaps deepslate bricks with nearby dark masonry variants for visual interest. This is treated as palette noise for learning and is collapsed into the same dark-stone family while preserving stairs/slabs/walls when present.

Ornaments such as flowers, torches, lanterns, signs, buttons, carpets, containers, and foliage are masked to air in the structural arrays. This keeps the Phase II clustering focused on house massing and load-bearing shapes.
