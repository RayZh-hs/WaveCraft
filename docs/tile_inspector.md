# Tile Inspector

Generate the static tile inspection report after Phase I and Phase II artifacts exist:

```bash
uv run wavecraft-render-tiles
```

Outputs:

- `datasets/phase2/inspection/index.html`: browser report for reviewing medoid tiles.
- `datasets/phase2/inspection/tile_analysis.json`: machine-readable tile diagnostics.

The report includes a shape-aware isometric voxel approximation, one selected Y-layer slice at a time, composition tables, face/socket summaries, shape counts, source and transform counts, cluster size, medoid provenance, and flags for sparse, source-specific, homogeneous, or low-contact tiles.

Use the `y=0` through `y=4` buttons on each tile card to inspect a single layer. Shape glyphs are visible by default; exact text labels are hidden until `Show cell labels` is enabled in the top toolbar.

Shape notation in the slice cells:

- `STN^`, `STEv`, etc.: stairs; `N/E/S/W` is facing, `^` is top half, `v` is bottom half.
- `SL^`, `SLv`, `SL2`: slab top, slab bottom, double slab.
- `WL`, `FN`, `PN`: wall, fence, pane. Connection spokes show north/east/south/west links where present.
- `DR`, `TD`, `FG`: door, trapdoor, fence gate. Direction suffixes show facing where available.
- `LGx`, `LGy`, `LGz`: log axis.

Generated inspection files are ignored through `datasets/.gitignore`.

## In-Game Schematic Gallery

Generate a Sponge schematic containing all prototype tiles laid out in a grid:

```bash
uv run wavecraft-export-tile-schem
```

Output:

- `datasets/phase2/inspection/tile_gallery.schem`

The exporter maps reduced WaveCraft categories back to representative Minecraft block states. For example, `dark_stone:stairs[...]` becomes deepslate brick stairs, `wood_dark:slab[...]` becomes spruce slabs, and `glass:pane[...]` becomes glass panes. This is meant for visual inspection of learned geometry, not for restoring exact original material choices.

Useful options:

```bash
uv run wavecraft-export-tile-schem --columns 5 --spacing 5
uv run wavecraft-export-tile-schem --output /tmp/tile_gallery.schem
```
