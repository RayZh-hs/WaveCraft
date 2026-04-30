# WaveCraft Algorithm Rethink

## Context

The current WaveCraft pipeline is centered on unsupervised local tile extraction and Wave Function Collapse:

1. Prepare a small set of curated Minecraft builds.
2. Extract overlapping 5x5x5 structural patches.
3. Cluster those patches into medoid prototype tiles.
4. Mine directional adjacency rules.
5. Run a 3D WFC solver to assemble new structures.

This was a reasonable first architecture for a six-build corpus, but the current results show that the representation is doing the wrong job. The learned tiles are local cut-throughs, not architectural parts. Many patches cross windows, roofs, walls, corners, and interior/exterior boundaries in ways that make them hard to reuse as WFC modules. WFC then has to assemble a building from pieces that do not encode the structure of a building.

The proposed Kaggle dataset of roughly 40,000 lower-quality Minecraft builds changes the opportunity. More data should not simply produce more local WFC tiles. Instead, it should be used to learn building-level structure: footprints, floors, facade rhythms, roof forms, openings, supports, and module distributions.

## Diagnosis

The current failure is structural, not just statistical.

Phase III currently mines many local patch adjacencies, but very few survive as meaningful reusable tile-pair constraints. Most prototype tiles have at least one dead face, which means they cannot function as clean kitbash modules. Phase IV then relies on loosened overlap compatibility to make solving possible, but those permissive constraints erase much of the architectural information the ruleset was meant to preserve.

The core mismatch is:

- WFC is local.
- Buildings are hierarchical.
- Sliding-window patches are accidental.
- Architectural modules are semantic.

A house is not defined by arbitrary 5x5x5 chunks. It is defined by a footprint, floors, walls, corners, roof spans, openings, supports, circulation, and style rules. Those global and mid-level relationships are not naturally expressible as tile adjacency alone.

## Recommendation

Shift WaveCraft from a WFC-centered system to a hierarchical generation system:

1. Use the large dataset to learn global and mid-level building priors.
2. Extract semantic modules instead of arbitrary voxel patches.
3. Assemble modules with a constraint solver or probabilistic grammar.
4. Use WFC only for bounded local detail.

The new architecture should be:

```text
dataset -> structure parser -> quality scorer -> grammar miner -> semantic module miner
        -> global planner -> constraint assembly -> local detail pass -> schematic export
```

WFC can remain in the project, but it should no longer be responsible for deciding building massing or high-level structure.

## Proposed Architecture

### 1. Structure Parser

Convert each build into a compact architectural analysis record.

For each normalized structure, extract:

- occupied bounding box
- ground contact footprint
- per-layer footprint masks
- connected components
- exterior shell
- interior air regions
- floor-height candidates
- facade planes
- roof volume
- roof slope and ridge candidates
- door and window candidates
- vertical supports
- material regions

Example output:

```json
{
  "source": "build_000123",
  "bounds": [21, 17, 14],
  "quality_score": 0.76,
  "footprint": {
    "type": "rectangle",
    "width": 17,
    "depth": 14,
    "area": 238
  },
  "floors": [0, 5, 10],
  "roof": {
    "type": "gable",
    "ridge_axis": "x",
    "eave_y": 11,
    "ridge_y": 15
  },
  "facades": [
    {
      "orientation": "+z",
      "length": 17,
      "height": 11,
      "openings": [
        {"type": "door", "x": 8, "y": 1, "width": 2, "height": 3},
        {"type": "window", "x": 4, "y": 6, "width": 2, "height": 2}
      ]
    }
  ]
}
```

This representation is more important than the raw blocks. Once this exists, the 40,000-build dataset becomes useful even if many builds are visually mediocre.

### 2. Quality Scoring

The large dataset should be weighted, not trusted uniformly.

Useful quality signals:

- single dominant connected component
- low floating-block ratio
- high supported-block ratio
- recognizable footprint
- exterior shell continuity
- plausible roof or top closure
- door-like opening near ground
- repeated facade rhythm
- reasonable density by volume
- low terrain contamination
- low random-noise material entropy

The score should not be a binary filter at first. Low-quality examples can still teach weak priors, such as common dimensions or footprint ratios. High-quality examples should dominate module extraction, style modeling, and evaluation.

Suggested quality buckets:

| Bucket | Use |
|---|---|
| `excellent` | Mine modules, style priors, evaluation exemplars |
| `usable` | Learn dimensions, footprints, roof frequencies, facade rhythms |
| `weak` | Learn only coarse statistics |
| `reject` | Exclude from training |

The current curated builds should be treated as high-weight style exemplars, not as ordinary records inside the 40,000-build corpus.

### 3. Grammar Mining

Learn distributions over building structure rather than voxel tiles.

Initial grammar variables:

- footprint type: rectangle, L-shape, T-shape, tower-attached, courtyard
- footprint dimensions
- floor count
- floor heights
- wall height
- roof type: gable, hip, flat, shed, tower cap, stepped
- roof axis
- roof overhang
- facade bay width
- door position
- window rhythm
- corner style
- material family per role

This can start with simple frequency tables, histograms, and conditional distributions.

Example:

```text
P(roof_type | footprint_type, width, depth)
P(floor_count | height)
P(window_spacing | facade_length, floor_count)
P(material_wall | material_roof, style_cluster)
P(door_offset | facade_length)
```

This does not need a neural model at first. A transparent probabilistic grammar will be easier to debug and will immediately encode global structure better than WFC.

### 4. Semantic Module Mining

Replace arbitrary patch clustering with semantic module extraction.

Candidate module types:

- `foundation_segment`
- `floor_slab`
- `wall_bay`
- `window_bay`
- `door_bay`
- `corner_pillar`
- `interior_wall`
- `roof_slope_span`
- `roof_ridge`
- `roof_eave`
- `roof_endcap`
- `gable_end`
- `tower_wall`
- `tower_roof`
- `porch`
- `stair_core`

Each module should be extracted from an architectural context and annotated with sockets.

Example module record:

```json
{
  "id": "wall_bay_0421",
  "role": "wall_bay",
  "size": [5, 1, 7],
  "orientation": "+z",
  "style_tags": ["medieval", "wood_dark", "stone_foundation"],
  "sockets": {
    "left": "wall_edge_wood_7",
    "right": "wall_edge_wood_7",
    "top": "roof_eave_support",
    "bottom": "floor_or_foundation",
    "back": "interior_air",
    "front": "exterior_air"
  },
  "features": {
    "has_window": true,
    "has_door": false,
    "solid_fraction": 0.41,
    "opening_count": 1
  },
  "quality_weight": 0.88
}
```

The important difference is that a module knows what it is. A wall bay can be repeated across a facade. A roof ridge knows that roof slopes must attach to both sides. A corner pillar knows that two wall planes meet there.

### 5. Global Planner

The planner samples a building specification before any blocks are placed.

Example plan:

```json
{
  "footprint": {"type": "rectangle", "width": 17, "depth": 13},
  "floors": 2,
  "floor_height": 5,
  "roof": {"type": "gable", "axis": "x", "overhang": 1},
  "front": "+z",
  "facades": {
    "+z": {"bay_pattern": ["corner", "window", "door", "window", "corner"]},
    "-z": {"bay_pattern": ["corner", "window", "window", "window", "corner"]},
    "+x": {"bay_pattern": ["corner", "wall", "window", "wall", "corner"]},
    "-x": {"bay_pattern": ["corner", "wall", "window", "wall", "corner"]}
  },
  "materials": {
    "foundation": "stone",
    "wall": "wood_dark",
    "roof": "dark_stone",
    "trim": "wood_medium"
  }
}
```

This phase ensures the output is a building before any local detail generation begins.

### 6. Constraint Assembly

Use a solver to fill the plan with compatible semantic modules.

Hard constraints:

- module dimensions must match assigned slots
- facade bay widths must sum to facade length
- corners must connect adjacent facade planes
- wall modules must align vertically across floors when required
- roof must cover the footprint
- ridge and slope modules must agree on roof axis
- door must connect exterior to interior
- upper-floor modules must be supported
- material roles must be compatible
- interior air should remain connected when generating habitable houses

Soft constraints:

- prefer high-quality source modules
- prefer style consistency
- prefer repeated facade rhythm
- prefer symmetry when the sampled grammar asks for it
- prefer modules from curated examples
- avoid overusing the same exact module

Implementation can start as custom backtracking or beam search. If constraints grow, use OR-Tools CP-SAT or another discrete constraint solver.

### 7. Local Detail Pass

WFC is still useful after structure exists.

Good WFC targets:

- wall surface trim
- roof texture variation
- mossy/cracked block variation
- beam and plank alternation
- small facade ornaments
- window trim variants
- lantern, trapdoor, fence, and flower placement

Bad WFC targets:

- footprint generation
- floor count
- wall enclosure
- roof topology
- door placement
- facade rhythm
- support structure

The detail pass should operate inside bounded semantic regions. For example, run a 2D or shallow-3D WFC on a wall surface after the wall bay layout already exists. This keeps WFC in the domain where local texture constraints are enough.

## Minimal Viable Redesign

The first implementation should be deliberately narrow.

### MVP Scope

Generate only rectangular medieval houses with gable roofs.

Supported features:

- rectangular footprint
- one or two floors
- fixed floor height
- four facades
- repeated wall/window bays
- one front door
- gable roof along X or Z
- simple material palette
- optional local decoration pass

This should produce better buildings than the current WFC system because it guarantees the major structural properties.

### MVP Phases

1. Add a structure parser that outputs JSON summaries for each training build.
2. Add a quality scorer.
3. Mine simple distributions for dimensions, floors, roof type, and facade openings.
4. Extract wall bays, corner modules, door bays, window bays, and gable roof modules.
5. Implement a rectangular-house planner.
6. Assemble modules with a small backtracking solver.
7. Export a schematic.
8. Add optional WFC/detail variation inside wall and roof surfaces.

## Relationship To Existing Code

The current code should not be thrown away wholesale.

Reusable pieces:

- palette reduction and block-state normalization from Phase I
- ornament masking and re-insertion concepts
- schematic export utilities
- category-to-block-state mapping
- visualization tools, with adaptation for semantic modules
- WFC solver, but only for local detail regions

Pieces to demote or replace:

- 5x5x5 patch clustering as the primary module source
- medoid prototype tiles as the main generation vocabulary
- global 3D WFC as the primary structure generator
- adjacency mining as the central learning objective

The old phases can be kept for comparison, but the new system should have a separate pipeline rather than trying to force semantic hierarchy into the existing tile pipeline.

## Suggested New Phases

```text
phase1_prepare_dataset.py          existing, lightly extended
phase2_parse_structures.py         new
phase3_score_quality.py            new
phase4_mine_grammar.py             new
phase5_mine_modules.py             new
phase6_generate_structure.py       new
phase7_detail_wfc.py               optional/new
```

Expected artifacts:

```text
datasets/phase2_structures/structure_records.jsonl
datasets/phase3_quality/quality_scores.json
datasets/phase4_grammar/grammar.json
datasets/phase5_modules/module_catalog.json
datasets/phase5_modules/modules/*.npy
datasets/phase6_generated/plan.json
datasets/phase6_generated/generated_house.schem
```

## Evaluation

Evaluate generated builds on structural criteria before aesthetics.

Suggested automatic metrics:

- connected structural component ratio
- floating-block ratio
- supported-block ratio
- exterior shell continuity
- roof coverage over footprint
- door existence
- window count and spacing
- interior air connectivity
- facade rhythm score
- material role consistency
- module socket violation count

Suggested qualitative review:

- silhouette readability
- roof plausibility
- facade coherence
- entrance placement
- repetition without obvious cloning
- style match to curated examples

The current WFC output can be kept as a baseline. The new system should beat it first on structural validity, then on visual quality.

## Longer-Term Options

Once the grammar and module pipeline works, machine learning can be added in targeted places.

Useful later additions:

- classifier for roof type and footprint type
- learned module embeddings for retrieval
- contrastive style embedding using curated builds as positives
- graph neural network over parsed building components
- transformer over symbolic building plans
- diffusion or autoregressive model for local facade/roof detail
- learned reranker over candidate assembled buildings

These should come after the symbolic representation exists. A neural model trained directly on raw voxels would still need constraints or repair to produce reliable buildings.

## Bottom Line

The main algorithmic change is:

> Stop learning arbitrary voxel tiles as the core representation. Learn building grammar, semantic modules, and sockets. Use constraints for structure. Use WFC only for local detail.

The 40,000-build dataset is valuable because it can teach the distribution of building structures. It should be parsed into architectural records, weighted by quality, and mined for reusable semantic modules. The six curated high-quality builds should guide style, calibration, and evaluation.

