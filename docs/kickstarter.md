# WaveCraft: Project Outline

Procedural generation of Minecraft structures typically requires hand-authored kitbash part libraries and connection rules. **WaveCraft** automates this pipeline: given only 6 integrated medieval houses, it uses unsupervised machine learning to reverse-engineer a kitbash tile set and derives Wave Function Collapse (WFC) adjacency rules without manual decomposition. The goal is style-consistent, novel building generation from minimal final-structure data.

---

## Phase I: Data Preparation & Augmentation
*Goal: Convert integrated builds into a trainable corpus despite tiny dataset size.*

- **Voxelization**: Import 6 `.nbt`/schematic houses into dense 3D arrays (block ID + metadata).
- **Palette Reduction**: Collapse rare/gradient blocks into canonical types (e.g., all wood variants → 4 tonal categories). This prevents clustering from fragmenting on decorative noise.
- **Structural Masking**: Tag blocks as *structural* (load-bearing walls, roof volumes) vs. *ornamental* (flower pots, item frames). The ML pipeline operates only on structural blocks; ornaments are re-inserted post-generation.
- **Data Augmentation**: Generate mirrored (X, Z) and 90° rotated copies of each house.  
  *Result: 6 houses → 24 effective structural volumes.*

**Deliverable**: Cleaned 3D NumPy arrays ready for patch extraction.

---

## Phase II: Unsupervised Kitbash Extraction
*Goal: Discover discrete, reusable 3D tiles via clustering.*

- **Sliding-Window Patch Extraction**: Extract all overlapping 5×5×5 (and optionally 7×7×7) patches from the augmented dataset. Exclude patches that are >70% air.
- **Feature Encoding**: Encode each patch as a vector:
  - Categorical block IDs → reduced palette indices
  - Optional: append relative height (Y-level) to help clustering distinguish foundations vs. roof peaks
- **Clustering**: Run **k-means** (elbow method or silhouette score to pick k ≈ 30–50) on patch vectors. Each cluster centroid represents a candidate kitbash tile.
- **Prototype Cleaning**: For each cluster, select the *most central actual patch* (medoid) as the canonical tile. Discard outlier clusters with <3 members (noise).

**Deliverable**: A tile library of 30–50 validated 3D kitbash prototypes with visual previews.

---

## 4. Phase III: WFC Rule Construction (Inverse WFC)
*Goal: Mine valid adjacencies and frequencies from the original data.*

- **Adjacency Mining**: Re-scan the original (non-augmented) houses. For every pair of adjacent patches that belong to identified clusters, record:
  - Tile A ↔ Tile B on face *f* (±X, ±Y, ±Z)
  - Co-occurrence frequency
- **Socket Definition**: Two tiles are compatible on a face if their voxel overlap is consistent (e.g., a wall block aligns with a wall block, not air). Derive hard constraints from observed adjacencies; treat unobserved but structurally logical pairs as *forbidden* unless manually validated.
- **Weight Calibration**: Set each tile’s WFC weight proportional to its cluster frequency in the dataset. This preserves the style distribution (e.g., plain walls appear more often than decorative dormers).
- **Constraint Validation**: Manually inspect the adjacency matrix for dead-end tiles (tiles with no valid neighbor on a critical face). Prune or merge them.

**Deliverable**: A complete WFC ruleset—tile catalog, 6-directional adjacency lists, and frequency weights.

---

## Phase IV: Procedural Generation Engine
*Goal: Assemble novel structures using the learned rules.*

- **WFC Solver**: Implement/generate on a 3D grid using the derived tile set and rules.
  - Grid initialization: user-defined bounding box (e.g., 20×15×20) or shape grammar seed.
  - Lowest-entropy heuristic with backtracking for contradiction recovery.
- **Post-Processing**:
  - Resolve block ID variations (re-introduce reduced palette diversity).
  - Add ornamental blocks (vegetation, lighting) via simple procedural rules after structural WFC completion.
- **Output**: `.nbt` or schematic files importable into Minecraft.

**Deliverable**: Working generator that produces novel, structurally valid medieval houses.
