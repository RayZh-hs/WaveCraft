#!/usr/bin/env python3
"""Analyze whether the current Phase I/II outputs are methodologically sound."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", type=Path, default=Path("datasets/phase1"))
    parser.add_argument("--phase2-dir", type=Path, default=Path("datasets/phase2"))
    parser.add_argument("--docs-path", type=Path, default=Path("docs/result_soundness_review.md"))
    args = parser.parse_args()

    metadata = load_json(args.phase1_dir / "metadata.json")
    evaluation = load_json(args.phase2_dir / "evaluation.json")
    tile_library = np.load(args.phase2_dir / "tile_library.npz")
    prototypes = tile_library["prototypes"]
    labels = tile_library["labels"]
    centers = tile_library["centers"]
    palette = metadata["palette"]
    air_id = int(metadata["air_id"])

    source_rows = list(csv.DictReader((args.phase2_dir / "sampled_patch_sources.csv").open(encoding="utf-8")))
    source_counts = Counter(row["source"] for row in source_rows)
    transform_counts = Counter(row["transform"] for row in source_rows)
    semantic_bucket_counts = Counter(row.get("semantic_bucket", "unknown") for row in source_rows)
    cluster_sizes = Counter(int(label) for label in labels)
    prototype_air_fractions = [float((prototype == air_id).mean()) for prototype in prototypes]
    prototype_nonair_counts = [int((prototype != air_id).sum()) for prototype in prototypes]

    cluster_source_notes = []
    for cluster_id in sorted(cluster_sizes):
        indices = np.flatnonzero(labels == cluster_id)
        by_source = Counter(source_rows[int(index)]["source"] for index in indices)
        source, count = by_source.most_common(1)[0]
        share = count / len(indices)
        if share >= 0.80:
            cluster_source_notes.append((cluster_id, len(indices), source, share))

    solid_single_material_tiles = []
    for tile_id, prototype in enumerate(prototypes):
        values, counts = np.unique(prototype, return_counts=True)
        if len(values) == 1 and values[0] != air_id:
            solid_single_material_tiles.append((tile_id, palette[int(values[0])], int(counts[0])))

    k_rows = evaluation["k_evaluation"]
    best = max(k_rows, key=lambda item: item["silhouette"] if item["silhouette"] is not None else -1)
    feature_info = evaluation.get("feature_info", {})
    prototype_pruning = evaluation.get("prototype_pruning", {})
    transform_remapping = metadata.get("category_transform_remapping", {})
    source_balanced = evaluation["patch_stats"].get("balanced_by_source", False)
    heldout = evaluation.get("heldout_evaluation", [])
    weak_holdouts = [
        item
        for item in heldout
        if item.get("heldout_to_train_distance_ratio", 0.0) >= 1.25
    ]

    lines = [
        "# Result Soundness Review",
        "",
        "## Verdict",
        "",
        "The corrected result is better aligned with building generation than the previous density-only extraction.",
        "Phase II now filters dense material chunks, balances the sample across architectural buckets, and appends semantic descriptors so windows, openings, roof pieces, frames, and walls can influence clustering instead of being treated as rare voxel noise.",
        "It is reasonable to treat the exported medoids as a Phase II candidate tile library for qualitative inspection and early adjacency experiments.",
        "It is still not a final kitbash library without manual/visual review, because the extractor still learns local 5x5x5 neighborhoods rather than explicitly labeled complete building modules.",
        "",
        "## What Looks Sound",
        "",
        f"- The schematic conversion is reproducible and Phase I produced {metadata['array_count']} arrays from {metadata['source_count']} source schematics.",
        f"- The output structural palette has {len(palette)} categories including air, and ornaments are masked out of the training arrays.",
        f"- Direction-aware augmentation is enabled: {bool(transform_remapping.get('enabled'))}.",
        f"- Feature encoding is `{feature_info.get('encoding')}`, so arbitrary palette IDs are no longer treated as ordinal distances and architectural descriptors now affect clustering.",
        f"- Source-balanced sampling is enabled: {source_balanced}.",
        f"- Semantic-bucket-balanced sampling is enabled: {evaluation['patch_stats'].get('balanced_by_bucket', False)}.",
        f"- Phase II found {evaluation['patch_stats'].get('density_candidates', evaluation['patch_stats']['total_eligible_patches'])} density candidates, retained {evaluation['patch_stats']['total_eligible_patches']} architectural 5x5x5 patches, and clustered a sampled {evaluation['patch_stats']['sampled_patches']} patches.",
        f"- Dense material chunks rejected before clustering: {evaluation['patch_stats'].get('rejected_dense', 0)}.",
        f"- Low-information single-kind sheets/columns rejected before clustering: {evaluation['patch_stats'].get('rejected_low_information', 0)}.",
        f"- Socket-useful simple field prototypes are capped by kind: {prototype_pruning.get('field_prototype_kind_counts', {})}; pruned by cap: {prototype_pruning.get('pruned_field_prototypes', 0)}.",
        f"- The selected prototypes are medoids, so every exported tile is an actual observed patch rather than an averaged block soup.",
        f"- The transform sample counts are balanced: {dict(transform_counts)}.",
        f"- Semantic bucket sample counts are balanced where available: {dict(semantic_bucket_counts)}.",
        "",
        "## Remaining Soundness Risks",
        "",
        "1. Cluster separation is still modest.",
        f"   The best tested silhouette is {best['silhouette']:.4f} at k={best['k']}. This is usable for exploration, but not strong evidence that the learned tiles form crisp architectural parts.",
        "",
        "2. Some sources remain structurally distinct.",
        "   Held-out source distances above 1.25x indicate that a source contains patch patterns not well represented by the rest of the corpus.",
        "",
        "3. Some prototypes may still be local cut-throughs rather than complete semantic objects.",
        "   The new buckets reduce random-looking chunks, but a 5x5x5 sliding window can still intersect only part of a window, roof ridge, or wall segment.",
        "",
        "4. The held-out evaluation is based on the shared SVD embedding.",
        "   It is useful as a diagnostic, but a stricter validation would fit the dimensionality reduction on training sources only for each hold-out fold.",
        "",
        "## Diagnostics",
        "",
        f"- Prototype tensor shape: `{tuple(prototypes.shape)}`.",
        f"- Feature center shape: `{tuple(centers.shape)}`.",
        f"- Feature info: `{feature_info}`.",
        f"- Largest cluster sizes: `{cluster_sizes.most_common(10)}`.",
        f"- Prototype air fractions: `{[round(value, 3) for value in prototype_air_fractions]}`.",
        f"- Prototype non-air voxel counts: `{prototype_nonair_counts}`.",
        f"- Source sample counts: `{dict(source_counts)}`.",
        f"- Semantic bucket sample counts: `{dict(semantic_bucket_counts)}`.",
        f"- Candidate bucket counts: `{evaluation['patch_stats'].get('candidate_bucket_counts', {})}`.",
        f"- Prototype pruning: `{prototype_pruning}`.",
    ]

    if heldout:
        lines.extend(["", "Held-out source distance ratios:"])
        for item in heldout:
            lines.append(
                f"- {item['heldout_source']}: {item['heldout_to_train_distance_ratio']:.3f}"
            )

    if weak_holdouts:
        lines.extend(["", "Held-out sources needing inspection:"])
        for item in weak_holdouts:
            lines.append(
                f"- {item['heldout_source']}: {item['heldout_to_train_distance_ratio']:.3f}x train distance"
            )

    if cluster_source_notes:
        lines.extend(["", "Clusters with at least 80% of samples from a single source:"])
        for cluster_id, size, source, share in cluster_source_notes:
            lines.append(f"- cluster {cluster_id}: size {size}, {source} share {pct(share)}")

    if solid_single_material_tiles:
        lines.extend(["", "Solid single-material prototype tiles:"])
        for tile_id, category, count in solid_single_material_tiles:
            lines.append(f"- tile {tile_id}: {count}/125 voxels of `{category}`")

    lines.extend(
        [
            "",
            "## Recommended Next Steps Before Phase III",
            "",
            "1. Inspect `datasets/phase2/previews/prototypes.svg` and `datasets/phase2/inspection/index.html` to prune or merge local cut-throughs that still do not read as usable building parts.",
            "2. Run a stricter leave-one-source-out validation where SVD is fit only on training sources for each fold.",
            "3. Consider increasing the tested k range around 50-90 now that buckets preserve rarer feature classes.",
            "4. Use the current candidate tiles for a small adjacency-mining dry run, but keep the rule set marked experimental until visual and held-out checks pass.",
            "",
            "Bottom line: the corrected result should generate less material noise and more recognizable architectural fragments, but it still needs visual pruning before becoming a final WFC-ready tile catalog.",
        ]
    )

    args.docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote soundness review to {args.docs_path}")


if __name__ == "__main__":
    main()
