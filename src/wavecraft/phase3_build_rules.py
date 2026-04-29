#!/usr/bin/env python3
"""Phase III: mine inverse-WFC adjacency rules from learned tiles."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from wavecraft.phase2_extract_tiles import (
    candidate_positions,
    category_kind,
    face_contact_counts,
    write_json,
)


AIR = "air"

DIRECTIONS = (
    ("-y", np.array((-1, 0, 0), dtype=np.int32)),
    ("+y", np.array((1, 0, 0), dtype=np.int32)),
    ("-z", np.array((0, -1, 0), dtype=np.int32)),
    ("+z", np.array((0, 1, 0), dtype=np.int32)),
    ("-x", np.array((0, 0, -1), dtype=np.int32)),
    ("+x", np.array((0, 0, 1), dtype=np.int32)),
)
DIRECTION_INDEX = {name: index for index, (name, _delta) in enumerate(DIRECTIONS)}
OPPOSITE_DIRECTION = {
    "-y": "+y",
    "+y": "-y",
    "-z": "+z",
    "+z": "-z",
    "-x": "+x",
    "+x": "-x",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_phase1_originals(metadata_path: Path) -> tuple[dict[str, Any], list[tuple[dict[str, Any], np.ndarray]]]:
    metadata = load_json(metadata_path)
    arrays = []
    for item in metadata["outputs"]:
        if item.get("transform") != "original":
            continue
        arrays.append((item, np.load(item["array"])))
    if not arrays:
        raise SystemExit(f"No original Phase I arrays found in {metadata_path}")
    return metadata, arrays


def patch_assignment_distances(
    patches: np.ndarray,
    prototypes: np.ndarray,
    air_id: int,
    air_mismatch_weight: float,
    category_mismatch_weight: float,
) -> np.ndarray:
    patch_flat = patches.reshape((patches.shape[0], -1))
    proto_flat = prototypes.reshape((prototypes.shape[0], -1))
    distances = np.empty((patch_flat.shape[0], proto_flat.shape[0]), dtype=np.float32)
    patch_air = patch_flat == air_id
    proto_air = proto_flat == air_id
    normalizer = patch_flat.shape[1] * max(air_mismatch_weight, category_mismatch_weight)

    for tile_id, prototype in enumerate(proto_flat):
        mismatched = patch_flat != prototype
        air_mismatch = mismatched & (patch_air | proto_air[tile_id])
        category_mismatch = mismatched & ~air_mismatch
        distances[:, tile_id] = (
            air_mismatch.sum(axis=1, dtype=np.float32) * air_mismatch_weight
            + category_mismatch.sum(axis=1, dtype=np.float32) * category_mismatch_weight
        ) / normalizer
    return distances


def assign_patches_to_tiles(
    volume: np.ndarray,
    positions: np.ndarray,
    prototypes: np.ndarray,
    air_id: int,
    max_assignment_distance: float,
    air_mismatch_weight: float,
    category_mismatch_weight: float,
    batch_size: int,
) -> tuple[dict[tuple[int, int, int], int], dict[str, Any], Counter[int]]:
    assignments: dict[tuple[int, int, int], int] = {}
    distance_values: list[float] = []
    tile_counts: Counter[int] = Counter()
    rejected = 0

    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        patches = np.stack(
            [
                volume[y : y + prototypes.shape[1], z : z + prototypes.shape[2], x : x + prototypes.shape[3]]
                for y, z, x in batch_positions
            ]
        )
        distances = patch_assignment_distances(
            patches,
            prototypes,
            air_id,
            air_mismatch_weight,
            category_mismatch_weight,
        )
        best_tiles = np.argmin(distances, axis=1)
        best_distances = distances[np.arange(distances.shape[0]), best_tiles]
        for position, tile_id, distance in zip(batch_positions, best_tiles, best_distances, strict=True):
            distance_float = float(distance)
            distance_values.append(distance_float)
            if distance_float > max_assignment_distance:
                rejected += 1
                continue
            key = tuple(int(value) for value in position)
            assignments[key] = int(tile_id)
            tile_counts[int(tile_id)] += 1

    assigned = len(assignments)
    stats = {
        "candidate_patches": int(len(positions)),
        "assigned_patches": int(assigned),
        "rejected_by_assignment_distance": int(rejected),
        "assignment_rate": float(assigned / max(len(positions), 1)),
        "distance_min": float(min(distance_values)) if distance_values else None,
        "distance_mean": float(np.mean(distance_values)) if distance_values else None,
        "distance_p50": float(np.quantile(distance_values, 0.50)) if distance_values else None,
        "distance_p90": float(np.quantile(distance_values, 0.90)) if distance_values else None,
        "distance_max": float(max(distance_values)) if distance_values else None,
    }
    return assignments, stats, tile_counts


def prototype_overlap_mismatch(prototype_a: np.ndarray, prototype_b: np.ndarray, direction: str) -> float:
    if direction == "+y":
        left = prototype_a[1:, :, :]
        right = prototype_b[:-1, :, :]
    elif direction == "-y":
        left = prototype_a[:-1, :, :]
        right = prototype_b[1:, :, :]
    elif direction == "+z":
        left = prototype_a[:, 1:, :]
        right = prototype_b[:, :-1, :]
    elif direction == "-z":
        left = prototype_a[:, :-1, :]
        right = prototype_b[:, 1:, :]
    elif direction == "+x":
        left = prototype_a[:, :, 1:]
        right = prototype_b[:, :, :-1]
    elif direction == "-x":
        left = prototype_a[:, :, :-1]
        right = prototype_b[:, :, 1:]
    else:
        raise ValueError(direction)
    return float(np.count_nonzero(left != right) / left.size)


def overlap_mismatch_table(prototypes: np.ndarray) -> np.ndarray:
    table = np.zeros((len(DIRECTIONS), prototypes.shape[0], prototypes.shape[0]), dtype=np.float32)
    for direction_index, (direction, _delta) in enumerate(DIRECTIONS):
        for tile_a in range(prototypes.shape[0]):
            for tile_b in range(prototypes.shape[0]):
                table[direction_index, tile_a, tile_b] = prototype_overlap_mismatch(
                    prototypes[tile_a],
                    prototypes[tile_b],
                    direction,
                )
    return table


def mine_observed_adjacencies(
    assignments: dict[tuple[int, int, int], int],
    counts: np.ndarray,
) -> int:
    observed_pairs = 0
    assigned_positions = set(assignments)
    for position, tile_a in assignments.items():
        position_array = np.asarray(position, dtype=np.int32)
        for direction, delta in DIRECTIONS:
            neighbor = tuple(int(value) for value in (position_array + delta))
            if neighbor not in assigned_positions:
                continue
            tile_b = assignments[neighbor]
            counts[DIRECTION_INDEX[direction], tile_a, tile_b] += 1
            observed_pairs += 1
    return observed_pairs


def prune_dead_end_tiles(allowed: np.ndarray, active_tiles: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    active = active_tiles.copy()
    iterations = 0
    removed_by_iteration: list[list[int]] = []

    while True:
        allowed_active = allowed[:, active, :][:, :, active]
        active_indices = np.flatnonzero(active)
        dead = []
        for local_index, tile_id in enumerate(active_indices):
            for direction_index in range(len(DIRECTIONS)):
                if not np.any(allowed_active[direction_index, local_index, :]):
                    dead.append(int(tile_id))
                    break
        if not dead:
            break
        active[dead] = False
        removed_by_iteration.append(dead)
        iterations += 1

    return active, {
        "iterations": iterations,
        "removed_tile_ids": [tile_id for group in removed_by_iteration for tile_id in group],
        "removed_by_iteration": removed_by_iteration,
    }


def rule_lists(allowed: np.ndarray) -> dict[str, dict[str, list[int]]]:
    rules: dict[str, dict[str, list[int]]] = {}
    for tile_id in range(allowed.shape[1]):
        rules[str(tile_id)] = {}
        for direction, _delta in DIRECTIONS:
            direction_index = DIRECTION_INDEX[direction]
            rules[str(tile_id)][direction] = [
                int(value)
                for value in np.flatnonzero(allowed[direction_index, tile_id])
            ]
    return rules


def observed_count_lists(counts: np.ndarray) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for tile_id in range(counts.shape[1]):
        result[str(tile_id)] = {}
        for direction, _delta in DIRECTIONS:
            direction_index = DIRECTION_INDEX[direction]
            result[str(tile_id)][direction] = {
                str(neighbor): int(count)
                for neighbor, count in enumerate(counts[direction_index, tile_id])
                if count > 0
            }
    return result


def face_summaries(prototypes: np.ndarray, palette: list[str], air_id: int) -> list[dict[str, Any]]:
    summaries = []
    for tile_id, prototype in enumerate(prototypes):
        contact_counts = face_contact_counts(prototype, air_id)
        values, counts = np.unique(prototype, return_counts=True)
        category_counts = [
            {"category": palette[int(value)], "count": int(count)}
            for value, count in sorted(zip(values, counts), key=lambda item: int(item[1]), reverse=True)
            if int(value) != air_id
        ]
        kind_counts: Counter[str] = Counter()
        for value, count in zip(values, counts, strict=True):
            if int(value) == air_id:
                continue
            kind_counts[category_kind(palette[int(value)])] += int(count)
        summaries.append(
            {
                "tile_id": tile_id,
                "non_air_voxels": int(np.count_nonzero(prototype != air_id)),
                "air_fraction": float(np.count_nonzero(prototype == air_id) / prototype.size),
                "face_contact_counts": contact_counts,
                "top_categories": category_counts[:8],
                "kind_counts": dict(kind_counts.most_common()),
            }
        )
    return summaries


def write_tile_catalog_csv(path: Path, catalog: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tile_id",
                "weight",
                "phase2_cluster_id",
                "phase2_cluster_size",
                "phase3_assignment_count",
                "non_air_voxels",
                "air_fraction",
                "dead_end_faces",
            ],
        )
        writer.writeheader()
        for item in catalog:
            writer.writerow(
                {
                    "tile_id": item["tile_id"],
                    "weight": item["weight"],
                    "phase2_cluster_id": item["phase2_cluster_id"],
                    "phase2_cluster_size": item["phase2_cluster_size"],
                    "phase3_assignment_count": item["phase3_assignment_count"],
                    "non_air_voxels": item["non_air_voxels"],
                    "air_fraction": item["air_fraction"],
                    "dead_end_faces": ",".join(item["dead_end_faces"]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", type=Path, default=Path("datasets/phase1"))
    parser.add_argument("--phase2-dir", type=Path, default=Path("datasets/phase2"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/phase3"))
    parser.add_argument("--docs-path", type=Path, default=Path("docs/phase3_ruleset.md"))
    parser.add_argument("--max-assignment-distance", type=float, default=0.52)
    parser.add_argument("--max-overlap-mismatch", type=float, default=0.42)
    parser.add_argument("--min-observed-count", type=int, default=1)
    parser.add_argument("--air-mismatch-weight", type=float, default=2.0)
    parser.add_argument("--category-mismatch-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--prune-dead-ends", action="store_true")
    args = parser.parse_args()

    metadata, arrays = load_phase1_originals(args.phase1_dir / "metadata.json")
    evaluation = load_json(args.phase2_dir / "evaluation.json")
    tile_library = np.load(args.phase2_dir / "tile_library.npz")
    prototypes = tile_library["prototypes"]
    palette = metadata["palette"]
    air_id = int(metadata["air_id"])
    window = int(evaluation["window"])
    kind_by_id = [category_kind(category) for category in palette]

    if tuple(prototypes.shape[1:]) != (window, window, window):
        raise SystemExit(f"Prototype shape {prototypes.shape[1:]} does not match evaluation window {window}")

    tile_count = int(prototypes.shape[0])
    observed_counts = np.zeros((len(DIRECTIONS), tile_count, tile_count), dtype=np.int32)
    total_tile_counts: Counter[int] = Counter()
    per_source_stats = []
    observed_pairs = 0

    for item, volume in arrays:
        positions, buckets, filter_stats = candidate_positions(
            volume=volume,
            window=window,
            air_id=air_id,
            max_air_fraction=float(evaluation["max_air_fraction"]),
            max_non_air_fraction=float(evaluation["max_non_air_fraction"]),
            min_architectural_score=float(evaluation["min_architectural_score"]),
            max_simple_kind_fraction=float(evaluation["max_simple_kind_fraction"]),
            max_simple_with_tiny_feature_fraction=float(evaluation["max_simple_with_tiny_feature_fraction"]),
            min_meaningful_feature_voxels=int(evaluation["min_meaningful_feature_voxels"]),
            min_field_boundary_contact=int(evaluation["min_field_boundary_contact"]),
            kind_by_id=kind_by_id,
        )
        assignments, assignment_stats, tile_counts = assign_patches_to_tiles(
            volume=volume,
            positions=positions,
            prototypes=prototypes,
            air_id=air_id,
            max_assignment_distance=args.max_assignment_distance,
            air_mismatch_weight=args.air_mismatch_weight,
            category_mismatch_weight=args.category_mismatch_weight,
            batch_size=args.batch_size,
        )
        observed_pairs += mine_observed_adjacencies(assignments, observed_counts)
        total_tile_counts.update(tile_counts)
        per_source_stats.append(
            {
                "source": item["source"],
                "shape_yzx": item["shape_yzx"],
                "semantic_bucket_counts": dict(Counter(buckets).most_common()),
                "filter_stats": filter_stats,
                "assignment_stats": assignment_stats,
                "assigned_tile_counts": {str(tile): int(count) for tile, count in tile_counts.most_common()},
            }
        )

    overlap_mismatches = overlap_mismatch_table(prototypes)
    observed_allowed = observed_counts >= args.min_observed_count
    overlap_allowed = overlap_mismatches <= args.max_overlap_mismatch
    allowed = observed_allowed & overlap_allowed

    for direction, _delta in DIRECTIONS:
        direction_index = DIRECTION_INDEX[direction]
        opposite_index = DIRECTION_INDEX[OPPOSITE_DIRECTION[direction]]
        allowed[opposite_index] |= allowed[direction_index].T
        observed_counts[opposite_index] = np.maximum(observed_counts[opposite_index], observed_counts[direction_index].T)

    active_tiles = np.asarray([total_tile_counts[tile_id] > 0 for tile_id in range(tile_count)], dtype=bool)
    prune_stats: dict[str, Any] = {"enabled": bool(args.prune_dead_ends), "iterations": 0, "removed_tile_ids": []}
    if args.prune_dead_ends:
        active_tiles, prune_stats = prune_dead_end_tiles(allowed, active_tiles)
        for tile_id in np.flatnonzero(~active_tiles):
            allowed[:, tile_id, :] = False
            allowed[:, :, tile_id] = False

    phase2_summaries = evaluation.get("cluster_summaries", [])
    phase2_sizes = [int(summary.get("size", 1)) for summary in phase2_summaries]
    if len(phase2_sizes) != tile_count:
        phase2_sizes = [max(int(total_tile_counts[tile_id]), 1) for tile_id in range(tile_count)]

    weights = np.asarray(
        [
            max(int(total_tile_counts[tile_id]), phase2_sizes[tile_id], 1)
            if active_tiles[tile_id]
            else 0
            for tile_id in range(tile_count)
        ],
        dtype=np.float32,
    )

    face_info = face_summaries(prototypes, palette, air_id)
    catalog = []
    for tile_id, face_summary in enumerate(face_info):
        dead_end_faces = [
            direction
            for direction, _delta in DIRECTIONS
            if active_tiles[tile_id] and not np.any(allowed[DIRECTION_INDEX[direction], tile_id, active_tiles])
        ]
        phase2_summary = phase2_summaries[tile_id] if tile_id < len(phase2_summaries) else {}
        catalog.append(
            {
                **face_summary,
                "active": bool(active_tiles[tile_id]),
                "weight": float(weights[tile_id]),
                "phase2_cluster_id": phase2_summary.get("cluster_id"),
                "phase2_cluster_size": phase2_sizes[tile_id],
                "phase3_assignment_count": int(total_tile_counts[tile_id]),
                "dead_end_faces": dead_end_faces,
                "allowed_neighbor_counts": {
                    direction: int(np.count_nonzero(allowed[DIRECTION_INDEX[direction], tile_id] & active_tiles))
                    for direction, _delta in DIRECTIONS
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "ruleset.npz",
        prototypes=prototypes.astype(np.uint16),
        allowed=allowed.astype(bool),
        observed_counts=observed_counts.astype(np.int32),
        overlap_mismatches=overlap_mismatches.astype(np.float32),
        weights=weights.astype(np.float32),
        active_tiles=active_tiles.astype(bool),
    )
    write_json(
        args.output_dir / "ruleset.json",
        {
            "direction_order": [direction for direction, _delta in DIRECTIONS],
            "opposite_direction": OPPOSITE_DIRECTION,
            "tile_count": tile_count,
            "active_tile_count": int(np.count_nonzero(active_tiles)),
            "window": window,
            "max_assignment_distance": args.max_assignment_distance,
            "max_overlap_mismatch": args.max_overlap_mismatch,
            "min_observed_count": args.min_observed_count,
            "observed_adjacent_pairs": int(observed_pairs),
            "allowed_adjacency_count": int(np.count_nonzero(allowed)),
            "rules": rule_lists(allowed),
            "observed_counts": observed_count_lists(observed_counts),
        },
    )
    write_json(args.output_dir / "tile_catalog.json", catalog)
    write_json(
        args.output_dir / "validation.json",
        {
            "per_source": per_source_stats,
            "assignment_tile_counts": {str(tile): int(count) for tile, count in total_tile_counts.most_common()},
            "prune_dead_ends": prune_stats,
            "dead_end_tiles": [
                {
                    "tile_id": item["tile_id"],
                    "dead_end_faces": item["dead_end_faces"],
                }
                for item in catalog
                if item["dead_end_faces"]
            ],
        },
    )
    write_tile_catalog_csv(args.output_dir / "tile_catalog.csv", catalog)

    dead_end_tiles = [item for item in catalog if item["dead_end_faces"]]
    lines = [
        "# Phase III Ruleset",
        "",
        f"Mined observed one-voxel adjacencies from {len(arrays)} original, non-augmented Phase I arrays.",
        f"Assigned candidate patches to {tile_count} Phase II medoid tiles with max assignment distance {args.max_assignment_distance:.2f}.",
        f"Observed adjacent patch pairs: {observed_pairs}. Allowed directional tile pairs after overlap validation: {int(np.count_nonzero(allowed))}.",
        f"Active tiles with nonzero observed assignments: {int(np.count_nonzero(active_tiles))}/{tile_count}.",
        "",
        "## Constraint Policy",
        "",
        f"- A pair is allowed only when it was observed at least {args.min_observed_count} time(s) in the original houses.",
        f"- The corresponding medoid prototypes must also agree across the shifted overlap with mismatch fraction <= {args.max_overlap_mismatch:.2f}.",
        "- Opposite-direction rules are mirrored after mining so the ruleset is symmetric for propagation.",
        f"- Dead-end pruning enabled: {args.prune_dead_ends}.",
        "",
        "## Source Assignment Diagnostics",
        "",
        "| Source | Candidates | Assigned | Assignment rate | Mean distance | p90 distance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in per_source_stats:
        stats = item["assignment_stats"]
        lines.append(
            f"| {item['source']} | {stats['candidate_patches']} | {stats['assigned_patches']} | "
            f"{stats['assignment_rate']:.1%} | {stats['distance_mean']:.3f} | {stats['distance_p90']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Dead-End Validation",
            "",
            f"Tiles with at least one empty critical face: {len(dead_end_tiles)}.",
        ]
    )
    if dead_end_tiles:
        for item in dead_end_tiles[:30]:
            lines.append(f"- tile {item['tile_id']}: {', '.join(item['dead_end_faces'])}")
        if len(dead_end_tiles) > 30:
            lines.append(f"- ... {len(dead_end_tiles) - 30} more")
    else:
        lines.append("- No active tile has an empty directional neighbor list.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Binary ruleset: `{args.output_dir / 'ruleset.npz'}`",
            f"- JSON adjacency lists: `{args.output_dir / 'ruleset.json'}`",
            f"- Tile catalog: `{args.output_dir / 'tile_catalog.json'}` and `{args.output_dir / 'tile_catalog.csv'}`",
        ]
    )
    args.docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote WFC ruleset to {args.output_dir / 'ruleset.npz'}")
    print(f"Wrote ruleset diagnostics to {args.output_dir / 'validation.json'}")
    print(f"Wrote observations to {args.docs_path}")


if __name__ == "__main__":
    main()
