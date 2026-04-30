#!/usr/bin/env python3
"""Phase IV: generate Minecraft structures from learned 3D WFC rules."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wavecraft.export_tile_schematic import (
    AIR_STATE,
    add_state,
    category_to_block_state,
    parse_category,
    write_sponge_schem,
)
from wavecraft.phase3_build_rules import DIRECTIONS, DIRECTION_INDEX, prune_dead_end_tiles, write_json
from wavecraft.phase2_extract_tiles import candidate_positions, category_kind
from wavecraft.phase3_build_rules import assign_patches_to_tiles, load_phase1_originals


LANTERN_STATE = "minecraft:lantern[hanging=true,waterlogged=false]"


@dataclass
class SolveStats:
    attempt: int
    decisions: int
    backtracks: int
    propagations: int
    max_depth: int


class WFCSolver:
    def __init__(
        self,
        allowed: np.ndarray,
        active_tiles: np.ndarray,
        weights: np.ndarray,
        cell_weights: np.ndarray | None,
        grid_shape: tuple[int, int, int],
        rng: np.random.Generator,
        max_backtracks: int,
    ) -> None:
        self.allowed = allowed
        self.active_tiles = active_tiles
        self.weights = weights.astype(np.float64)
        self.cell_weights = cell_weights.astype(np.float64) if cell_weights is not None else None
        self.grid_shape = grid_shape
        self.rng = rng
        self.max_backtracks = max_backtracks
        self.decisions = 0
        self.backtracks = 0
        self.propagations = 0
        self.max_depth = 0

    def initial_domains(self) -> np.ndarray:
        domains = np.zeros((*self.grid_shape, self.active_tiles.shape[0]), dtype=bool)
        domains[..., self.active_tiles] = True
        if self.cell_weights is not None:
            domains &= self.cell_weights > 0.0
        return domains

    def propagate(self, domains: np.ndarray, queue: deque[tuple[int, int, int]]) -> bool:
        y_size, z_size, x_size = self.grid_shape
        while queue:
            y, z, x = queue.popleft()
            current = domains[y, z, x]
            if not np.any(current):
                return False
            for direction, delta in DIRECTIONS:
                ny = y + int(delta[0])
                nz = z + int(delta[1])
                nx = x + int(delta[2])
                if ny < 0 or nz < 0 or nx < 0 or ny >= y_size or nz >= z_size or nx >= x_size:
                    continue
                direction_index = DIRECTION_INDEX[direction]
                supported = np.any(self.allowed[direction_index, current, :], axis=0)
                new_domain = domains[ny, nz, nx] & supported
                if not np.any(new_domain):
                    return False
                if not np.array_equal(new_domain, domains[ny, nz, nx]):
                    domains[ny, nz, nx] = new_domain
                    queue.append((ny, nz, nx))
                    self.propagations += 1
        return True

    def choose_cell(self, domains: np.ndarray) -> tuple[int, int, int] | None:
        counts = domains.sum(axis=-1)
        unresolved = counts > 1
        if not np.any(unresolved):
            return None

        entropies = np.full(counts.shape, np.inf, dtype=np.float64)
        for index in np.argwhere(unresolved):
            y, z, x = (int(value) for value in index)
            domain = domains[y, z, x]
            weights = self.cell_weights[y, z, x] if self.cell_weights is not None else self.weights
            domain_weights = weights[domain]
            total = float(domain_weights.sum())
            if total <= 0.0:
                entropies[y, z, x] = -math.inf
                continue
            entropies[y, z, x] = math.log(total) - float((domain_weights * np.log(domain_weights)).sum() / total)
        minimum = float(np.min(entropies[unresolved]))
        candidate_positions = np.argwhere(np.isclose(entropies, minimum))
        choice = candidate_positions[int(self.rng.integers(0, len(candidate_positions)))]
        return int(choice[0]), int(choice[1]), int(choice[2])

    def ordered_tiles(self, domain: np.ndarray, position: tuple[int, int, int]) -> list[int]:
        candidates = np.flatnonzero(domain)
        y, z, x = position
        weights = self.cell_weights[y, z, x] if self.cell_weights is not None else self.weights
        candidate_weights = weights[candidates].astype(np.float64)
        if candidate_weights.sum() <= 0:
            probabilities = None
        else:
            probabilities = candidate_weights / candidate_weights.sum()
        return [int(value) for value in self.rng.choice(candidates, size=len(candidates), replace=False, p=probabilities)]

    def solve_recursive(self, domains: np.ndarray, depth: int = 0) -> np.ndarray | None:
        if self.backtracks > self.max_backtracks:
            return None
        self.max_depth = max(self.max_depth, depth)
        position = self.choose_cell(domains)
        if position is None:
            return domains

        y, z, x = position
        for tile_id in self.ordered_tiles(domains[y, z, x], position):
            next_domains = domains.copy()
            next_domains[y, z, x] = False
            next_domains[y, z, x, tile_id] = True
            self.decisions += 1
            if self.propagate(next_domains, deque([(y, z, x)])):
                solved = self.solve_recursive(next_domains, depth + 1)
                if solved is not None:
                    return solved
            self.backtracks += 1
            if self.backtracks > self.max_backtracks:
                return None
        return None

    def solve(self) -> np.ndarray | None:
        domains = self.initial_domains()
        if not self.propagate(domains, deque()):
            return None
        return self.solve_recursive(domains)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_grid_size(value: str) -> tuple[int, int, int]:
    pieces = [piece.strip() for piece in value.split(",")]
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("Grid size must be Y,Z,X, for example 12,14,12")
    y, z, x = (int(piece) for piece in pieces)
    if min(y, z, x) <= 0:
        raise argparse.ArgumentTypeError("Grid dimensions must be positive")
    return y, z, x


def shifted_overlap_collision_fraction(
    prototype_a: np.ndarray,
    prototype_b: np.ndarray,
    direction: str,
    stride: int,
    air_id: int,
) -> float:
    if stride >= prototype_a.shape[0]:
        return 0.0
    if stride <= 0:
        raise ValueError("stride must be positive")

    if direction == "+y":
        left = prototype_a[stride:, :, :]
        right = prototype_b[:-stride, :, :]
    elif direction == "-y":
        left = prototype_a[:-stride, :, :]
        right = prototype_b[stride:, :, :]
    elif direction == "+z":
        left = prototype_a[:, stride:, :]
        right = prototype_b[:, :-stride, :]
    elif direction == "-z":
        left = prototype_a[:, :-stride, :]
        right = prototype_b[:, stride:, :]
    elif direction == "+x":
        left = prototype_a[:, :, stride:]
        right = prototype_b[:, :, :-stride]
    elif direction == "-x":
        left = prototype_a[:, :, :-stride]
        right = prototype_b[:, :, stride:]
    else:
        raise ValueError(direction)

    if left.size == 0:
        return 0.0
    collisions = (left != air_id) & (right != air_id) & (left != right)
    return float(np.count_nonzero(collisions) / left.size)


def stride_overlap_collision_table(prototypes: np.ndarray, stride: int, air_id: int) -> np.ndarray:
    table = np.zeros((len(DIRECTIONS), prototypes.shape[0], prototypes.shape[0]), dtype=np.float32)
    for direction_index, (direction, _delta) in enumerate(DIRECTIONS):
        axis = int(np.argmax(np.abs(_delta)))
        axis_window = int(prototypes.shape[1 + axis])
        if stride >= axis_window:
            continue
        for tile_a in range(prototypes.shape[0]):
            for tile_b in range(prototypes.shape[0]):
                table[direction_index, tile_a, tile_b] = shifted_overlap_collision_fraction(
                    prototypes[tile_a],
                    prototypes[tile_b],
                    direction,
                    stride,
                    air_id,
                )
    return table


def stride_allowed_rules(
    prototypes: np.ndarray,
    stride: int,
    air_id: int,
    max_collision_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    collision_table = stride_overlap_collision_table(prototypes, stride, air_id)
    allowed = collision_table <= np.float32(max_collision_fraction)
    stats = {
        "tile_stride": int(stride),
        "max_stride_overlap_collision_fraction": float(max_collision_fraction),
        "stride_allowed_adjacency_count": int(np.count_nonzero(allowed)),
        "stride_collision_nonzero_count": int(np.count_nonzero(collision_table)),
    }
    return allowed, collision_table, stats


def prepare_rules(
    allowed: np.ndarray,
    active_tiles: np.ndarray,
    weights: np.ndarray,
    avoid_dead_end_tiles: bool,
    repair_dead_ends: bool,
    overlap_mismatches: np.ndarray,
    max_repair_overlap_collision_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    prepared_allowed = allowed.copy()
    prepared_active = active_tiles.copy() & (weights > 0)
    repair_count = 0

    if repair_dead_ends:
        for tile_id in np.flatnonzero(prepared_active):
            for direction_index in range(len(DIRECTIONS)):
                if np.any(prepared_allowed[direction_index, tile_id] & prepared_active):
                    continue
                fallback = (overlap_mismatches[direction_index, tile_id] <= max_repair_overlap_collision_fraction) & prepared_active
                if np.any(fallback):
                    prepared_allowed[direction_index, tile_id, fallback] = True
                    repair_count += int(np.count_nonzero(fallback))

    prune_stats: dict[str, Any] = {"enabled": bool(avoid_dead_end_tiles), "iterations": 0, "removed_tile_ids": []}
    if avoid_dead_end_tiles:
        prepared_active, prune_stats = prune_dead_end_tiles(prepared_allowed, prepared_active)

    prepared_allowed[:, ~prepared_active, :] = False
    prepared_allowed[:, :, ~prepared_active] = False
    prepared_weights = np.where(prepared_active, np.maximum(weights, 1.0), 0.0).astype(np.float64)

    if not np.any(prepared_active):
        raise SystemExit("No active tiles remain after rule preparation. Try --allow-dead-end-tiles or --repair-dead-ends.")
    if np.count_nonzero(prepared_active) < 2:
        raise SystemExit("Need at least two active tiles for useful generation")

    stats = {
        "active_tile_count": int(np.count_nonzero(prepared_active)),
        "allowed_adjacency_count": int(np.count_nonzero(prepared_allowed)),
        "avoid_dead_end_tiles": bool(avoid_dead_end_tiles),
        "repair_dead_ends": bool(repair_dead_ends),
        "repair_added_adjacencies": int(repair_count),
        "dead_end_pruning": prune_stats,
    }
    return prepared_allowed, prepared_active, prepared_weights, stats


ROLE_NAMES = ("foundation", "exterior_wall", "opening", "roof", "interior")
ROLE_INDEX = {name: index for index, name in enumerate(ROLE_NAMES)}


def category_flags(category: str) -> dict[str, bool]:
    family, kind, _props = parse_category(category)
    return {
        "air": family == "air",
        "roof": family == "dark_stone" or kind == "tiles",
        "opening": family == "glass" or kind in {"pane", "door", "fence", "fence_gate", "trapdoor"} or family == "ladder",
        "bulk": kind in {"full", "bricks", "planks", "log", "wood"},
    }


def tile_content_scores(prototypes: np.ndarray, palette: list[str], air_id: int) -> dict[str, np.ndarray]:
    roof_ids = np.asarray([category_flags(category)["roof"] for category in palette], dtype=bool)
    opening_ids = np.asarray([category_flags(category)["opening"] for category in palette], dtype=bool)
    bulk_ids = np.asarray([category_flags(category)["bulk"] for category in palette], dtype=bool)
    non_air = prototypes != air_id
    non_air_count = np.count_nonzero(non_air, axis=(1, 2, 3)).astype(np.float32)
    normalizer = np.maximum(non_air_count, 1.0)
    return {
        "density": non_air_count / float(prototypes.shape[1] * prototypes.shape[2] * prototypes.shape[3]),
        "roof": np.count_nonzero(roof_ids[prototypes] & non_air, axis=(1, 2, 3)).astype(np.float32) / normalizer,
        "opening": np.count_nonzero(opening_ids[prototypes] & non_air, axis=(1, 2, 3)).astype(np.float32) / normalizer,
        "bulk": np.count_nonzero(bulk_ids[prototypes] & non_air, axis=(1, 2, 3)).astype(np.float32) / normalizer,
    }


def volume_bounds(volume: np.ndarray, air_id: int) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.argwhere(volume != air_id)
    if occupied.size == 0:
        zeros = np.zeros(3, dtype=np.int32)
        return zeros, zeros
    return occupied.min(axis=0), occupied.max(axis=0)


def classify_training_patch(
    patch: np.ndarray,
    position: tuple[int, int, int],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    palette: list[str],
    air_id: int,
) -> tuple[str, float]:
    y, z, x = position
    window = patch.shape[0]
    center = np.asarray((y, z, x), dtype=np.float32) + ((window - 1) / 2.0)
    span = np.maximum(bounds_max - bounds_min, 1).astype(np.float32)
    y_norm = float((center[0] - bounds_min[0]) / span[0])
    edge_distance = min(
        abs(float(center[1] - bounds_min[1])),
        abs(float(bounds_max[1] - center[1])),
        abs(float(center[2] - bounds_min[2])),
        abs(float(bounds_max[2] - center[2])),
    )
    is_exterior = edge_distance <= max(2.0, window / 2.0)
    ids = np.asarray(patch, dtype=np.int32)
    non_air = ids != air_id
    non_air_count = max(int(np.count_nonzero(non_air)), 1)
    roof_count = sum(
        1
        for value in ids[non_air].ravel()
        if category_flags(palette[int(value)])["roof"]
    )
    opening_count = sum(
        1
        for value in ids[non_air].ravel()
        if category_flags(palette[int(value)])["opening"]
    )
    roof_fraction = roof_count / non_air_count
    opening_fraction = opening_count / non_air_count
    density = non_air_count / patch.size

    if roof_fraction >= 0.18 or (y_norm >= 0.62 and roof_count >= 4):
        return "roof", y_norm
    if opening_fraction >= 0.04 and is_exterior:
        return "opening", y_norm
    if y_norm <= 0.18 and density >= 0.25:
        return "foundation", y_norm
    if is_exterior and density >= 0.18:
        return "exterior_wall", y_norm
    return "interior", y_norm


def mine_tile_role_affinities(
    phase1_dir: Path,
    phase2_dir: Path,
    prototypes: np.ndarray,
    palette: list[str],
    air_id: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    _metadata, arrays = load_phase1_originals(phase1_dir / "metadata.json")
    evaluation = load_json(phase2_dir / "evaluation.json")
    window = int(evaluation["window"])
    kind_by_id = [category_kind(category) for category in palette]
    role_counts = np.zeros((prototypes.shape[0], len(ROLE_NAMES)), dtype=np.float32)
    role_y_values: dict[str, list[float]] = {role: [] for role in ROLE_NAMES}
    assigned_total = 0

    for _item, volume in arrays:
        positions, _buckets, _filter_stats = candidate_positions(
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
        assignments, _assignment_stats, _tile_counts = assign_patches_to_tiles(
            volume=volume,
            positions=positions,
            prototypes=prototypes,
            air_id=air_id,
            max_assignment_distance=0.52,
            air_mismatch_weight=2.0,
            category_mismatch_weight=1.0,
            batch_size=batch_size,
        )
        bounds_min, bounds_max = volume_bounds(volume, air_id)
        for position, tile_id in assignments.items():
            y, z, x = position
            patch = volume[y : y + window, z : z + window, x : x + window]
            role, y_norm = classify_training_patch(patch, position, bounds_min, bounds_max, palette, air_id)
            role_counts[tile_id, ROLE_INDEX[role]] += 1.0
            role_y_values[role].append(y_norm)
            assigned_total += 1

    role_affinities = (role_counts + 1.0) / np.maximum(role_counts.sum(axis=1, keepdims=True) + len(ROLE_NAMES), 1.0)
    roof_values = role_y_values["roof"]
    foundation_values = role_y_values["foundation"]
    stats = {
        "enabled": True,
        "assigned_training_patches": int(assigned_total),
        "role_counts": {role: int(role_counts[:, ROLE_INDEX[role]].sum()) for role in ROLE_NAMES},
        "roof_start_y_norm": float(np.quantile(roof_values, 0.25)) if roof_values else 0.62,
        "foundation_end_y_norm": float(np.quantile(foundation_values, 0.75)) if foundation_values else 0.16,
        "top_tiles_by_role": {},
    }
    for role in ROLE_NAMES:
        index = ROLE_INDEX[role]
        top_tiles = np.argsort(role_affinities[:, index])[-8:][::-1]
        stats["top_tiles_by_role"][role] = [
            {"tile_id": int(tile_id), "affinity": float(role_affinities[tile_id, index]), "observations": int(role_counts[tile_id, index])}
            for tile_id in top_tiles
        ]
    return role_affinities.astype(np.float32), stats


def target_role_for_cell(
    y: int,
    z: int,
    x: int,
    grid_shape: tuple[int, int, int],
    roof_start_y_norm: float,
    foundation_end_y_norm: float,
    opening_rate: float,
    seed: int,
) -> str:
    y_size, z_size, x_size = grid_shape
    y_norm = (y + 0.5) / y_size
    is_perimeter = z == 0 or x == 0 or z == z_size - 1 or x == x_size - 1
    if y_norm >= roof_start_y_norm:
        return "roof"
    if y_norm <= foundation_end_y_norm:
        return "foundation"
    if is_perimeter:
        if 0.24 <= y_norm <= min(0.68, roof_start_y_norm - 0.05) and stable_fraction(seed, y, z, x, 71) <= opening_rate:
            return "opening"
        return "exterior_wall"
    return "interior"


def build_scaffold_cell_weights(
    base_weights: np.ndarray,
    active_tiles: np.ndarray,
    prototypes: np.ndarray,
    palette: list[str],
    air_id: int,
    grid_shape: tuple[int, int, int],
    role_affinities: np.ndarray,
    scaffold_stats: dict[str, Any],
    seed: int,
    roof_boost: float,
    wall_boost: float,
    opening_boost: float,
    foundation_boost: float,
    interior_boost: float,
    opening_rate: float,
    roof_min_score: float,
    roof_content_min: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    content = tile_content_scores(prototypes, palette, air_id)
    y_size, z_size, x_size = grid_shape
    cell_weights = np.zeros((*grid_shape, base_weights.shape[0]), dtype=np.float64)
    role_grid_counts: Counter[str] = Counter()
    role_boosts = {
        "foundation": foundation_boost,
        "exterior_wall": wall_boost,
        "opening": opening_boost,
        "roof": roof_boost,
        "interior": interior_boost,
    }
    role_content = {
        "foundation": content["density"],
        "exterior_wall": np.maximum(content["bulk"], content["opening"] * 0.5),
        "opening": content["opening"],
        "roof": content["roof"],
        "interior": np.maximum(1.0 - content["roof"] - content["opening"], 0.0),
    }
    roof_start = min(max(float(scaffold_stats["roof_start_y_norm"]), 0.82), 0.90)
    foundation_end = min(max(float(scaffold_stats["foundation_end_y_norm"]), 0.08), 0.24)

    for y in range(y_size):
        for z in range(z_size):
            for x in range(x_size):
                role = target_role_for_cell(y, z, x, grid_shape, roof_start, foundation_end, opening_rate, seed)
                role_grid_counts[role] += 1
                role_index = ROLE_INDEX[role]
                inferred = role_affinities[:, role_index]
                content_score = role_content[role]
                role_score = np.clip((0.7 * inferred) + (0.3 * content_score), 0.0, 1.0)
                factor = 0.05 + (role_boosts[role] * role_score)
                local_weights = base_weights * factor
                if role == "roof":
                    local_weights[role_score < roof_min_score] = 0.0
                    local_weights[content["roof"] < roof_content_min] = 0.0
                local_weights[~active_tiles] = 0.0
                cell_weights[y, z, x] = np.maximum(local_weights, 0.0)

    stats = {
        **scaffold_stats,
        "role_grid_counts": dict(role_grid_counts),
        "role_boosts": {role: float(value) for role, value in role_boosts.items()},
        "opening_rate": float(opening_rate),
        "roof_min_score": float(roof_min_score),
        "roof_content_min": float(roof_content_min),
        "roof_start_y_norm_used": float(roof_start),
        "foundation_end_y_norm_used": float(foundation_end),
        "min_cell_domain_weight_sum": float(cell_weights.sum(axis=-1).min()),
    }
    return cell_weights, stats


def run_solver(
    allowed: np.ndarray,
    active_tiles: np.ndarray,
    weights: np.ndarray,
    cell_weights: np.ndarray | None,
    grid_shape: tuple[int, int, int],
    seed: int,
    retries: int,
    max_backtracks: int,
) -> tuple[np.ndarray, SolveStats]:
    sys.setrecursionlimit(max(10000, int(np.prod(grid_shape)) + 1000))
    last_stats: SolveStats | None = None
    for attempt in range(retries + 1):
        rng = np.random.default_rng(seed + attempt)
        solver = WFCSolver(
            allowed=allowed,
            active_tiles=active_tiles,
            weights=weights,
            cell_weights=cell_weights,
            grid_shape=grid_shape,
            rng=rng,
            max_backtracks=max_backtracks,
        )
        solved_domains = solver.solve()
        stats = SolveStats(
            attempt=attempt,
            decisions=solver.decisions,
            backtracks=solver.backtracks,
            propagations=solver.propagations,
            max_depth=solver.max_depth,
        )
        last_stats = stats
        if solved_domains is not None:
            return np.argmax(solved_domains, axis=-1).astype(np.int16), stats
    assert last_stats is not None
    raise SystemExit(
        "WFC failed to find a solution after "
        f"{retries + 1} attempt(s); last attempt used {last_stats.backtracks} backtracks. "
        "Try a smaller grid, higher --max-backtracks, or --repair-dead-ends."
    )


def reconstruction_category_weights(palette_categories: list[str], air_id: int) -> np.ndarray:
    weights = np.ones(len(palette_categories), dtype=np.float32)
    weights[air_id] = 0.0
    for category_id, category in enumerate(palette_categories):
        family, kind, _props = parse_category(category)
        if family == "air":
            weights[category_id] = 0.0
        elif family == "glass" or kind == "pane":
            weights[category_id] = 12.0
        elif kind == "door":
            weights[category_id] = 8.0
        elif kind in {"fence", "fence_gate", "trapdoor"} or family == "ladder":
            weights[category_id] = 6.0
        elif kind in {"stairs", "slab", "wall"}:
            weights[category_id] = 3.0
    return weights


def reconstruct_overlapping_volume(
    chosen_tiles: np.ndarray,
    prototypes: np.ndarray,
    palette_categories: list[str],
    air_id: int,
    non_air_threshold: float,
    tile_stride: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    palette_size = len(palette_categories)
    grid_y, grid_z, grid_x = chosen_tiles.shape
    window_y, window_z, window_x = prototypes.shape[1:]
    output_shape = (
        ((grid_y - 1) * tile_stride) + window_y,
        ((grid_z - 1) * tile_stride) + window_z,
        ((grid_x - 1) * tile_stride) + window_x,
    )
    votes = np.zeros((*output_shape, palette_size), dtype=np.uint16)

    for y in range(grid_y):
        for z in range(grid_z):
            for x in range(grid_x):
                out_y = y * tile_stride
                out_z = z * tile_stride
                out_x = x * tile_stride
                patch = prototypes[int(chosen_tiles[y, z, x])]
                for py in range(window_y):
                    for pz in range(window_z):
                        for px in range(window_x):
                            votes[out_y + py, out_z + pz, out_x + px, int(patch[py, pz, px])] += 1

    raw_majority = np.argmax(votes, axis=-1).astype(np.uint16)
    non_air_categories = np.ones(palette_size, dtype=bool)
    non_air_categories[air_id] = False

    total_votes = votes.sum(axis=-1, dtype=np.uint16)
    non_air_votes = votes[..., non_air_categories].sum(axis=-1, dtype=np.uint16)
    non_air_probability = np.divide(
        non_air_votes,
        total_votes,
        out=np.zeros(output_shape, dtype=np.float32),
        where=total_votes > 0,
    )
    category_weights = reconstruction_category_weights(palette_categories, air_id)
    weighted_non_air_scores = votes.astype(np.float32) * category_weights
    most_probable_non_air = np.argmax(weighted_non_air_scores, axis=-1).astype(np.uint16)
    union_poll = np.full(output_shape, air_id, dtype=np.uint16)
    accepted_non_air = (non_air_votes > 0) & (non_air_probability > np.float32(non_air_threshold))
    union_poll[accepted_non_air] = most_probable_non_air[accepted_non_air]
    detail_categories = category_weights > 1.0

    stats = {
        "non_air_threshold": float(non_air_threshold),
        "tile_stride": int(tile_stride),
        "detail_category_weights": {
            "glass_or_pane": 12.0,
            "door": 8.0,
            "fence_gate_trapdoor_ladder": 6.0,
            "stairs_slab_wall": 3.0,
            "structural": 1.0,
        },
        "raw_majority_non_air_voxels": int(np.count_nonzero(raw_majority != air_id)),
        "raw_majority_air_fraction": float(np.count_nonzero(raw_majority == air_id) / raw_majority.size),
        "union_poll_non_air_voxels": int(np.count_nonzero(union_poll != air_id)),
        "union_poll_air_fraction": float(np.count_nonzero(union_poll == air_id) / union_poll.size),
        "union_poll_detail_voxels": int(np.count_nonzero(detail_categories[union_poll])),
        "mean_non_air_probability": float(non_air_probability.mean()),
    }
    return union_poll, stats


def stable_fraction(seed: int, y: int, z: int, x: int, salt: int) -> float:
    value = (
        (seed + 0x9E3779B9)
        ^ (y * 0x85EBCA6B)
        ^ (z * 0xC2B2AE35)
        ^ (x * 0x27D4EB2F)
        ^ (salt * 0x165667B1)
    ) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    return value / 0xFFFFFFFF


def varied_block_state(category: str, y: int, z: int, x: int, seed: int, variation_rate: float) -> str:
    if category == "air":
        return AIR_STATE
    base_state = category_to_block_state(category)
    family, kind, _props = parse_category(category)
    if stable_fraction(seed, y, z, x, 11) > variation_rate:
        return base_state

    if family == "stone" and kind in {"full", "bricks"}:
        return "minecraft:mossy_stone_bricks" if stable_fraction(seed, y, z, x, 12) < 0.5 else "minecraft:cracked_stone_bricks"
    if family == "dark_stone" and kind in {"full", "bricks", "tiles"}:
        return "minecraft:cracked_deepslate_bricks"
    if family == "masonry" and kind in {"full", "bricks"}:
        return "minecraft:mud_bricks" if stable_fraction(seed, y, z, x, 13) < 0.35 else base_state
    if family.startswith("wood_") and kind in {"full", "planks"}:
        return base_state
    return base_state


def volume_to_block_ids(
    categories: np.ndarray,
    palette_categories: list[str],
    seed: int,
    variation_rate: float,
    ornament_rate: float,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    palette: dict[str, int] = {AIR_STATE: 0}
    blocks = np.zeros(categories.shape, dtype=np.int32)
    for y in range(categories.shape[0]):
        for z in range(categories.shape[1]):
            for x in range(categories.shape[2]):
                category = palette_categories[int(categories[y, z, x])]
                state = varied_block_state(category, y, z, x, seed, variation_rate)
                blocks[y, z, x] = add_state(palette, state)

    air_id = palette[AIR_STATE]
    lantern_id = add_state(palette, LANTERN_STATE)
    ornament_count = 0
    lantern_candidates: list[tuple[int, int, int]] = []
    if ornament_rate > 0:
        for y in range(1, categories.shape[0] - 1):
            for z in range(1, categories.shape[1] - 1):
                for x in range(1, categories.shape[2] - 1):
                    if blocks[y, z, x] != air_id:
                        continue
                    if blocks[y + 1, z, x] == air_id or blocks[y - 1, z, x] != air_id:
                        continue
                    lantern_candidates.append((y, z, x))
                    if stable_fraction(seed, y, z, x, 31) <= ornament_rate:
                        blocks[y, z, x] = lantern_id
                        ornament_count += 1
        if ornament_count == 0 and lantern_candidates:
            y, z, x = lantern_candidates[int(stable_fraction(seed, 0, 0, 0, 37) * len(lantern_candidates)) % len(lantern_candidates)]
            blocks[y, z, x] = lantern_id
            ornament_count = 1

    state_counts = Counter()
    inverse_palette = {index: state for state, index in palette.items()}
    for value, count in zip(*np.unique(blocks, return_counts=True), strict=True):
        state_counts[inverse_palette[int(value)]] = int(count)
    return blocks, palette, {
        "lantern_candidates": len(lantern_candidates),
        "lanterns_added": ornament_count,
        "state_counts": dict(state_counts.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", type=Path, default=Path("datasets/phase1"))
    parser.add_argument("--phase2-dir", type=Path, default=Path("datasets/phase2"))
    parser.add_argument("--phase3-dir", type=Path, default=Path("datasets/phase3"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/phase4"))
    parser.add_argument("--output", type=Path, default=Path("datasets/phase4/generated_house.schem"))
    parser.add_argument("--docs-path", type=Path, default=Path("docs/phase4_generation.md"))
    parser.add_argument("--grid-size", type=parse_grid_size, default=parse_grid_size("10,12,10"), help="Cell grid as Y,Z,X")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--max-backtracks", type=int, default=20000)
    parser.add_argument("--allow-dead-end-tiles", action="store_true")
    parser.add_argument("--repair-dead-ends", action="store_true")
    parser.add_argument("--max-repair-overlap-collision-fraction", type=float, default=0.0)
    parser.add_argument("--tile-stride", type=int, default=4)
    parser.add_argument("--max-stride-overlap-collision-fraction", type=float, default=0.0)
    parser.add_argument("--scaffold", choices=["none", "inferred"], default="inferred")
    parser.add_argument("--scaffold-batch-size", type=int, default=2048)
    parser.add_argument("--roof-boost", type=float, default=24.0)
    parser.add_argument("--wall-boost", type=float, default=7.0)
    parser.add_argument("--opening-boost", type=float, default=9.0)
    parser.add_argument("--foundation-boost", type=float, default=5.0)
    parser.add_argument("--interior-boost", type=float, default=2.0)
    parser.add_argument("--opening-rate", type=float, default=0.20)
    parser.add_argument("--roof-min-score", type=float, default=0.18)
    parser.add_argument("--roof-content-min", type=float, default=0.04)
    parser.add_argument("--non-air-threshold", type=float, default=0.5)
    parser.add_argument("--variation-rate", type=float, default=0.08)
    parser.add_argument("--ornament-rate", type=float, default=0.004)
    parser.add_argument("--data-version", type=int, default=4556)
    args = parser.parse_args()

    if args.non_air_threshold < 0.0 or args.non_air_threshold > 1.0:
        raise SystemExit("--non-air-threshold must be between 0 and 1")
    if args.opening_rate < 0.0 or args.opening_rate > 1.0:
        raise SystemExit("--opening-rate must be between 0 and 1")
    if args.tile_stride <= 0:
        raise SystemExit("--tile-stride must be positive")
    if args.max_stride_overlap_collision_fraction < 0.0 or args.max_stride_overlap_collision_fraction > 1.0:
        raise SystemExit("--max-stride-overlap-collision-fraction must be between 0 and 1")

    metadata = load_json(args.phase1_dir / "metadata.json")
    rules = np.load(args.phase3_dir / "ruleset.npz")
    prototypes = rules["prototypes"]
    stride_allowed, stride_overlap_mismatches, stride_rule_stats = stride_allowed_rules(
        prototypes=prototypes,
        stride=args.tile_stride,
        air_id=int(metadata["air_id"]),
        max_collision_fraction=args.max_stride_overlap_collision_fraction,
    )
    allowed, active_tiles, weights, rule_stats = prepare_rules(
        allowed=stride_allowed,
        active_tiles=rules["active_tiles"],
        weights=rules["weights"],
        avoid_dead_end_tiles=not args.allow_dead_end_tiles,
        repair_dead_ends=args.repair_dead_ends,
        overlap_mismatches=stride_overlap_mismatches,
        max_repair_overlap_collision_fraction=args.max_repair_overlap_collision_fraction,
    )
    rule_stats["source_ruleset_allowed_adjacency_count"] = int(np.count_nonzero(rules["allowed"]))
    rule_stats["stride_rules"] = stride_rule_stats
    cell_weights = None
    scaffold_stats: dict[str, Any] = {"enabled": False}
    if args.scaffold == "inferred":
        role_affinities, mined_scaffold_stats = mine_tile_role_affinities(
            phase1_dir=args.phase1_dir,
            phase2_dir=args.phase2_dir,
            prototypes=prototypes,
            palette=metadata["palette"],
            air_id=int(metadata["air_id"]),
            batch_size=args.scaffold_batch_size,
        )
        cell_weights, scaffold_stats = build_scaffold_cell_weights(
            base_weights=weights,
            active_tiles=active_tiles,
            prototypes=prototypes,
            palette=metadata["palette"],
            air_id=int(metadata["air_id"]),
            grid_shape=args.grid_size,
            role_affinities=role_affinities,
            scaffold_stats=mined_scaffold_stats,
            seed=args.seed,
            roof_boost=args.roof_boost,
            wall_boost=args.wall_boost,
            opening_boost=args.opening_boost,
            foundation_boost=args.foundation_boost,
            interior_boost=args.interior_boost,
            opening_rate=args.opening_rate,
            roof_min_score=args.roof_min_score,
            roof_content_min=args.roof_content_min,
        )

    chosen_tiles, solve_stats = run_solver(
        allowed=allowed,
        active_tiles=active_tiles,
        weights=weights,
        cell_weights=cell_weights,
        grid_shape=args.grid_size,
        seed=args.seed,
        retries=args.retries,
        max_backtracks=args.max_backtracks,
    )
    generated_categories, reconstruction_stats = reconstruct_overlapping_volume(
        chosen_tiles,
        prototypes,
        metadata["palette"],
        int(metadata["air_id"]),
        args.non_air_threshold,
        args.tile_stride,
    )
    blocks, block_palette, post_stats = volume_to_block_ids(
        generated_categories,
        metadata["palette"],
        args.seed,
        args.variation_rate,
        args.ornament_rate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "chosen_tiles.npy", chosen_tiles)
    np.save(args.output_dir / "generated_categories.npy", generated_categories)
    write_sponge_schem(args.output, blocks, block_palette, args.data_version)

    tile_counts = Counter(int(value) for value in chosen_tiles.ravel())
    report = {
        "grid_shape_yzx": list(args.grid_size),
        "output_shape_yzx": list(generated_categories.shape),
        "seed": args.seed,
        "rules": rule_stats,
        "scaffold": scaffold_stats,
        "solve": {
            "attempt": solve_stats.attempt,
            "decisions": solve_stats.decisions,
            "backtracks": solve_stats.backtracks,
            "propagations": solve_stats.propagations,
            "max_depth": solve_stats.max_depth,
        },
        "reconstruction": reconstruction_stats,
        "tile_counts": {str(tile): int(count) for tile, count in tile_counts.most_common()},
        "post_processing": post_stats,
        "output_schematic": str(args.output),
    }
    write_json(args.output_dir / "generation_report.json", report)

    lines = [
        "# Phase IV Generation",
        "",
        f"Generated a {args.grid_size[0]}x{args.grid_size[1]}x{args.grid_size[2]} WFC cell grid from the Phase III ruleset.",
        f"The overlapping tile reconstruction produced a block volume of {generated_categories.shape[0]}x{generated_categories.shape[1]}x{generated_categories.shape[2]} (Y,Z,X).",
        f"Output schematic: `{args.output}`.",
        "",
        "## Solver",
        "",
        f"- Seed: {args.seed}",
        f"- Successful attempt: {solve_stats.attempt}",
        f"- Decisions: {solve_stats.decisions}",
        f"- Backtracks: {solve_stats.backtracks}",
        f"- Propagations: {solve_stats.propagations}",
        f"- Active tiles after preparation: {rule_stats['active_tile_count']}",
        f"- Allowed directional adjacencies after preparation: {rule_stats['allowed_adjacency_count']}",
        f"- Source ruleset allowed adjacencies: {rule_stats['source_ruleset_allowed_adjacency_count']}",
        f"- Tile stride: {args.tile_stride}",
        f"- Stride-overlap allowed adjacencies: {stride_rule_stats['stride_allowed_adjacency_count']}",
        f"- Dead-end tile avoidance: {rule_stats['avoid_dead_end_tiles']}",
        f"- Dead-end repair enabled: {rule_stats['repair_dead_ends']}",
        f"- Scaffold: {args.scaffold}",
        "",
    ]
    if scaffold_stats.get("enabled"):
        lines.extend(
            [
                "## Scaffold",
                "",
                f"- Mined training patch assignments: {scaffold_stats['assigned_training_patches']}",
                f"- Target role cells: {scaffold_stats['role_grid_counts']}",
                f"- Roof start y norm: {scaffold_stats['roof_start_y_norm_used']:.2f}",
                f"- Foundation end y norm: {scaffold_stats['foundation_end_y_norm_used']:.2f}",
                "",
            ]
        )
    lines.extend(
        [
        "## Reconstruction",
        "",
        f"- Tile stride: {reconstruction_stats['tile_stride']}",
        f"- Non-air union threshold: {reconstruction_stats['non_air_threshold']:.2f}",
        f"- Raw majority non-air voxels: {reconstruction_stats['raw_majority_non_air_voxels']}",
        f"- Union-poll non-air voxels: {reconstruction_stats['union_poll_non_air_voxels']}",
        f"- Union-poll detail voxels: {reconstruction_stats['union_poll_detail_voxels']}",
        f"- Union-poll air fraction: {reconstruction_stats['union_poll_air_fraction']:.1%}",
        f"- Mean non-air probability: {reconstruction_stats['mean_non_air_probability']:.1%}",
        "",
        "## Post-Processing",
        "",
        f"- Palette variation rate: {args.variation_rate:.3f}",
        f"- Ornament rate: {args.ornament_rate:.3f}",
        f"- Lanterns added: {post_stats['lanterns_added']}",
        "",
        "## Most Used Tiles",
        "",
        ]
    )
    for tile_id, count in tile_counts.most_common(12):
        lines.append(f"- tile {tile_id}: {count} cells")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Schematic: `{args.output}`",
            f"- Chosen tile grid: `{args.output_dir / 'chosen_tiles.npy'}`",
            f"- Generated category volume: `{args.output_dir / 'generated_categories.npy'}`",
            f"- Generation report: `{args.output_dir / 'generation_report.json'}`",
        ]
    )
    args.docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote generated schematic to {args.output}")
    print(f"Wrote generation report to {args.output_dir / 'generation_report.json'}")
    print(f"Wrote observations to {args.docs_path}")


if __name__ == "__main__":
    main()
