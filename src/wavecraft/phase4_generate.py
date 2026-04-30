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
        grid_shape: tuple[int, int, int],
        rng: np.random.Generator,
        max_backtracks: int,
    ) -> None:
        self.allowed = allowed
        self.active_tiles = active_tiles
        self.weights = weights.astype(np.float64)
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
            domain_weights = self.weights[domain]
            total = float(domain_weights.sum())
            if total <= 0.0:
                entropies[y, z, x] = -math.inf
                continue
            entropies[y, z, x] = math.log(total) - float((domain_weights * np.log(domain_weights)).sum() / total)
        minimum = float(np.min(entropies[unresolved]))
        candidate_positions = np.argwhere(np.isclose(entropies, minimum))
        choice = candidate_positions[int(self.rng.integers(0, len(candidate_positions)))]
        return int(choice[0]), int(choice[1]), int(choice[2])

    def ordered_tiles(self, domain: np.ndarray) -> list[int]:
        candidates = np.flatnonzero(domain)
        candidate_weights = self.weights[candidates].astype(np.float64)
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
        for tile_id in self.ordered_tiles(domains[y, z, x]):
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


def prepare_rules(
    allowed: np.ndarray,
    active_tiles: np.ndarray,
    weights: np.ndarray,
    avoid_dead_end_tiles: bool,
    repair_dead_ends: bool,
    overlap_mismatches: np.ndarray,
    max_repair_overlap_mismatch: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    prepared_allowed = allowed.copy()
    prepared_active = active_tiles.copy() & (weights > 0)
    repair_count = 0

    if repair_dead_ends:
        for tile_id in np.flatnonzero(prepared_active):
            for direction_index in range(len(DIRECTIONS)):
                if np.any(prepared_allowed[direction_index, tile_id] & prepared_active):
                    continue
                fallback = (overlap_mismatches[direction_index, tile_id] <= max_repair_overlap_mismatch) & prepared_active
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


def run_solver(
    allowed: np.ndarray,
    active_tiles: np.ndarray,
    weights: np.ndarray,
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
) -> tuple[np.ndarray, dict[str, Any]]:
    palette_size = len(palette_categories)
    grid_y, grid_z, grid_x = chosen_tiles.shape
    window_y, window_z, window_x = prototypes.shape[1:]
    output_shape = (grid_y + window_y - 1, grid_z + window_z - 1, grid_x + window_x - 1)
    votes = np.zeros((*output_shape, palette_size), dtype=np.uint16)

    for y in range(grid_y):
        for z in range(grid_z):
            for x in range(grid_x):
                patch = prototypes[int(chosen_tiles[y, z, x])]
                for py in range(window_y):
                    for pz in range(window_z):
                        for px in range(window_x):
                            votes[y + py, z + pz, x + px, int(patch[py, pz, px])] += 1

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
    parser.add_argument("--max-repair-overlap-mismatch", type=float, default=0.35)
    parser.add_argument("--non-air-threshold", type=float, default=0.5)
    parser.add_argument("--variation-rate", type=float, default=0.08)
    parser.add_argument("--ornament-rate", type=float, default=0.004)
    parser.add_argument("--data-version", type=int, default=4556)
    args = parser.parse_args()

    if args.non_air_threshold < 0.0 or args.non_air_threshold > 1.0:
        raise SystemExit("--non-air-threshold must be between 0 and 1")

    metadata = load_json(args.phase1_dir / "metadata.json")
    rules = np.load(args.phase3_dir / "ruleset.npz")
    prototypes = rules["prototypes"]
    allowed, active_tiles, weights, rule_stats = prepare_rules(
        allowed=rules["allowed"],
        active_tiles=rules["active_tiles"],
        weights=rules["weights"],
        avoid_dead_end_tiles=not args.allow_dead_end_tiles,
        repair_dead_ends=args.repair_dead_ends,
        overlap_mismatches=rules["overlap_mismatches"],
        max_repair_overlap_mismatch=args.max_repair_overlap_mismatch,
    )

    chosen_tiles, solve_stats = run_solver(
        allowed=allowed,
        active_tiles=active_tiles,
        weights=weights,
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
        f"- Dead-end tile avoidance: {rule_stats['avoid_dead_end_tiles']}",
        f"- Dead-end repair enabled: {rule_stats['repair_dead_ends']}",
        "",
        "## Reconstruction",
        "",
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
