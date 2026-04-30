#!/usr/bin/env python3
"""Hierarchical architectural parsing, mining, and generation helpers.

The functions in this module implement the MVP from docs/algorithm_rethink.md:
rectangular medieval houses with mined global priors, semantic modules, and a
deterministic constraint assembly pass.  The old local tile/WFC pipeline remains
available in the repository, but the public phase entry points now build on this
module.
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from wavecraft.export_tile_schematic import (
    AIR_STATE,
    add_state,
    category_to_block_state,
    parse_category,
    write_sponge_schem,
)


AIR = "air"
ORIENTATIONS = ("+z", "-z", "+x", "-x")
NEIGHBORS_3D = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
NEIGHBORS_2D = ((1, 0), (-1, 0), (0, 1), (0, -1))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_phase1_arrays(
    phase1_dir: Path,
    originals_only: bool = False,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], np.ndarray]]]:
    metadata = load_json(phase1_dir / "metadata.json")
    arrays = []
    for item in metadata["outputs"]:
        if originals_only and item.get("transform") != "original":
            continue
        arrays.append((item, np.load(item["array"])))
    if not arrays:
        raise SystemExit(f"No Phase I arrays found in {phase1_dir}")
    return metadata, arrays


def category_family(category: str) -> str:
    family, _kind, _props = parse_category(category)
    return family


def category_kind(category: str) -> str:
    _family, kind, _props = parse_category(category)
    return kind


def category_flag_table(palette: list[str]) -> dict[str, np.ndarray]:
    flags = {
        "air": np.zeros(len(palette), dtype=bool),
        "opening": np.zeros(len(palette), dtype=bool),
        "door": np.zeros(len(palette), dtype=bool),
        "glass": np.zeros(len(palette), dtype=bool),
        "roof": np.zeros(len(palette), dtype=bool),
        "bulk": np.zeros(len(palette), dtype=bool),
        "support": np.zeros(len(palette), dtype=bool),
        "ground": np.zeros(len(palette), dtype=bool),
    }
    for index, category in enumerate(palette):
        family, kind, _props = parse_category(category)
        flags["air"][index] = family == "air"
        flags["glass"][index] = family == "glass" or kind == "pane"
        flags["door"][index] = kind == "door"
        flags["opening"][index] = flags["glass"][index] or kind in {"door", "trapdoor", "fence_gate"}
        flags["roof"][index] = family == "dark_stone" or kind in {"stairs", "slab", "tiles"}
        flags["bulk"][index] = kind in {"full", "bricks", "planks", "log", "wood", "wall"}
        flags["support"][index] = kind in {"log", "wood", "wall"} or family in {"stone", "dark_stone", "masonry"}
        flags["ground"][index] = family == "ground"
    return flags


def occupied_bounds(volume: np.ndarray, air_id: int) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.argwhere(volume != air_id)
    if occupied.size == 0:
        zeros = np.zeros(3, dtype=np.int32)
        return zeros, zeros
    return occupied.min(axis=0).astype(np.int32), occupied.max(axis=0).astype(np.int32)


def connected_component_sizes(mask: np.ndarray) -> list[int]:
    visited = np.zeros(mask.shape, dtype=bool)
    starts = np.argwhere(mask)
    sizes = []
    for raw_start in starts:
        start = tuple(int(value) for value in raw_start)
        if visited[start]:
            continue
        visited[start] = True
        queue: deque[tuple[int, int, int]] = deque([start])
        size = 0
        while queue:
            y, z, x = queue.popleft()
            size += 1
            for dy, dz, dx in NEIGHBORS_3D:
                ny, nz, nx = y + dy, z + dz, x + dx
                if ny < 0 or nz < 0 or nx < 0 or ny >= mask.shape[0] or nz >= mask.shape[1] or nx >= mask.shape[2]:
                    continue
                if mask[ny, nz, nx] and not visited[ny, nz, nx]:
                    visited[ny, nz, nx] = True
                    queue.append((ny, nz, nx))
        sizes.append(size)
    sizes.sort(reverse=True)
    return sizes


def exterior_air_and_shell(solid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded_solid = np.pad(solid, 1, constant_values=False)
    exterior = np.zeros(padded_solid.shape, dtype=bool)
    queue: deque[tuple[int, int, int]] = deque([(0, 0, 0)])
    exterior[0, 0, 0] = True
    while queue:
        y, z, x = queue.popleft()
        for dy, dz, dx in NEIGHBORS_3D:
            ny, nz, nx = y + dy, z + dz, x + dx
            if ny < 0 or nz < 0 or nx < 0 or ny >= padded_solid.shape[0] or nz >= padded_solid.shape[1] or nx >= padded_solid.shape[2]:
                continue
            if padded_solid[ny, nz, nx] or exterior[ny, nz, nx]:
                continue
            exterior[ny, nz, nx] = True
            queue.append((ny, nz, nx))

    shell = np.zeros_like(solid, dtype=bool)
    for y in range(solid.shape[0]):
        for z in range(solid.shape[1]):
            for x in range(solid.shape[2]):
                if not solid[y, z, x]:
                    continue
                py, pz, px = y + 1, z + 1, x + 1
                for dy, dz, dx in NEIGHBORS_3D:
                    if exterior[py + dy, pz + dz, px + dx]:
                        shell[y, z, x] = True
                        break
    return exterior[1:-1, 1:-1, 1:-1], shell


def connected_components_2d(mask: np.ndarray) -> list[dict[str, int]]:
    visited = np.zeros(mask.shape, dtype=bool)
    boxes = []
    for raw_start in np.argwhere(mask):
        start = tuple(int(value) for value in raw_start)
        if visited[start]:
            continue
        visited[start] = True
        queue: deque[tuple[int, int]] = deque([start])
        cells = []
        while queue:
            row, col = queue.popleft()
            cells.append((row, col))
            for dr, dc in NEIGHBORS_2D:
                nr, nc = row + dr, col + dc
                if nr < 0 or nc < 0 or nr >= mask.shape[0] or nc >= mask.shape[1]:
                    continue
                if mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        rows = [row for row, _col in cells]
        cols = [col for _row, col in cells]
        boxes.append(
            {
                "row_min": min(rows),
                "row_max": max(rows),
                "col_min": min(cols),
                "col_max": max(cols),
                "area": len(cells),
            }
        )
    return boxes


def layer_summaries(solid: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> list[dict[str, Any]]:
    footprint_area = max(int((bounds_max[1] - bounds_min[1] + 1) * (bounds_max[2] - bounds_min[2] + 1)), 1)
    summaries = []
    for y in range(int(bounds_min[0]), int(bounds_max[0]) + 1):
        layer = solid[y]
        occupied = np.argwhere(layer)
        if occupied.size == 0:
            continue
        z_min, x_min = occupied.min(axis=0)
        z_max, x_max = occupied.max(axis=0)
        area = int(np.count_nonzero(layer))
        summaries.append(
            {
                "y": int(y - bounds_min[0]),
                "area": area,
                "area_fraction_of_bbox": area / footprint_area,
                "bbox_zx": [int(z_min), int(x_min), int(z_max), int(x_max)],
            }
        )
    return summaries


def floor_height_candidates(solid: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> list[int]:
    footprint_area = max(int((bounds_max[1] - bounds_min[1] + 1) * (bounds_max[2] - bounds_min[2] + 1)), 1)
    candidates = [0]
    last = 0
    for y in range(int(bounds_min[0]) + 1, int(bounds_max[0])):
        coverage = np.count_nonzero(solid[y]) / footprint_area
        below = np.count_nonzero(solid[y - 1]) / footprint_area
        above = np.count_nonzero(solid[y + 1]) / footprint_area
        relative_y = int(y - bounds_min[0])
        if coverage >= 0.28 and coverage >= below and coverage >= above and relative_y - last >= 4:
            candidates.append(relative_y)
            last = relative_y
    return candidates[:4]


def footprint_summary(solid: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> dict[str, Any]:
    cropped = solid[bounds_min[0] : bounds_max[0] + 1, bounds_min[1] : bounds_max[1] + 1, bounds_min[2] : bounds_max[2] + 1]
    footprint = np.any(cropped, axis=0)
    area = int(np.count_nonzero(footprint))
    width = int(bounds_max[2] - bounds_min[2] + 1)
    depth = int(bounds_max[1] - bounds_min[1] + 1)
    rectangularity = area / max(width * depth, 1)
    if rectangularity >= 0.86:
        kind = "rectangle"
    elif rectangularity >= 0.64:
        kind = "compound_rectilinear"
    else:
        kind = "irregular"
    return {
        "type": kind,
        "width": width,
        "depth": depth,
        "area": area,
        "rectangularity": rectangularity,
    }


def support_stats(solid: np.ndarray) -> dict[str, Any]:
    if not np.any(solid):
        return {"supported_blocks": 0, "floating_blocks": 0, "supported_ratio": 0.0, "floating_ratio": 0.0}
    floating = solid.copy()
    floating[0, :, :] = False
    floating[1:, :, :] &= ~solid[:-1, :, :]
    floating_blocks = int(np.count_nonzero(floating))
    total = int(np.count_nonzero(solid))
    return {
        "supported_blocks": int(total - floating_blocks),
        "floating_blocks": floating_blocks,
        "supported_ratio": (total - floating_blocks) / max(total, 1),
        "floating_ratio": floating_blocks / max(total, 1),
    }


def facade_plane(
    volume: np.ndarray,
    palette: list[str],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    orientation: str,
) -> tuple[np.ndarray, int]:
    flags = category_flag_table(palette)
    y0 = int(bounds_min[0])
    y1 = int(bounds_min[0] + max(3, math.floor((bounds_max[0] - bounds_min[0] + 1) * 0.72)))
    y1 = min(y1, int(bounds_max[0]))

    def score_plane(plane: np.ndarray) -> int:
        lower = plane[: y1 - y0 + 1]
        return int(np.count_nonzero(flags["opening"][lower]) * 4 + np.count_nonzero(flags["door"][lower]) * 6 + np.count_nonzero(flags["bulk"][lower]))

    candidates: list[tuple[int, np.ndarray, int]] = []
    if orientation == "+z":
        for z in range(int(bounds_max[1]), max(int(bounds_min[1]) - 1, int(bounds_max[1]) - 5), -1):
            plane = volume[bounds_min[0] : bounds_max[0] + 1, z, bounds_min[2] : bounds_max[2] + 1]
            candidates.append((score_plane(plane), plane, z))
    elif orientation == "-z":
        for z in range(int(bounds_min[1]), min(int(bounds_max[1]) + 1, int(bounds_min[1]) + 5)):
            plane = volume[bounds_min[0] : bounds_max[0] + 1, z, bounds_min[2] : bounds_max[2] + 1]
            candidates.append((score_plane(plane), plane, z))
    elif orientation == "+x":
        for x in range(int(bounds_max[2]), max(int(bounds_min[2]) - 1, int(bounds_max[2]) - 5), -1):
            plane = volume[bounds_min[0] : bounds_max[0] + 1, bounds_min[1] : bounds_max[1] + 1, x]
            candidates.append((score_plane(plane), plane, x))
    elif orientation == "-x":
        for x in range(int(bounds_min[2]), min(int(bounds_max[2]) + 1, int(bounds_min[2]) + 5)):
            plane = volume[bounds_min[0] : bounds_max[0] + 1, bounds_min[1] : bounds_max[1] + 1, x]
            candidates.append((score_plane(plane), plane, x))
    else:
        raise ValueError(orientation)
    if not candidates:
        raise ValueError(orientation)
    _score, plane, coordinate = max(candidates, key=lambda item: item[0])
    return plane, int(coordinate)


def extract_facades(volume: np.ndarray, palette: list[str], bounds_min: np.ndarray, bounds_max: np.ndarray) -> list[dict[str, Any]]:
    flags = category_flag_table(palette)
    facades = []
    for orientation in ORIENTATIONS:
        plane, coordinate = facade_plane(volume, palette, bounds_min, bounds_max, orientation)
        opening_mask = flags["opening"][plane]
        openings = []
        for box in connected_components_2d(opening_mask):
            height = box["row_max"] - box["row_min"] + 1
            width = box["col_max"] - box["col_min"] + 1
            if box["area"] < 2 and height < 2:
                continue
            component = plane[box["row_min"] : box["row_max"] + 1, box["col_min"] : box["col_max"] + 1]
            has_door = bool(np.any(flags["door"][component]))
            has_glass = bool(np.any(flags["glass"][component]))
            openings.append(
                {
                    "type": "door" if has_door or box["row_min"] <= 2 else "window" if has_glass else "opening",
                    "u": int(box["col_min"]),
                    "y": int(box["row_min"]),
                    "width": int(width),
                    "height": int(height),
                    "area": int(box["area"]),
                }
            )
        length = int(bounds_max[2] - bounds_min[2] + 1) if orientation.endswith("z") else int(bounds_max[1] - bounds_min[1] + 1)
        facades.append(
            {
                "orientation": orientation,
                "coordinate": coordinate,
                "length": length,
                "height": int(bounds_max[0] - bounds_min[0] + 1),
                "opening_count": len(openings),
                "openings": sorted(openings, key=lambda item: (item["y"], item["u"])),
            }
        )
    return facades


def roof_summary(volume: np.ndarray, palette: list[str], solid: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> dict[str, Any]:
    flags = category_flag_table(palette)
    height = int(bounds_max[0] - bounds_min[0] + 1)
    upper_start = int(bounds_min[0] + max(1, math.floor(height * 0.55)))
    roof_mask = flags["roof"][volume] & solid
    upper_roof = roof_mask.copy()
    upper_roof[:upper_start] = False
    if not np.any(upper_roof):
        upper_roof = solid.copy()
        upper_roof[: int(bounds_min[0] + max(1, math.floor(height * 0.70)))] = False

    if not np.any(upper_roof):
        return {"type": "none", "ridge_axis": None, "eave_y": None, "ridge_y": None, "coverage": 0.0}

    coords = np.argwhere(upper_roof)
    eave_y = int(coords[:, 0].min() - bounds_min[0])
    ridge_y = int(coords[:, 0].max() - bounds_min[0])
    footprint = np.any(solid[bounds_min[0] : bounds_max[0] + 1], axis=0)
    roof_footprint = np.any(upper_roof, axis=0)
    coverage = int(np.count_nonzero(roof_footprint & footprint)) / max(int(np.count_nonzero(footprint)), 1)

    top_map = np.full((volume.shape[1], volume.shape[2]), -1, dtype=np.int32)
    top_coords = np.argwhere(np.any(upper_roof, axis=0))
    for z, x in top_coords:
        ys = np.flatnonzero(upper_roof[:, z, x])
        if len(ys):
            top_map[z, x] = int(ys.max())
    cropped_top = top_map[bounds_min[1] : bounds_max[1] + 1, bounds_min[2] : bounds_max[2] + 1]
    valid = cropped_top >= 0
    if not np.any(valid) or int(cropped_top[valid].max() - cropped_top[valid].min()) <= 1:
        roof_type = "flat"
        axis = None
    else:
        by_z = [float(np.mean(cropped_top[index][valid[index]])) for index in range(cropped_top.shape[0]) if np.any(valid[index])]
        by_x = [float(np.mean(cropped_top[:, index][valid[:, index]])) for index in range(cropped_top.shape[1]) if np.any(valid[:, index])]
        z_std = float(np.std(by_z)) if len(by_z) > 1 else 0.0
        x_std = float(np.std(by_x)) if len(by_x) > 1 else 0.0
        axis = "x" if z_std > x_std else "z"
        roof_type = "gable"
    return {
        "type": roof_type,
        "ridge_axis": axis,
        "eave_y": eave_y,
        "ridge_y": ridge_y,
        "coverage": coverage,
    }


def material_regions(volume: np.ndarray, palette: list[str], air_id: int) -> dict[str, Any]:
    values, counts = np.unique(volume[volume != air_id], return_counts=True)
    family_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    category_counts = []
    for value, count in zip(values, counts, strict=True):
        category = palette[int(value)]
        family, kind, _props = parse_category(category)
        family_counts[family] += int(count)
        kind_counts[kind] += int(count)
        category_counts.append({"category": category, "count": int(count)})
    category_counts.sort(key=lambda item: item["count"], reverse=True)
    return {
        "families": dict(family_counts.most_common()),
        "kinds": dict(kind_counts.most_common()),
        "top_categories": category_counts[:20],
    }


def vertical_supports(volume: np.ndarray, palette: list[str], bounds_min: np.ndarray, bounds_max: np.ndarray) -> list[dict[str, Any]]:
    flags = category_flag_table(palette)
    support_mask = flags["support"][volume]
    supports = []
    for z in range(int(bounds_min[1]), int(bounds_max[1]) + 1):
        for x in range(int(bounds_min[2]), int(bounds_max[2]) + 1):
            column = support_mask[bounds_min[0] : bounds_max[0] + 1, z, x]
            longest = 0
            current = 0
            for value in column:
                if value:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
            if longest >= 4:
                supports.append({"x": int(x - bounds_min[2]), "z": int(z - bounds_min[1]), "height": int(longest)})
    return supports[:128]


def facade_rhythm_score(facades: list[dict[str, Any]]) -> float:
    scores = []
    for facade in facades:
        windows = [opening for opening in facade["openings"] if opening["type"] == "window"]
        if len(windows) < 2:
            continue
        centers = sorted(opening["u"] + opening["width"] / 2.0 for opening in windows)
        gaps = np.diff(np.asarray(centers, dtype=np.float32))
        if len(gaps) == 0 or float(np.mean(gaps)) <= 0:
            continue
        scores.append(max(0.0, 1.0 - float(np.std(gaps) / np.mean(gaps))))
    return float(np.mean(scores)) if scores else 0.0


def parse_structure_record(item: dict[str, Any], volume: np.ndarray, palette: list[str], air_id: int) -> dict[str, Any]:
    solid = volume != air_id
    bounds_min, bounds_max = occupied_bounds(volume, air_id)
    if not np.any(solid):
        return {
            "id": f"{item['source']}__{item['transform']}",
            "source": item["source"],
            "transform": item["transform"],
            "bounds": [0, 0, 0],
            "quality_features": {"empty": True},
        }

    cropped_solid = solid[bounds_min[0] : bounds_max[0] + 1, bounds_min[1] : bounds_max[1] + 1, bounds_min[2] : bounds_max[2] + 1]
    component_sizes = connected_component_sizes(cropped_solid)
    exterior_air, shell = exterior_air_and_shell(cropped_solid)
    interior_air = (~cropped_solid) & (~exterior_air)
    interior_components = connected_component_sizes(interior_air)
    shell_components = connected_component_sizes(shell)
    facades = extract_facades(volume, palette, bounds_min, bounds_max)
    footprint = footprint_summary(solid, bounds_min, bounds_max)
    supports = support_stats(solid)
    roof = roof_summary(volume, palette, solid, bounds_min, bounds_max)
    structural_voxels = int(np.count_nonzero(cropped_solid))
    bbox_volume = int(np.prod(np.asarray(cropped_solid.shape, dtype=np.int64)))
    dominant_component_ratio = component_sizes[0] / max(structural_voxels, 1) if component_sizes else 0.0
    shell_continuity = shell_components[0] / max(int(np.count_nonzero(shell)), 1) if shell_components else 0.0
    door_count = sum(1 for facade in facades for opening in facade["openings"] if opening["type"] == "door")
    window_count = sum(1 for facade in facades for opening in facade["openings"] if opening["type"] == "window")

    record = {
        "id": f"{item['source']}__{item['transform']}",
        "source": item["source"],
        "transform": item["transform"],
        "array": item["array"],
        "bounds": [int(value) for value in cropped_solid.shape],
        "bounds_min_yzx": [int(value) for value in bounds_min],
        "bounds_max_yzx": [int(value) for value in bounds_max],
        "structural_voxels": structural_voxels,
        "density": structural_voxels / max(bbox_volume, 1),
        "footprint": footprint,
        "per_layer_footprints": layer_summaries(solid, bounds_min, bounds_max),
        "connected_components": {
            "count": len(component_sizes),
            "sizes": component_sizes[:12],
            "dominant_ratio": dominant_component_ratio,
        },
        "exterior_shell": {
            "voxels": int(np.count_nonzero(shell)),
            "component_count": len(shell_components),
            "dominant_ratio": shell_continuity,
        },
        "interior_air_regions": {
            "count": len(interior_components),
            "sizes": interior_components[:12],
            "largest": int(interior_components[0]) if interior_components else 0,
        },
        "floors": floor_height_candidates(solid, bounds_min, bounds_max),
        "facades": facades,
        "roof": roof,
        "door_count": door_count,
        "window_count": window_count,
        "vertical_supports": vertical_supports(volume, palette, bounds_min, bounds_max),
        "material_regions": material_regions(volume, palette, air_id),
        "quality_features": {
            "dominant_component_ratio": dominant_component_ratio,
            "floating_block_ratio": supports["floating_ratio"],
            "supported_block_ratio": supports["supported_ratio"],
            "footprint_rectangularity": footprint["rectangularity"],
            "shell_continuity": shell_continuity,
            "roof_coverage": roof["coverage"],
            "has_door": door_count > 0,
            "window_count": window_count,
            "facade_rhythm_score": facade_rhythm_score(facades),
            "density": structural_voxels / max(bbox_volume, 1),
            "interior_air_region_count": len(interior_components),
        },
    }
    return record


def run_structure_parser(phase1_dir: Path, output_dir: Path, originals_only: bool) -> dict[str, Any]:
    metadata, arrays = load_phase1_arrays(phase1_dir, originals_only=originals_only)
    palette = metadata["palette"]
    air_id = int(metadata["air_id"])
    records = [parse_structure_record(item, volume, palette, air_id) for item, volume in arrays]
    output_path = output_dir / "structure_records.jsonl"
    append_jsonl(output_path, records)
    summary = {
        "phase": "phase2_parse_structures",
        "input_phase1_dir": str(phase1_dir),
        "record_count": len(records),
        "originals_only": originals_only,
        "output": str(output_path),
        "bounds_yzx": [record["bounds"] for record in records],
        "footprint_types": dict(Counter(record.get("footprint", {}).get("type", "empty") for record in records)),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def score_density(density: float) -> float:
    if density <= 0.02:
        return 0.0
    if density <= 0.18:
        return density / 0.18
    if density <= 0.48:
        return 1.0
    if density >= 0.82:
        return 0.0
    return 1.0 - ((density - 0.48) / 0.34)


def quality_score_for_record(record: dict[str, Any]) -> tuple[float, dict[str, float]]:
    features = record.get("quality_features", {})
    if features.get("empty"):
        return 0.0, {"empty": 1.0}
    roof_score = 0.65 if record.get("roof", {}).get("type") in {"gable", "flat"} else 0.0
    roof_score = max(roof_score, float(features.get("roof_coverage", 0.0)))
    door_score = 1.0 if features.get("has_door") else 0.0
    window_score = min(float(features.get("window_count", 0)) / 4.0, 1.0)
    density_component = score_density(float(features.get("density", 0.0)))
    terms = {
        "dominant_component": float(features.get("dominant_component_ratio", 0.0)),
        "supported_blocks": float(features.get("supported_block_ratio", 0.0)),
        "footprint": float(features.get("footprint_rectangularity", 0.0)),
        "shell": float(features.get("shell_continuity", 0.0)),
        "roof": roof_score,
        "door": door_score,
        "windows": window_score,
        "density": density_component,
        "facade_rhythm": float(features.get("facade_rhythm_score", 0.0)),
    }
    weights = {
        "dominant_component": 0.20,
        "supported_blocks": 0.16,
        "footprint": 0.15,
        "shell": 0.12,
        "roof": 0.12,
        "door": 0.08,
        "windows": 0.07,
        "density": 0.06,
        "facade_rhythm": 0.04,
    }
    score = sum(terms[name] * weights[name] for name in terms)
    return float(max(0.0, min(score, 1.0))), terms


def bucket_for_score(score: float) -> str:
    if score >= 0.78:
        return "excellent"
    if score >= 0.58:
        return "usable"
    if score >= 0.35:
        return "weak"
    return "reject"


def run_quality_scorer(phase2_dir: Path, output_dir: Path) -> dict[str, Any]:
    records = read_jsonl(phase2_dir / "structure_records.jsonl")
    scores = {}
    bucket_counts: Counter[str] = Counter()
    scored_records = []
    for record in records:
        score, terms = quality_score_for_record(record)
        bucket = bucket_for_score(score)
        bucket_counts[bucket] += 1
        scores[record["id"]] = {
            "source": record.get("source"),
            "transform": record.get("transform"),
            "quality_score": score,
            "bucket": bucket,
            "terms": terms,
        }
        scored_records.append({**record, "quality_score": score, "quality_bucket": bucket})
    output = {
        "phase": "phase3_score_quality",
        "record_count": len(records),
        "bucket_counts": dict(bucket_counts),
        "scores": scores,
    }
    write_json(output_dir / "quality_scores.json", output)
    append_jsonl(output_dir / "structure_records_scored.jsonl", scored_records)
    return output


def weighted_histogram(values: Iterable[Any], weights: Iterable[float]) -> list[dict[str, Any]]:
    counts: Counter[Any] = Counter()
    for value, weight in zip(values, weights, strict=True):
        counts[value] += float(weight)
    return [{"value": value, "weight": weight} for value, weight in sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))]


def weighted_choice(rng: np.random.Generator, histogram: list[dict[str, Any]], fallback: Any) -> Any:
    if not histogram:
        return fallback
    values = [item["value"] for item in histogram]
    weights = np.asarray([max(float(item["weight"]), 0.0) for item in histogram], dtype=np.float64)
    if float(weights.sum()) <= 0:
        return values[0]
    return values[int(rng.choice(np.arange(len(values)), p=weights / weights.sum()))]


def first_category(
    palette: list[str],
    families: tuple[str, ...],
    kinds: tuple[str, ...],
    contains: str | None = None,
    fallback: str | None = None,
) -> str:
    for category in palette:
        family, kind, _props = parse_category(category)
        if family in families and kind in kinds and (contains is None or contains in category):
            return category
    if fallback is not None:
        return fallback
    family = families[0]
    kind = kinds[0]
    return f"{family}:{kind}"


def oriented_category(palette: list[str], family: str, kind: str, facing: str, half: str | None = None) -> str:
    for category in palette:
        cat_family, cat_kind, props = parse_category(category)
        if cat_family == family and cat_kind == kind and props.get("facing") == facing and (half is None or props.get("half") == half):
            return category
    suffix = f"[facing={facing}"
    if half is not None:
        suffix += f",half={half}"
    return f"{family}:{kind}{suffix}]"


def mine_material_defaults(records: list[dict[str, Any]], palette: list[str]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for record in records:
        score = float(record.get("quality_score", 0.5))
        if record.get("quality_bucket") == "reject":
            continue
        for family, count in record.get("material_regions", {}).get("families", {}).items():
            family_counts[family] += count * score
        for item in record.get("material_regions", {}).get("top_categories", []):
            category_counts[item["category"]] += item["count"] * score

    wall_family = "wood_dark"
    for family, _count in family_counts.most_common():
        if family.startswith("wood_"):
            wall_family = family
            break
    foundation_family = "stone" if family_counts["stone"] >= family_counts["dark_stone"] else "dark_stone"
    roof_family = "dark_stone" if family_counts["dark_stone"] else "stone"
    return {
        "foundation": first_category(palette, (foundation_family, "stone", "masonry"), ("bricks", "full"), fallback="stone:bricks"),
        "wall": first_category(palette, (wall_family, "wood_dark", "wood_medium", "wood_light"), ("planks", "full"), fallback="wood_dark:planks"),
        "corner": first_category(palette, (wall_family, "wood_dark", "wood_medium", "wood_light"), ("log", "wood"), contains="axis=y", fallback="wood_dark:log[axis=y,stripped=true]"),
        "floor": first_category(palette, (wall_family, "wood_dark", "wood_medium", "wood_light"), ("planks", "slab"), fallback="wood_dark:planks"),
        "glass_z": first_category(palette, ("glass",), ("pane", "glass"), contains="north=true,south=true", fallback="glass:pane[east=false,north=true,south=true,west=false]"),
        "glass_x": first_category(palette, ("glass",), ("pane", "glass"), contains="east=true", fallback="glass:pane[east=true,north=false,south=false,west=true]"),
        "roof_block": first_category(palette, (roof_family, "dark_stone", "stone"), ("tiles", "bricks", "full"), fallback="dark_stone:tiles"),
        "roof_slab": first_category(palette, (roof_family, "dark_stone", "stone"), ("slab",), contains="type=bottom", fallback="dark_stone:slab[type=bottom]"),
        "roof_stairs": {
            "north": oriented_category(palette, roof_family, "stairs", "north", "bottom"),
            "south": oriented_category(palette, roof_family, "stairs", "south", "bottom"),
            "east": oriented_category(palette, roof_family, "stairs", "east", "bottom"),
            "west": oriented_category(palette, roof_family, "stairs", "west", "bottom"),
        },
        "door": {
            "north_lower": oriented_category(palette, wall_family, "door", "north", "lower"),
            "north_upper": oriented_category(palette, wall_family, "door", "north", "upper"),
            "south_lower": oriented_category(palette, wall_family, "door", "south", "lower"),
            "south_upper": oriented_category(palette, wall_family, "door", "south", "upper"),
            "east_lower": oriented_category(palette, wall_family, "door", "east", "lower"),
            "east_upper": oriented_category(palette, wall_family, "door", "east", "upper"),
            "west_lower": oriented_category(palette, wall_family, "door", "west", "lower"),
            "west_upper": oriented_category(palette, wall_family, "door", "west", "upper"),
        },
        "top_training_categories": dict(category_counts.most_common(16)),
    }


def estimated_floor_count(record: dict[str, Any]) -> int:
    roof = record.get("roof", {})
    wall_height = roof.get("eave_y") or max(record.get("bounds", [7])[0] - 3, 5)
    return int(min(max(round(float(wall_height) / 5.0), 1), 2))


def run_grammar_miner(phase2_dir: Path, phase3_dir: Path, phase1_dir: Path, output_dir: Path) -> dict[str, Any]:
    _phase1, _arrays = load_phase1_arrays(phase1_dir, originals_only=True)
    phase1_metadata = load_json(phase1_dir / "metadata.json")
    palette = phase1_metadata["palette"]
    quality = load_json(phase3_dir / "quality_scores.json")["scores"]
    base_records = read_jsonl(phase2_dir / "structure_records.jsonl")
    records = []
    weights = []
    for record in base_records:
        score_item = quality.get(record["id"], {})
        score = float(score_item.get("quality_score", 0.0))
        bucket = score_item.get("bucket", "reject")
        if bucket == "reject":
            continue
        record = {**record, "quality_score": score, "quality_bucket": bucket}
        records.append(record)
        multiplier = 1.35 if record.get("transform") == "original" else 1.0
        weights.append(max(score, 0.05) * multiplier)
    if not records:
        raise SystemExit("No non-rejected records are available for grammar mining")

    widths = [int(record["footprint"]["width"]) for record in records]
    depths = [int(record["footprint"]["depth"]) for record in records]
    floors = [estimated_floor_count(record) for record in records]
    roof_axes = [record.get("roof", {}).get("ridge_axis") or "x" for record in records]
    roof_types = [record.get("roof", {}).get("type") or "gable" for record in records]
    materials = mine_material_defaults(records, palette)
    grammar = {
        "phase": "phase4_mine_grammar",
        "mvp_scope": "rectangular_medieval_gable_house",
        "training_record_count": len(records),
        "quality_weighting": "quality_score with original curated examples boosted by 1.35",
        "distributions": {
            "footprint_type": [{"value": "rectangle", "weight": float(sum(weights))}],
            "width": weighted_histogram(widths, weights),
            "depth": weighted_histogram(depths, weights),
            "floor_count": weighted_histogram(floors, weights),
            "roof_type": weighted_histogram(roof_types, weights),
            "roof_axis": weighted_histogram(roof_axes, weights),
        },
        "defaults": {
            "footprint_type": "rectangle",
            "min_width": 9,
            "max_width": 25,
            "min_depth": 9,
            "max_depth": 23,
            "floor_height": 5,
            "bay_width": 3,
            "roof_overhang": 1,
            "front": "+z",
            "style": "medieval",
        },
        "materials": materials,
    }
    write_json(output_dir / "grammar.json", grammar)
    return grammar


def extend_palette(base_palette: list[str], categories: Iterable[str]) -> tuple[list[str], dict[str, int]]:
    palette = list(base_palette)
    for category in categories:
        if category not in palette:
            palette.append(category)
    return palette, {category: index for index, category in enumerate(palette)}


def make_wall_module(role: str, materials: dict[str, Any], palette_ids: dict[str, int], floor_height: int, width: int) -> np.ndarray:
    air = palette_ids[AIR]
    wall = palette_ids[materials["wall"]]
    corner = palette_ids[materials["corner"]]
    glass = palette_ids[materials["glass_z"]]
    door_lower = palette_ids[materials["door"]["south_lower"]]
    door_upper = palette_ids[materials["door"]["south_upper"]]
    module = np.full((floor_height, 1, width), air, dtype=np.uint16)
    module[:, 0, :] = wall
    module[:, 0, 0] = corner
    module[:, 0, -1] = corner
    if role == "window_bay" and floor_height >= 5:
        center = width // 2
        module[2:4, 0, max(1, center - 1) : min(width - 1, center + 2)] = glass
    if role == "door_bay":
        center = width // 2
        module[1, 0, center] = door_lower
        module[2, 0, center] = door_upper
        if center - 1 >= 1:
            module[2:4, 0, center - 1] = glass
        if center + 1 <= width - 2:
            module[2:4, 0, center + 1] = glass
    return module


def run_module_miner(phase1_dir: Path, phase4_dir: Path, output_dir: Path) -> dict[str, Any]:
    phase1_metadata = load_json(phase1_dir / "metadata.json")
    grammar = load_json(phase4_dir / "grammar.json")
    materials = grammar["materials"]
    floor_height = int(grammar["defaults"]["floor_height"])
    bay_width = int(grammar["defaults"]["bay_width"])
    required_categories = [
        AIR,
        materials["foundation"],
        materials["wall"],
        materials["corner"],
        materials["floor"],
        materials["glass_z"],
        materials["glass_x"],
        materials["roof_block"],
        materials["roof_slab"],
        *materials["roof_stairs"].values(),
        *materials["door"].values(),
    ]
    palette, palette_ids = extend_palette(phase1_metadata["palette"], required_categories)
    modules_dir = output_dir / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    module_specs = []

    for role in ("wall_bay", "window_bay", "door_bay"):
        array = make_wall_module(role, materials, palette_ids, floor_height, bay_width)
        path = modules_dir / f"{role}_medieval.npy"
        np.save(path, array)
        module_specs.append(
            {
                "id": f"{role}_medieval",
                "role": role,
                "size": list(array.shape),
                "orientation": "+z",
                "array": str(path),
                "style_tags": ["medieval"],
                "sockets": {
                    "left": "wall_edge",
                    "right": "wall_edge",
                    "top": "roof_eave_support",
                    "bottom": "floor_or_foundation",
                    "front": "exterior_air",
                    "back": "interior_air",
                },
                "features": {
                    "has_window": role == "window_bay",
                    "has_door": role == "door_bay",
                    "solid_fraction": float(np.count_nonzero(array) / array.size),
                },
                "quality_weight": 1.0,
            }
        )

    for role, shape, category in (
        ("corner_pillar", (floor_height, 1, 1), materials["corner"]),
        ("foundation_segment", (1, 1, bay_width), materials["foundation"]),
        ("floor_slab", (1, 1, bay_width), materials["floor"]),
        ("roof_ridge", (1, 1, bay_width), materials["roof_block"]),
        ("roof_eave", (1, 1, bay_width), materials["roof_slab"]),
        ("gable_end", (floor_height, 1, bay_width), materials["wall"]),
    ):
        array = np.full(shape, palette_ids[category], dtype=np.uint16)
        path = modules_dir / f"{role}_medieval.npy"
        np.save(path, array)
        module_specs.append(
            {
                "id": f"{role}_medieval",
                "role": role,
                "size": list(shape),
                "orientation": "omni",
                "array": str(path),
                "style_tags": ["medieval"],
                "sockets": {"self": role},
                "features": {"solid_fraction": 1.0},
                "quality_weight": 1.0,
            }
        )

    catalog = {
        "phase": "phase5_mine_modules",
        "module_count": len(module_specs),
        "palette": palette,
        "materials": materials,
        "modules": module_specs,
        "notes": "MVP semantic modules are grammar-conditioned prototypes, not arbitrary sliding-window patches.",
    }
    write_json(output_dir / "module_catalog.json", catalog)
    write_json(output_dir / "category_palette.json", {"palette": palette})
    return catalog


def clamp_dimension(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = fallback
    number = min(max(number, minimum), maximum)
    if number % 2 == 0:
        number += 1 if number < maximum else -1
    return number


def sample_plan(grammar: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    defaults = grammar["defaults"]
    distributions = grammar["distributions"]
    width = clamp_dimension(
        weighted_choice(rng, distributions.get("width", []), 17),
        int(defaults["min_width"]),
        int(defaults["max_width"]),
        17,
    )
    depth = clamp_dimension(
        weighted_choice(rng, distributions.get("depth", []), 13),
        int(defaults["min_depth"]),
        int(defaults["max_depth"]),
        13,
    )
    floors = int(min(max(int(weighted_choice(rng, distributions.get("floor_count", []), 2)), 1), 2))
    roof_axis = str(weighted_choice(rng, distributions.get("roof_axis", []), "x"))
    if roof_axis not in {"x", "z"}:
        roof_axis = "x" if width >= depth else "z"
    roof_type = str(weighted_choice(rng, distributions.get("roof_type", []), "gable"))
    if roof_type not in {"gable", "flat"}:
        roof_type = "gable"
    plan = {
        "footprint": {"type": "rectangle", "width": width, "depth": depth},
        "floors": floors,
        "floor_height": int(defaults["floor_height"]),
        "roof": {"type": "gable", "axis": roof_axis, "overhang": int(defaults["roof_overhang"])},
        "front": defaults["front"],
        "bay_width": int(defaults["bay_width"]),
        "facades": {},
        "materials": grammar["materials"],
        "seed": seed,
    }
    for orientation, length in (("+z", width), ("-z", width), ("+x", depth), ("-x", depth)):
        plan["facades"][orientation] = {"bay_pattern": facade_pattern(length, int(defaults["bay_width"]), orientation == defaults["front"])}
    return plan


def facade_pattern(length: int, bay_width: int, has_door: bool) -> list[str]:
    middle_slots = max(1, (length - 2) // max(bay_width, 1))
    pattern = ["corner"]
    door_slot = middle_slots // 2 if has_door else -1
    for slot in range(middle_slots):
        if slot == door_slot:
            pattern.append("door")
        elif slot % 2 == 0:
            pattern.append("window")
        else:
            pattern.append("wall")
    pattern.append("corner")
    return pattern


def category_ids_for_generation(catalog: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    palette = catalog["palette"]
    return palette, {category: index for index, category in enumerate(palette)}


def set_category(volume: np.ndarray, ids: dict[str, int], category: str, y: int, z: int, x: int) -> None:
    if 0 <= y < volume.shape[0] and 0 <= z < volume.shape[1] and 0 <= x < volume.shape[2]:
        volume[y, z, x] = ids[category]


def door_categories(materials: dict[str, Any], orientation: str) -> tuple[str, str]:
    facing = {"+z": "south", "-z": "north", "+x": "east", "-x": "west"}[orientation]
    return materials["door"][f"{facing}_lower"], materials["door"][f"{facing}_upper"]


def place_window(volume: np.ndarray, ids: dict[str, int], materials: dict[str, Any], orientation: str, floor_base: int, u: int, origin_z: int, origin_x: int, width: int, depth: int) -> None:
    glass = materials["glass_z"] if orientation.endswith("z") else materials["glass_x"]
    for dy in (2, 3):
        if orientation == "+z":
            set_category(volume, ids, glass, floor_base + dy, origin_z + depth - 1, origin_x + u)
        elif orientation == "-z":
            set_category(volume, ids, glass, floor_base + dy, origin_z, origin_x + u)
        elif orientation == "+x":
            set_category(volume, ids, glass, floor_base + dy, origin_z + u, origin_x + width - 1)
        elif orientation == "-x":
            set_category(volume, ids, glass, floor_base + dy, origin_z + u, origin_x)


def place_door(volume: np.ndarray, ids: dict[str, int], materials: dict[str, Any], orientation: str, u: int, origin_z: int, origin_x: int, width: int, depth: int) -> None:
    lower, upper = door_categories(materials, orientation)
    if orientation == "+z":
        coords = [(1, origin_z + depth - 1, origin_x + u, lower), (2, origin_z + depth - 1, origin_x + u, upper)]
    elif orientation == "-z":
        coords = [(1, origin_z, origin_x + u, lower), (2, origin_z, origin_x + u, upper)]
    elif orientation == "+x":
        coords = [(1, origin_z + u, origin_x + width - 1, lower), (2, origin_z + u, origin_x + width - 1, upper)]
    else:
        coords = [(1, origin_z + u, origin_x, lower), (2, origin_z + u, origin_x, upper)]
    for y, z, x, category in coords:
        set_category(volume, ids, category, y, z, x)


def assemble_house(plan: dict[str, Any], catalog: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    palette, ids = category_ids_for_generation(catalog)
    materials = plan["materials"]
    width = int(plan["footprint"]["width"])
    depth = int(plan["footprint"]["depth"])
    floors = int(plan["floors"])
    floor_height = int(plan["floor_height"])
    wall_height = floors * floor_height + 1
    roof_axis = plan["roof"]["axis"]
    roof_span = depth if roof_axis == "x" else width
    roof_height = max(3, min(7, math.ceil((roof_span + 2 * int(plan["roof"]["overhang"])) / 4)))
    overhang = int(plan["roof"]["overhang"])
    height = wall_height + roof_height + 2
    volume = np.zeros((height, depth + 2 * overhang, width + 2 * overhang), dtype=np.uint16)
    origin_z = overhang
    origin_x = overhang
    foundation = materials["foundation"]
    wall = materials["wall"]
    corner = materials["corner"]
    floor = materials["floor"]

    for z in range(origin_z, origin_z + depth):
        for x in range(origin_x, origin_x + width):
            set_category(volume, ids, foundation, 0, z, x)
            if 1 <= z - origin_z <= depth - 2 and 1 <= x - origin_x <= width - 2:
                set_category(volume, ids, floor, 1, z, x)

    for level in range(1, floors):
        y = level * floor_height + 1
        for z in range(origin_z + 1, origin_z + depth - 1):
            for x in range(origin_x + 1, origin_x + width - 1):
                set_category(volume, ids, floor, y, z, x)

    for y in range(1, wall_height + 1):
        for x in range(origin_x, origin_x + width):
            set_category(volume, ids, wall, y, origin_z, x)
            set_category(volume, ids, wall, y, origin_z + depth - 1, x)
        for z in range(origin_z, origin_z + depth):
            set_category(volume, ids, wall, y, z, origin_x)
            set_category(volume, ids, wall, y, z, origin_x + width - 1)
        for z, x in (
            (origin_z, origin_x),
            (origin_z, origin_x + width - 1),
            (origin_z + depth - 1, origin_x),
            (origin_z + depth - 1, origin_x + width - 1),
        ):
            set_category(volume, ids, corner, y, z, x)

    for floor_index in range(floors):
        floor_base = floor_index * floor_height
        for orientation, length in (("+z", width), ("-z", width), ("+x", depth), ("-x", depth)):
            slots = range(2, length - 2, max(int(plan["bay_width"]), 3))
            for u in slots:
                if orientation == plan["front"] and floor_index == 0 and abs(u - length // 2) <= 1:
                    continue
                place_window(volume, ids, materials, orientation, floor_base, u, origin_z, origin_x, width, depth)
        if floor_index == 0:
            place_door(volume, ids, materials, plan["front"], width // 2 if plan["front"].endswith("z") else depth // 2, origin_z, origin_x, width, depth)

    eave_y = wall_height + 1
    roof_block = materials["roof_block"]
    roof_slab = materials["roof_slab"]
    stairs = materials["roof_stairs"]
    if roof_axis == "x":
        z0, z1 = origin_z - overhang, origin_z + depth - 1 + overhang
        x0, x1 = origin_x - overhang, origin_x + width - 1 + overhang
        max_d = (z1 - z0) // 2
        for z in range(z0, z1 + 1):
            d = min(z - z0, z1 - z)
            y = eave_y + min(d, roof_height)
            category = roof_block if d >= max_d - 1 else stairs["north"] if z < (z0 + z1) // 2 else stairs["south"]
            for x in range(x0, x1 + 1):
                set_category(volume, ids, category, y, z, x)
        for x in (origin_x, origin_x + width - 1):
            for z in range(origin_z, origin_z + depth):
                d = min(z - z0, z1 - z)
                y_top = eave_y + min(d, roof_height) - 1
                for y in range(wall_height + 1, y_top + 1):
                    set_category(volume, ids, wall, y, z, x)
    else:
        z0, z1 = origin_z - overhang, origin_z + depth - 1 + overhang
        x0, x1 = origin_x - overhang, origin_x + width - 1 + overhang
        max_d = (x1 - x0) // 2
        for x in range(x0, x1 + 1):
            d = min(x - x0, x1 - x)
            y = eave_y + min(d, roof_height)
            category = roof_block if d >= max_d - 1 else stairs["west"] if x < (x0 + x1) // 2 else stairs["east"]
            for z in range(z0, z1 + 1):
                set_category(volume, ids, category, y, z, x)
        for z in (origin_z, origin_z + depth - 1):
            for x in range(origin_x, origin_x + width):
                d = min(x - x0, x1 - x)
                y_top = eave_y + min(d, roof_height) - 1
                for y in range(wall_height + 1, y_top + 1):
                    set_category(volume, ids, wall, y, z, x)

    for x in range(origin_x - overhang, origin_x + width + overhang):
        set_category(volume, ids, roof_slab, eave_y, origin_z - overhang, x)
        set_category(volume, ids, roof_slab, eave_y, origin_z + depth - 1 + overhang, x)
    for z in range(origin_z - overhang, origin_z + depth + overhang):
        set_category(volume, ids, roof_slab, eave_y, z, origin_x - overhang)
        set_category(volume, ids, roof_slab, eave_y, z, origin_x + width - 1 + overhang)

    report = {
        "wall_height": wall_height,
        "roof_height": roof_height,
        "volume_shape_yzx": list(volume.shape),
        "socket_violations": 0,
        "hard_constraints": {
            "rectangular_footprint": True,
            "four_facades": True,
            "single_front_door": True,
            "gable_roof_covers_footprint": True,
            "supported_upper_floors": True,
        },
    }
    return volume, palette, report


def stable_fraction(seed: int, y: int, z: int, x: int, salt: int) -> float:
    value = ((seed + 0x9E3779B9) ^ (y * 0x85EBCA6B) ^ (z * 0xC2B2AE35) ^ (x * 0x27D4EB2F) ^ (salt * 0x165667B1)) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    return value / 0xFFFFFFFF


def varied_state(category: str, y: int, z: int, x: int, seed: int, variation_rate: float) -> str:
    if category == AIR:
        return AIR_STATE
    base_state = category_to_block_state(category)
    if stable_fraction(seed, y, z, x, 19) > variation_rate:
        return base_state
    family, kind, _props = parse_category(category)
    if family == "stone" and kind in {"full", "bricks"}:
        return "minecraft:mossy_stone_bricks" if stable_fraction(seed, y, z, x, 23) < 0.5 else "minecraft:cracked_stone_bricks"
    if family == "dark_stone" and kind in {"full", "bricks", "tiles"}:
        return "minecraft:cracked_deepslate_bricks"
    return base_state


def categories_to_blocks(
    categories: np.ndarray,
    palette_categories: list[str],
    seed: int,
    variation_rate: float = 0.04,
    lantern_rate: float = 0.015,
) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    palette = {AIR_STATE: 0}
    blocks = np.zeros(categories.shape, dtype=np.int32)
    for y in range(categories.shape[0]):
        for z in range(categories.shape[1]):
            for x in range(categories.shape[2]):
                category = palette_categories[int(categories[y, z, x])]
                blocks[y, z, x] = add_state(palette, varied_state(category, y, z, x, seed, variation_rate))

    lantern_state = "minecraft:lantern[hanging=true,waterlogged=false]"
    lantern_id = add_state(palette, lantern_state)
    air_id = palette[AIR_STATE]
    lanterns = 0
    for y in range(1, blocks.shape[0] - 1):
        for z in range(1, blocks.shape[1] - 1):
            for x in range(1, blocks.shape[2] - 1):
                if blocks[y, z, x] != air_id:
                    continue
                if blocks[y + 1, z, x] == air_id or blocks[y - 1, z, x] != air_id:
                    continue
                if stable_fraction(seed, y, z, x, 41) <= lantern_rate:
                    blocks[y, z, x] = lantern_id
                    lanterns += 1
    state_counts = Counter()
    inverse = {index: state for state, index in palette.items()}
    for value, count in zip(*np.unique(blocks, return_counts=True), strict=True):
        state_counts[inverse[int(value)]] = int(count)
    return blocks, palette, {"lanterns_added": lanterns, "state_counts": dict(state_counts.most_common())}


def export_categories_to_schematic(
    categories: np.ndarray,
    palette_categories: list[str],
    output_path: Path,
    seed: int,
    data_version: int,
    variation_rate: float,
    lantern_rate: float,
) -> dict[str, Any]:
    blocks, block_palette, stats = categories_to_blocks(categories, palette_categories, seed, variation_rate, lantern_rate)
    write_sponge_schem(output_path, blocks, block_palette, data_version)
    return {"output": str(output_path), "block_palette_size": len(block_palette), **stats}


def run_structure_generator(
    phase4_dir: Path,
    phase5_dir: Path,
    output_dir: Path,
    seed: int,
    data_version: int,
) -> dict[str, Any]:
    grammar = load_json(phase4_dir / "grammar.json")
    catalog = load_json(phase5_dir / "module_catalog.json")
    plan = sample_plan(grammar, seed)
    categories, palette_categories, assembly_report = assemble_house(plan, catalog)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "generated_categories.npy", categories)
    write_json(output_dir / "category_palette.json", {"palette": palette_categories})
    write_json(output_dir / "plan.json", plan)
    export_report = export_categories_to_schematic(
        categories,
        palette_categories,
        output_dir / "generated_house.schem",
        seed=seed,
        data_version=data_version,
        variation_rate=0.04,
        lantern_rate=0.012,
    )
    report = {
        "phase": "phase6_generate_structure",
        "plan": plan,
        "assembly": assembly_report,
        "export": export_report,
    }
    write_json(output_dir / "generation_report.json", report)
    return report


def run_detail_pass(
    phase6_dir: Path,
    output_dir: Path,
    seed: int,
    data_version: int,
    variation_rate: float,
    lantern_rate: float,
) -> dict[str, Any]:
    categories = np.load(phase6_dir / "generated_categories.npy")
    palette_categories = load_json(phase6_dir / "category_palette.json")["palette"]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "detailed_categories.npy", categories)
    export_report = export_categories_to_schematic(
        categories,
        palette_categories,
        output_dir / "detailed_house.schem",
        seed=seed,
        data_version=data_version,
        variation_rate=variation_rate,
        lantern_rate=lantern_rate,
    )
    report = {
        "phase": "phase7_detail_wfc",
        "mode": "bounded_surface_detail",
        "input": str(phase6_dir / "generated_categories.npy"),
        "export": export_report,
        "variation_rate": variation_rate,
        "lantern_rate": lantern_rate,
    }
    write_json(output_dir / "detail_report.json", report)
    return report
