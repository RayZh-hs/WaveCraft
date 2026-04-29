#!/usr/bin/env python3
"""Render an HTML inspection report for learned WaveCraft tiles."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


AIR = "air"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def category_family(category: str) -> str:
    if category == AIR:
        return AIR
    return category.split(":", 1)[0]


def category_kind(category: str) -> str:
    if category == AIR:
        return AIR
    without_props = category.split("[", 1)[0]
    if ":" not in without_props:
        return without_props
    return without_props.split(":", 1)[1]


def color_for_category(category: str) -> str:
    if category == AIR:
        return "#f8fafc"
    family = category_family(category)
    kind = category_kind(category)
    family_colors = {
        "wood_dark": "#5b3a29",
        "wood_medium": "#9a6a3a",
        "wood_light": "#d3b06b",
        "wood_red": "#9e3f36",
        "dark_stone": "#334155",
        "stone": "#8a8f93",
        "masonry": "#a45f48",
        "glass": "#8ecae6",
        "cloth": "#e8e2d0",
        "ground": "#6f7f3f",
    }
    base = family_colors.get(family, "#94a3b8")
    if "stripped=true" in category and kind in {"log", "wood", "full"}:
        base = shade_hex(base, 1.24)
    if kind in {"stairs", "slab", "wall", "fence", "fence_gate"}:
        return shade_hex(base, 0.86)
    if kind in {"door", "trapdoor", "pane"}:
        return shade_hex(base, 1.14)
    return base


def shade_hex(color: str, factor: float) -> str:
    color = color.lstrip("#")
    channels = [int(color[i : i + 2], 16) for i in (0, 2, 4)]
    adjusted = [max(0, min(255, int(channel * factor))) for channel in channels]
    return "#" + "".join(f"{channel:02x}" for channel in adjusted)


def contrast_text_color(color: str) -> str:
    color = color.lstrip("#")
    red, green, blue = [int(color[i : i + 2], 16) for i in (0, 2, 4)]
    luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
    return "#111827" if luminance > 0.58 else "#ffffff"


def short_label(category: str) -> str:
    if category == AIR:
        return "air"
    base = category.split("[", 1)[0]
    pieces = base.replace(":", "_").split("_")
    return "".join(piece[0] for piece in pieces if piece)[:4] or "blk"


def parse_category_props(category: str) -> tuple[str, dict[str, str]]:
    if "[" not in category:
        return category, {}
    base, raw_props = category.split("[", 1)
    props = {}
    for item in raw_props.rstrip("]").split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            props[key] = value
    return base, props


def facing_symbol(value: str | None) -> str:
    return {
        "north": "N",
        "east": "E",
        "south": "S",
        "west": "W",
    }.get(value or "", "")


def shape_label(category: str) -> str:
    kind = category_kind(category)
    _, props = parse_category_props(category)
    if category == AIR:
        return ""
    if kind == "stairs":
        half = "^" if props.get("half") == "top" else "v"
        return f"ST{facing_symbol(props.get('facing'))}{half}"
    if kind == "slab":
        slab_type = props.get("type", "")
        if slab_type == "top":
            return "SL^"
        if slab_type == "bottom":
            return "SLv"
        if slab_type == "double":
            return "SL2"
        return "SL"
    if kind == "wall":
        return "WL"
    if kind == "fence":
        return "FN"
    if kind == "fence_gate":
        return f"FG{facing_symbol(props.get('facing'))}"
    if kind == "pane":
        return "PN"
    if kind == "door":
        return f"DR{facing_symbol(props.get('facing'))}"
    if kind == "trapdoor":
        half = "^" if props.get("half") == "top" else "v"
        return f"TD{facing_symbol(props.get('facing'))}{half}"
    if kind in {"log", "wood"}:
        prefix = "S" if props.get("stripped") == "true" else ""
        return f"{prefix}LG{props.get('axis', '')}"
    if kind == "full" and props.get("stripped") == "true":
        return "SFL"
    return short_label(category).upper()


def shape_class_counts(patch: np.ndarray, palette: list[str], air_id: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for value in patch.ravel():
        if int(value) == air_id:
            continue
        category = palette[int(value)]
        counts[category_kind(category)] += 1
    return counts


def overlay_slice_shape(parts: list[str], category: str, x: float, y: float, cell: int) -> None:
    kind = category_kind(category)
    if category == AIR:
        return
    _, props = parse_category_props(category)
    cx = x + cell / 2
    cy = y + cell / 2
    inset = 2
    stroke = "#111827"
    if kind == "slab":
        slab_type = props.get("type", "bottom")
        if slab_type == "top":
            y0 = y + inset
        elif slab_type == "bottom":
            y0 = y + cell / 2
        else:
            y0 = y + inset
        h = cell / 2 - inset if slab_type in {"top", "bottom"} else cell - inset * 2
        parts.append(f'<rect x="{x + inset}" y="{y0}" width="{cell - inset * 2}" height="{h}" fill="#ffffff" fill-opacity=".42" stroke="{stroke}" stroke-opacity=".55"/>')
    elif kind == "stairs":
        facing = props.get("facing", "north")
        if facing == "north":
            pts = f"{x+inset},{y+inset} {x+cell-inset},{y+inset} {x+inset},{y+cell-inset}"
        elif facing == "south":
            pts = f"{x+inset},{y+cell-inset} {x+cell-inset},{y+cell-inset} {x+cell-inset},{y+inset}"
        elif facing == "east":
            pts = f"{x+cell-inset},{y+inset} {x+cell-inset},{y+cell-inset} {x+inset},{y+inset}"
        else:
            pts = f"{x+inset},{y+inset} {x+inset},{y+cell-inset} {x+cell-inset},{y+cell-inset}"
        parts.append(f'<polygon points="{pts}" fill="#ffffff" fill-opacity=".46" stroke="{stroke}" stroke-opacity=".65"/>')
        if props.get("half") == "top":
            parts.append(f'<line x1="{x+inset}" y1="{y+inset}" x2="{x+cell-inset}" y2="{y+cell-inset}" stroke="{stroke}" stroke-opacity=".45"/>')
    elif kind in {"wall", "fence", "pane"}:
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{cell * 0.18}" fill="#ffffff" fill-opacity=".7" stroke="{stroke}" stroke-opacity=".5"/>')
        for direction, (x2, y2) in {
            "north": (cx, y + inset),
            "south": (cx, y + cell - inset),
            "east": (x + cell - inset, cy),
            "west": (x + inset, cy),
        }.items():
            if props.get(direction) not in {None, "false", "none"}:
                parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x2}" y2="{y2}" stroke="#ffffff" stroke-width="2.2" stroke-opacity=".85"/>')
                parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width=".7" stroke-opacity=".55"/>')
    elif kind in {"door", "trapdoor", "fence_gate"}:
        facing = props.get("facing", "north")
        if facing in {"north", "south"}:
            parts.append(f'<rect x="{x+inset}" y="{cy-2}" width="{cell-inset*2}" height="4" fill="#ffffff" fill-opacity=".7" stroke="{stroke}" stroke-opacity=".45"/>')
        else:
            parts.append(f'<rect x="{cx-2}" y="{y+inset}" width="4" height="{cell-inset*2}" fill="#ffffff" fill-opacity=".7" stroke="{stroke}" stroke-opacity=".45"/>')
    elif kind in {"log", "wood"}:
        axis = props.get("axis", "y")
        if axis == "x":
            parts.append(f'<line x1="{x+inset}" y1="{cy}" x2="{x+cell-inset}" y2="{cy}" stroke="#ffffff" stroke-width="2.2" stroke-opacity=".65"/>')
        elif axis == "z":
            parts.append(f'<line x1="{cx}" y1="{y+inset}" x2="{cx}" y2="{y+cell-inset}" stroke="#ffffff" stroke-width="2.2" stroke-opacity=".65"/>')
        else:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{cell * 0.24}" fill="none" stroke="#ffffff" stroke-width="1.7" stroke-opacity=".72"/>')
        if props.get("stripped") == "true":
            parts.append(f'<line x1="{x+inset}" y1="{y+inset}" x2="{x+cell-inset}" y2="{y+cell-inset}" stroke="#111827" stroke-width="1.5" stroke-opacity=".6"/>')
            parts.append(f'<line x1="{x+inset}" y1="{y+cell-inset}" x2="{x+cell-inset}" y2="{y+inset}" stroke="#ffffff" stroke-width="1.1" stroke-opacity=".65"/>')
    elif kind == "full" and props.get("stripped") == "true":
        parts.append(f'<line x1="{x+inset}" y1="{y+inset}" x2="{x+cell-inset}" y2="{y+cell-inset}" stroke="#111827" stroke-width="1.5" stroke-opacity=".6"/>')
        parts.append(f'<line x1="{x+inset}" y1="{y+cell-inset}" x2="{x+cell-inset}" y2="{y+inset}" stroke="#ffffff" stroke-width="1.1" stroke-opacity=".65"/>')


def patch_bounds(patch: np.ndarray, air_id: int) -> dict[str, Any]:
    positions = np.argwhere(patch != air_id)
    if len(positions) == 0:
        return {"empty": True}
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    return {
        "empty": False,
        "min_yzx": [int(value) for value in mins],
        "max_yzx": [int(value) for value in maxs],
        "span_yzx": [int(value) for value in (maxs - mins + 1)],
    }


def face_stats(patch: np.ndarray, palette: list[str], air_id: int) -> dict[str, Any]:
    faces = {
        "-y": patch[0, :, :],
        "+y": patch[-1, :, :],
        "-z": patch[:, 0, :],
        "+z": patch[:, -1, :],
        "-x": patch[:, :, 0],
        "+x": patch[:, :, -1],
    }
    results: dict[str, Any] = {}
    for name, face in faces.items():
        values, counts = np.unique(face, return_counts=True)
        total = int(face.size)
        non_air = int(total - counts[values == air_id].sum()) if np.any(values == air_id) else total
        top = []
        for value, count in sorted(zip(values, counts), key=lambda item: int(item[1]), reverse=True)[:5]:
            top.append({"category": palette[int(value)], "count": int(count), "share": float(count / total)})
        results[name] = {
            "non_air": non_air,
            "non_air_fraction": float(non_air / total),
            "top": top,
        }
    return results


def exposed_contact_score(patch: np.ndarray, air_id: int) -> int:
    score = 0
    score += int(np.count_nonzero(patch[0, :, :] != air_id))
    score += int(np.count_nonzero(patch[-1, :, :] != air_id))
    score += int(np.count_nonzero(patch[:, 0, :] != air_id))
    score += int(np.count_nonzero(patch[:, -1, :] != air_id))
    score += int(np.count_nonzero(patch[:, :, 0] != air_id))
    score += int(np.count_nonzero(patch[:, :, -1] != air_id))
    return score


def svg_single_layer(patch: np.ndarray, palette: list[str], tile_id: int, layer_y: int, active: bool) -> str:
    cell = 44
    width = patch.shape[2] * cell
    height = patch.shape[1] * cell
    active_class = " active" if active else ""
    parts = [
        f'<svg class="slice-layer{active_class}" data-layer="{layer_y}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Tile {tile_id} y={layer_y} layer">',
    ]
    for z in range(patch.shape[1]):
        for x in range(patch.shape[2]):
            category = palette[int(patch[layer_y, z, x])]
            color = color_for_category(category)
            opacity = "0.12" if category == AIR else "1"
            stroke = "#e5e7eb" if category == AIR else "#111827"
            px = x * cell
            py = z * cell
            parts.append(
                f'<rect x="{px}" y="{py}" width="{cell}" height="{cell}" '
                f'fill="{color}" opacity="{opacity}" stroke="{stroke}" stroke-opacity=".24">'
                f'<title>{html.escape(category)}</title></rect>'
            )
            overlay_slice_shape(parts, category, px, py, cell)
            label = shape_label(category)
            if label:
                text_color = contrast_text_color(color)
                parts.append(
                    f'<text class="cell-label" x="{px + cell / 2}" y="{py + cell / 2 + 5}" '
                    f'text-anchor="middle" fill="{text_color}">{html.escape(label)}</text>'
                )
    parts.append("</svg>")
    return "\n".join(parts)


def layer_non_air_counts(patch: np.ndarray, air_id: int) -> list[int]:
    return [int(np.count_nonzero(patch[y] != air_id)) for y in range(patch.shape[0])]


def default_layer_index(patch: np.ndarray, air_id: int) -> int:
    counts = layer_non_air_counts(patch, air_id)
    if not any(counts):
        return 0
    return int(max(range(len(counts)), key=lambda index: counts[index]))


def svg_layer_viewer(patch: np.ndarray, palette: list[str], air_id: int, tile_id: int) -> str:
    counts = layer_non_air_counts(patch, air_id)
    active_layer = default_layer_index(patch, air_id)
    buttons = []
    layers = []
    for y, count in enumerate(counts):
        active = y == active_layer
        active_class = " active" if active else ""
        buttons.append(
            f'<button type="button" class="layer-button{active_class}" data-layer="{y}" '
            f'title="{count} non-air voxels in layer y={y}">y={y}<span>{count}</span></button>'
        )
        layers.append(svg_single_layer(patch, palette, tile_id, y, active))
    return (
        '<div class="layer-viewer">'
        '<div class="layer-viewer-head"><strong>Layer slice</strong><span>choose Y level; number is non-air cells</span></div>'
        f'<div class="layer-buttons">{"".join(buttons)}</div>'
        f'<div class="layer-stage">{"".join(layers)}</div>'
        '</div>'
    )


def block_height_and_offset(category: str) -> tuple[float, float]:
    kind = category_kind(category)
    _, props = parse_category_props(category)
    if kind == "slab":
        slab_type = props.get("type", "bottom")
        if slab_type == "double":
            return 1.0, 0.0
        if slab_type == "top":
            return 0.5, 0.0
        return 0.5, 0.5
    if kind == "trapdoor":
        return 0.16, 0.0 if props.get("half") == "top" else 0.84
    if kind in {"pane", "fence", "fence_gate", "wall"}:
        return 0.85, 0.12
    return 1.0, 0.0


def iso_cube_parts(sx: float, sy: float, scale: int, category: str) -> list[str]:
    height_factor, top_offset = block_height_and_offset(category)
    top_sy = sy + scale * top_offset
    side_h = scale * 1.35 * height_factor
    base = color_for_category(category)
    top = shade_hex(base, 1.18)
    left = shade_hex(base, 0.84)
    right = shade_hex(base, 0.98)
    p_top = f"{sx},{top_sy} {sx + scale},{top_sy + scale * 0.5} {sx},{top_sy + scale} {sx - scale},{top_sy + scale * 0.5}"
    p_left = f"{sx - scale},{top_sy + scale * 0.5} {sx},{top_sy + scale} {sx},{top_sy + scale + side_h} {sx - scale},{top_sy + scale * 0.5 + side_h}"
    p_right = f"{sx + scale},{top_sy + scale * 0.5} {sx},{top_sy + scale} {sx},{top_sy + scale + side_h} {sx + scale},{top_sy + scale * 0.5 + side_h}"
    title = html.escape(category)
    return [
        f'<polygon points="{p_left}" fill="{left}" stroke="#111827" stroke-opacity=".24"><title>{title}</title></polygon>',
        f'<polygon points="{p_right}" fill="{right}" stroke="#111827" stroke-opacity=".24"><title>{title}</title></polygon>',
        f'<polygon points="{p_top}" fill="{top}" stroke="#111827" stroke-opacity=".24"><title>{title}</title></polygon>',
    ]


def iso_shape_overlay(sx: float, sy: float, scale: int, category: str) -> str:
    kind = category_kind(category)
    _, props = parse_category_props(category)
    if (kind == "full" and props.get("stripped") != "true") or category == AIR:
        return ""
    label = shape_label(category)
    top_sy = sy + scale * block_height_and_offset(category)[1]
    text_y = top_sy + scale * 0.78
    text_color = contrast_text_color(color_for_category(category))
    overlays = []
    if kind == "stairs":
        facing = props.get("facing", "north")
        if facing in {"north", "south"}:
            overlays.append(f'<line x1="{sx-scale*.55}" y1="{top_sy+scale*.52}" x2="{sx+scale*.55}" y2="{top_sy+scale*.52}" stroke="#ffffff" stroke-width="2.2" stroke-opacity=".9"/>')
        else:
            overlays.append(f'<line x1="{sx}" y1="{top_sy+scale*.18}" x2="{sx}" y2="{top_sy+scale*.92}" stroke="#ffffff" stroke-width="2.2" stroke-opacity=".9"/>')
    elif kind in {"pane", "wall", "fence", "fence_gate"}:
        overlays.append(f'<circle cx="{sx}" cy="{top_sy+scale*.55}" r="{scale*.18}" fill="#ffffff" fill-opacity=".78" stroke="#111827" stroke-opacity=".35"/>')
    elif kind in {"door", "trapdoor"}:
        overlays.append(f'<rect x="{sx-scale*.42}" y="{top_sy+scale*.38}" width="{scale*.84}" height="{scale*.26}" fill="#ffffff" fill-opacity=".76" stroke="#111827" stroke-opacity=".35"/>')
    elif kind in {"log", "wood"}:
        overlays.append(f'<circle cx="{sx}" cy="{top_sy+scale*.55}" r="{scale*.22}" fill="none" stroke="#ffffff" stroke-width="2" stroke-opacity=".85"/>')
        if props.get("stripped") == "true":
            overlays.append(f'<line x1="{sx-scale*.5}" y1="{top_sy+scale*.28}" x2="{sx+scale*.5}" y2="{top_sy+scale*.82}" stroke="#111827" stroke-width="1.5" stroke-opacity=".62"/>')
            overlays.append(f'<line x1="{sx-scale*.5}" y1="{top_sy+scale*.82}" x2="{sx+scale*.5}" y2="{top_sy+scale*.28}" stroke="#ffffff" stroke-width="1.1" stroke-opacity=".75"/>')
    elif kind == "full" and props.get("stripped") == "true":
        overlays.append(f'<line x1="{sx-scale*.5}" y1="{top_sy+scale*.28}" x2="{sx+scale*.5}" y2="{top_sy+scale*.82}" stroke="#111827" stroke-width="1.5" stroke-opacity=".62"/>')
        overlays.append(f'<line x1="{sx-scale*.5}" y1="{top_sy+scale*.82}" x2="{sx+scale*.5}" y2="{top_sy+scale*.28}" stroke="#ffffff" stroke-width="1.1" stroke-opacity=".75"/>')
    overlays.append(
        f'<text class="iso-label" x="{sx}" y="{text_y}" text-anchor="middle" fill="{text_color}">{html.escape(label)}</text>'
    )
    return "".join(overlays)


def svg_isometric(patch: np.ndarray, palette: list[str], air_id: int, tile_id: int) -> str:
    # Draw far-to-near cubes in an isometric projection. This is an inspection
    # view, not a physically exact Minecraft renderer.
    scale = 16
    ox = 92
    oy = 84
    width = 205
    height = 225
    cubes = []
    for y in range(patch.shape[0]):
        for z in range(patch.shape[1]):
            for x in range(patch.shape[2]):
                value = int(patch[y, z, x])
                if value == air_id:
                    continue
                sx = ox + (x - z) * scale
                sy = oy + (x + z) * scale * 0.5 - y * scale
                cubes.append((x + z + y, sx, sy, palette[value]))
    cubes.sort()

    parts = [
        f'<svg class="iso" viewBox="0 0 {width} {height}" role="img" aria-label="Tile {tile_id} isometric view">',
    ]
    for _, sx, sy, category in cubes:
        parts.extend(iso_cube_parts(sx, sy, scale, category))
        parts.append(iso_shape_overlay(sx, sy, scale, category))
    parts.append("</svg>")
    return "\n".join(parts)


def table_rows(rows: list[list[Any]]) -> str:
    return "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )


def analyze_tiles(
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
    tile_library: Any,
    source_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    palette = metadata["palette"]
    air_id = int(metadata["air_id"])
    prototypes = tile_library["prototypes"]
    labels = tile_library["labels"]
    cluster_summaries = evaluation.get("cluster_summaries", [])
    by_cluster_summary = {int(item["cluster_id"]): item for item in cluster_summaries}
    cluster_sizes = Counter(int(label) for label in labels)

    tiles = []
    for tile_index, patch in enumerate(prototypes):
        cluster_id = int(cluster_summaries[tile_index]["cluster_id"]) if tile_index < len(cluster_summaries) else tile_index
        indices = np.flatnonzero(labels == cluster_id)
        source_counts = Counter(source_rows[int(index)]["source"] for index in indices)
        transform_counts = Counter(source_rows[int(index)]["transform"] for index in indices)
        semantic_bucket_counts = Counter(source_rows[int(index)].get("semantic_bucket", "unknown") for index in indices)
        values, counts = np.unique(patch, return_counts=True)
        total = int(patch.size)
        composition = [
            {
                "category": palette[int(value)],
                "count": int(count),
                "share": float(count / total),
                "family": category_family(palette[int(value)]),
                "kind": category_kind(palette[int(value)]),
            }
            for value, count in sorted(zip(values, counts), key=lambda item: int(item[1]), reverse=True)
        ]
        family_counts = Counter()
        kind_counts = Counter()
        for item in composition:
            family_counts[item["family"]] += item["count"]
            kind_counts[item["kind"]] += item["count"]

        non_air = int(total - sum(item["count"] for item in composition if item["category"] == AIR))
        dominant_non_air = next((item for item in composition if item["category"] != AIR), None)
        max_source = source_counts.most_common(1)[0] if source_counts else ("n/a", 0)
        summary = by_cluster_summary.get(cluster_id, {})
        medoid_index = int(summary.get("medoid_patch_index", -1))
        medoid_source = source_rows[medoid_index] if 0 <= medoid_index < len(source_rows) else {}

        flags = []
        if non_air == total and len([item for item in composition if item["category"] != AIR]) == 1:
            flags.append("solid-single-material")
        if non_air / total < 0.35:
            flags.append("sparse")
        if source_counts and max_source[1] / max(cluster_sizes[cluster_id], 1) >= 0.80:
            flags.append("source-specific")
        if exposed_contact_score(patch, air_id) < 40:
            flags.append("low-boundary-contact")
        shape_counts = shape_class_counts(patch, palette, air_id)

        tiles.append(
            {
                "tile_index": tile_index,
                "cluster_id": cluster_id,
                "cluster_size": int(cluster_sizes[cluster_id]),
                "air_fraction": float((patch == air_id).mean()),
                "non_air": non_air,
                "unique_categories": int(len(values)),
                "composition": composition[:12],
                "family_counts": dict(family_counts.most_common()),
                "kind_counts": dict(kind_counts.most_common()),
                "shape_counts": dict(shape_counts.most_common()),
                "dominant_non_air": dominant_non_air,
                "source_counts": dict(source_counts.most_common()),
                "transform_counts": dict(transform_counts.most_common()),
                "semantic_bucket_counts": dict(semantic_bucket_counts.most_common()),
                "top_source": {"source": max_source[0], "count": int(max_source[1])},
                "medoid_source": medoid_source,
                "bounds": patch_bounds(patch, air_id),
                "face_stats": face_stats(patch, palette, air_id),
                "boundary_contact_score": exposed_contact_score(patch, air_id),
                "flags": flags,
            }
        )

    global_flags = Counter(flag for tile in tiles for flag in tile["flags"])
    global_shapes = Counter()
    for tile in tiles:
        global_shapes.update(tile["shape_counts"])
    global_summary = {
        "tile_count": len(tiles),
        "total_clustered_patches": int(len(labels)),
        "prototype_shape": [int(value) for value in prototypes.shape[1:]],
        "average_air_fraction": float(np.mean([tile["air_fraction"] for tile in tiles])),
        "flag_counts": dict(global_flags.most_common()),
        "shape_counts": dict(global_shapes.most_common()),
        "largest_clusters": [
            {"cluster_id": int(cluster_id), "size": int(size)}
            for cluster_id, size in cluster_sizes.most_common(10)
        ],
        "feature_info": evaluation.get("feature_info", {}),
        "heldout_evaluation": evaluation.get("heldout_evaluation", []),
        "k_evaluation": evaluation.get("k_evaluation", []),
        "source_counts": evaluation.get("patch_stats", {}).get("actual_source_counts", {}),
        "semantic_bucket_counts": evaluation.get("patch_stats", {}).get("actual_bucket_counts", {}),
    }
    return tiles, global_summary


def render_metric_cards(summary: dict[str, Any], evaluation: dict[str, Any], metadata: dict[str, Any]) -> str:
    best_k = max(
        evaluation.get("k_evaluation", []),
        key=lambda item: item["silhouette"] if item.get("silhouette") is not None else -1,
    )
    cards = [
        ("Tiles", summary["tile_count"]),
        ("Patch Sample", f"{summary['total_clustered_patches']:,}"),
        ("Palette", len(metadata["palette"])),
        ("Best k", f"{best_k['k']} ({best_k['silhouette']:.3f})"),
        ("Avg Air", f"{summary['average_air_fraction']:.1%}"),
        ("SVD Var", f"{summary['feature_info'].get('svd_explained_variance_ratio_sum', 0):.3f}"),
    ]
    return "\n".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'
        for label, value in cards
    )


def render_tile_card(tile: dict[str, Any], patch: np.ndarray, palette: list[str], air_id: int) -> str:
    top_comp = tile["composition"][:6]
    comp_rows = []
    for item in top_comp:
        category = item["category"]
        color = color_for_category(category)
        comp_rows.append(
            "<tr>"
            f'<td><span class="swatch" style="background:{color}"></span>{html.escape(category)}</td>'
            f"<td>{item['count']}</td>"
            f"<td>{item['share']:.1%}</td>"
            "</tr>"
        )

    face_rows = []
    for face_name, face in tile["face_stats"].items():
        face_rows.append([face_name, f"{face['non_air_fraction']:.0%}", face["top"][0]["category"]])
    shape_rows = table_rows([[shape, count] for shape, count in list(tile["shape_counts"].items())[:8]])

    flags = "".join(f'<span class="flag">{html.escape(flag)}</span>' for flag in tile["flags"]) or '<span class="flag quiet">ok</span>'
    source_rows = table_rows([[source, count] for source, count in list(tile["source_counts"].items())[:5]])
    transform_rows = table_rows([[transform, count] for transform, count in tile["transform_counts"].items()])
    bucket_rows = table_rows([[bucket, count] for bucket, count in tile["semantic_bucket_counts"].items()])
    medoid = tile["medoid_source"]
    medoid_text = "n/a"
    if medoid:
        medoid_text = f"{medoid.get('source')} / {medoid.get('transform')} @ y={medoid.get('y')} z={medoid.get('z')} x={medoid.get('x')}"

    return f"""
<section class="tile-card" id="tile-{tile['tile_index']:03d}" data-flags="{html.escape(' '.join(tile['flags']))}">
  <header>
    <div>
      <h2>Tile {tile['tile_index']:03d}</h2>
      <p>Cluster {tile['cluster_id']} · {tile['cluster_size']} patches · {tile['non_air']}/125 non-air · contact {tile['boundary_contact_score']}</p>
    </div>
    <div class="flags">{flags}</div>
  </header>
  <div class="tile-layout">
    <div class="render-block">
      {svg_isometric(patch, palette, air_id, tile['tile_index'])}
      {svg_layer_viewer(patch, palette, air_id, tile['tile_index'])}
    </div>
    <div class="tile-data">
      <div class="stat-grid">
        <div><span>Air</span><strong>{tile['air_fraction']:.1%}</strong></div>
        <div><span>Unique</span><strong>{tile['unique_categories']}</strong></div>
        <div><span>Top Source</span><strong>{html.escape(str(tile['top_source']['source']))}</strong></div>
        <div><span>Medoid</span><strong>{html.escape(medoid_text)}</strong></div>
      </div>
      <h3>Composition</h3>
      <table><thead><tr><th>Category</th><th>Count</th><th>Share</th></tr></thead><tbody>{''.join(comp_rows)}</tbody></table>
      <h3>Faces</h3>
      <table><thead><tr><th>Face</th><th>Non-Air</th><th>Dominant</th></tr></thead><tbody>{table_rows(face_rows)}</tbody></table>
      <h3>Shape Counts</h3>
      <table><thead><tr><th>Shape</th><th>Count</th></tr></thead><tbody>{shape_rows}</tbody></table>
      <details>
        <summary>Source and transform counts</summary>
        <div class="two-tables">
          <table><thead><tr><th>Source</th><th>Count</th></tr></thead><tbody>{source_rows}</tbody></table>
          <table><thead><tr><th>Transform</th><th>Count</th></tr></thead><tbody>{transform_rows}</tbody></table>
        </div>
        <h3>Semantic Buckets</h3>
        <table><thead><tr><th>Bucket</th><th>Count</th></tr></thead><tbody>{bucket_rows}</tbody></table>
      </details>
    </div>
  </div>
</section>
"""


def render_html(
    output_path: Path,
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
    tiles: list[dict[str, Any]],
    summary: dict[str, Any],
    prototypes: np.ndarray,
) -> None:
    palette = metadata["palette"]
    air_id = int(metadata["air_id"])
    flag_counts = summary["flag_counts"]
    heldout_rows = table_rows(
        [
            [
                item["heldout_source"],
                item["heldout_patches"],
                f"{item['heldout_to_train_distance_ratio']:.3f}",
            ]
            for item in summary["heldout_evaluation"]
        ]
    )
    k_rows = table_rows(
        [
            [
                item["k"],
                f"{item['inertia']:.2f}",
                "n/a" if item.get("silhouette") is None else f"{item['silhouette']:.4f}",
            ]
            for item in summary["k_evaluation"]
        ]
    )
    tile_cards = "\n".join(render_tile_card(tile, prototypes[tile["tile_index"]], palette, air_id) for tile in tiles)
    source_rows = table_rows([[source, count] for source, count in summary["source_counts"].items()])
    bucket_rows = table_rows([[bucket, count] for bucket, count in summary["semantic_bucket_counts"].items()])
    shape_rows = table_rows([[shape, count] for shape, count in list(summary["shape_counts"].items())[:12]])
    flag_text = ", ".join(f"{flag}: {count}" for flag, count in flag_counts.items()) or "none"

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WaveCraft Tile Inspector</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #111827;
      --muted: #64748b;
      --line: #d8dee9;
      --panel: #ffffff;
      --band: #f4f6f8;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--band);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header.page {{
      padding: 24px 28px 18px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    h2 {{ margin: 0; font-size: 18px; }}
    h3 {{ margin: 16px 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }}
    p {{ margin: 0; color: var(--muted); }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto auto;
      gap: 12px;
      margin-top: 16px;
      align-items: center;
    }}
    .toggle-labels {{
      height: 36px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #ffffff;
      color: #334155;
      font-size: 13px;
      white-space: nowrap;
    }}
    .toggle-labels input {{ width: 16px; height: 16px; padding: 0; }}
    input, select {{
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #ffffff;
      font: inherit;
    }}
    main {{ padding: 20px 28px 40px; max-width: 1480px; margin: 0 auto; }}
    .metrics {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin-bottom: 18px; }}
    .metric, .panel, .tile-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{ padding: 12px; min-height: 74px; }}
    .metric span, .stat-grid span {{ display:block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display:block; margin-top: 8px; font-size: 20px; }}
    .panel {{ padding: 16px; margin-bottom: 18px; }}
    .panel-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }}
    .tile-card {{ padding: 16px; margin: 16px 0; }}
    .tile-card > header {{ display: flex; justify-content: space-between; gap: 12px; margin-bottom: 14px; }}
    .flags {{ display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }}
    .flag {{ display: inline-flex; align-items: center; height: 24px; padding: 0 8px; border-radius: 999px; background: #e0ecff; color: #1e3a8a; font-size: 12px; white-space: nowrap; }}
    .flag.quiet {{ background: #ecfdf5; color: #166534; }}
    .tile-layout {{ display: grid; grid-template-columns: minmax(360px, .95fr) minmax(420px, 1.05fr); gap: 18px; align-items: start; }}
    .render-block {{ display: grid; grid-template-columns: 270px minmax(260px, 1fr); gap: 14px; align-items: start; overflow-x: auto; }}
    .iso {{ width: 260px; height: 210px; display: block; background: #fbfdff; border: 1px solid var(--line); border-radius: 6px; }}
    .layer-viewer {{ min-width: 270px; }}
    .layer-viewer-head {{ display: flex; justify-content: space-between; gap: 8px; align-items: baseline; margin-bottom: 8px; }}
    .layer-viewer-head strong {{ font-size: 13px; }}
    .layer-viewer-head span {{ color: var(--muted); font-size: 11px; }}
    .layer-buttons {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 5px; margin-bottom: 8px; }}
    .layer-button {{
      border: 1px solid var(--line);
      border-radius: 6px;
      min-height: 40px;
      background: #ffffff;
      color: #334155;
      cursor: pointer;
      font: inherit;
      font-size: 12px;
      padding: 4px;
    }}
    .layer-button span {{ display: block; color: var(--muted); font-size: 10px; }}
    .layer-button.active {{ border-color: var(--accent); background: #dbeafe; color: #1e3a8a; }}
    .layer-stage {{
      width: 100%;
      max-width: 360px;
      aspect-ratio: 1 / 1;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfdff;
      overflow: hidden;
    }}
    .slice-layer {{ display: none; width: 100%; height: 100%; }}
    .slice-layer.active {{ display: block; }}
    .cell-label, .iso-label {{ display: none; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 700; font-size: 10px; paint-order: stroke; stroke: #00000099; stroke-width: 1.6px; }}
    .show-labels .cell-label, .show-labels .iso-label {{ display: block; }}
    .iso-label {{ font-size: 6px; stroke-width: .8px; }}
    .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .stat-grid div {{ padding: 10px; border: 1px solid var(--line); border-radius: 6px; background: #fbfdff; }}
    .stat-grid strong {{ display:block; margin-top: 4px; font-size: 13px; overflow-wrap: anywhere; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 4px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .swatch {{ display: inline-block; width: 12px; height: 12px; border: 1px solid #11182733; margin-right: 6px; vertical-align: -2px; }}
    .two-tables {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .legend span {{ display: inline-flex; gap: 4px; align-items: center; min-height: 26px; border: 1px solid var(--line); border-radius: 6px; padding: 3px 8px; background: #fbfdff; color: #334155; font-size: 12px; }}
    details summary {{ cursor: pointer; color: var(--accent); margin-top: 12px; }}
    .hidden {{ display: none; }}
    @media (max-width: 980px) {{
      header.page {{ position: static; }}
      .metrics, .panel-grid, .tile-layout, .toolbar {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header class="page">
    <h1>WaveCraft Tile Inspector</h1>
    <p>Generated from Phase I/II artifacts. Use this report to review medoid tiles before WFC rule mining.</p>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Filter by tile id, category, source, or flag">
      <select id="flagFilter" aria-label="Flag filter">
        <option value="">All flags</option>
        <option value="source-specific">Source-specific</option>
        <option value="solid-single-material">Solid single material</option>
        <option value="sparse">Sparse</option>
        <option value="low-boundary-contact">Low boundary contact</option>
      </select>
      <label class="toggle-labels"><input id="labelToggle" type="checkbox"> Show cell labels</label>
    </div>
  </header>
  <main>
    <section class="metrics">{render_metric_cards(summary, evaluation, metadata)}</section>
    <section class="panel">
      <h2>Run Analysis</h2>
      <p>Flags: {html.escape(flag_text)}</p>
      <div class="panel-grid">
        <div>
          <h3>Source Sample Counts</h3>
          <table><thead><tr><th>Source</th><th>Count</th></tr></thead><tbody>{source_rows}</tbody></table>
        </div>
        <div>
          <h3>k Evaluation</h3>
          <table><thead><tr><th>k</th><th>Inertia</th><th>Silhouette</th></tr></thead><tbody>{k_rows}</tbody></table>
        </div>
        <div>
          <h3>Held-Out Sources</h3>
          <table><thead><tr><th>Source</th><th>Patches</th><th>Distance Ratio</th></tr></thead><tbody>{heldout_rows}</tbody></table>
        </div>
        <div>
          <h3>Semantic Buckets</h3>
          <table><thead><tr><th>Bucket</th><th>Count</th></tr></thead><tbody>{bucket_rows}</tbody></table>
        </div>
      </div>
      <h3>Prototype Shape Counts</h3>
      <table><thead><tr><th>Shape</th><th>Count</th></tr></thead><tbody>{shape_rows}</tbody></table>
      <div class="legend">
        <span><strong>ST</strong> stairs, suffix N/E/S/W is facing, ^ top half, v bottom half</span>
        <span><strong>SL</strong> slab, ^ top, v bottom, 2 double</span>
        <span><strong>WL/FN/PN</strong> wall, fence, pane with connection spokes</span>
        <span><strong>DR/TD/FG</strong> door, trapdoor, fence gate</span>
        <span><strong>LGx/LGy/LGz</strong> log axis; S prefix marks stripped</span>
      </div>
    </section>
    {tile_cards}
  </main>
  <script>
    const search = document.getElementById('search');
    const flagFilter = document.getElementById('flagFilter');
    const labelToggle = document.getElementById('labelToggle');
    const cards = Array.from(document.querySelectorAll('.tile-card'));
    function applyFilters() {{
      const needle = search.value.trim().toLowerCase();
      const flag = flagFilter.value;
      for (const card of cards) {{
        const textMatch = !needle || card.innerText.toLowerCase().includes(needle) || card.id.includes(needle);
        const flagMatch = !flag || card.dataset.flags.includes(flag);
        card.classList.toggle('hidden', !(textMatch && flagMatch));
      }}
    }}
    search.addEventListener('input', applyFilters);
    flagFilter.addEventListener('change', applyFilters);
    labelToggle.addEventListener('change', () => {{
      document.body.classList.toggle('show-labels', labelToggle.checked);
    }});
    document.querySelectorAll('.tile-card').forEach((card) => {{
      card.querySelectorAll('.layer-button').forEach((button) => {{
        button.addEventListener('click', () => {{
          const layer = button.dataset.layer;
          card.querySelectorAll('.layer-button').forEach((item) => item.classList.toggle('active', item === button));
          card.querySelectorAll('.slice-layer').forEach((item) => item.classList.toggle('active', item.dataset.layer === layer));
        }});
      }});
    }});
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", type=Path, default=Path("datasets/phase1"))
    parser.add_argument("--phase2-dir", type=Path, default=Path("datasets/phase2"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/phase2/inspection"))
    args = parser.parse_args()

    metadata = load_json(args.phase1_dir / "metadata.json")
    evaluation = load_json(args.phase2_dir / "evaluation.json")
    tile_library = np.load(args.phase2_dir / "tile_library.npz")
    source_rows = list(csv.DictReader((args.phase2_dir / "sampled_patch_sources.csv").open(encoding="utf-8")))

    tiles, summary = analyze_tiles(metadata, evaluation, tile_library, source_rows)
    output_path = args.output_dir / "index.html"
    data_path = args.output_dir / "tile_analysis.json"
    render_html(output_path, metadata, evaluation, tiles, summary, tile_library["prototypes"])
    write_json(data_path, {"summary": summary, "tiles": tiles})
    print(f"Wrote tile inspector to {output_path}")
    print(f"Wrote tile analysis data to {data_path}")


if __name__ == "__main__":
    main()
