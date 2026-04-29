#!/usr/bin/env python3
"""Phase I: load schematics, reduce palettes, mask ornaments, and augment."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbtlib
import numpy as np


AIR = "air"
ORNAMENT = "ornament"

DIRECTIONS = {"north", "east", "south", "west"}
TRANSFORM_DIRECTION_MAPS = {
    "original": {"north": "north", "east": "east", "south": "south", "west": "west"},
    "mirror_x": {"north": "north", "east": "west", "south": "south", "west": "east"},
    "mirror_z": {"north": "south", "east": "east", "south": "north", "west": "west"},
    "rot90_y": {"north": "west", "east": "north", "south": "east", "west": "south"},
}

DECORATIVE_NAME_TERMS = (
    "allium",
    "azure_bluet",
    "blue_orchid",
    "cornflower",
    "daisy",
    "dead_bush",
    "dandelion",
    "fern",
    "flower",
    "grass",
    "lilac",
    "lily",
    "orchid",
    "peony",
    "poppy",
    "rose",
    "sapling",
    "sunflower",
    "tulip",
    "vine",
)


@dataclass(frozen=True)
class LoadedSchematic:
    name: str
    shape_yzx: tuple[int, int, int]
    block_ids: np.ndarray
    palette_by_id: dict[int, str]


def plain(value: Any) -> Any:
    if hasattr(value, "unpack"):
        return value.unpack()
    return value


def get_root(nbt_file: Any) -> Any:
    if hasattr(nbt_file, "root"):
        return nbt_file.root
    if hasattr(nbt_file, "root_name"):
        root_name = nbt_file.root_name
        if root_name:
            return nbt_file[root_name]
        if len(nbt_file) == 1:
            return next(iter(nbt_file.values()))
    return nbt_file


def decode_varints(data: bytes, expected: int) -> np.ndarray:
    values: list[int] = []
    value = 0
    shift = 0
    for byte in data:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            continue
        values.append(value)
        value = 0
        shift = 0
        if len(values) == expected:
            break
    if len(values) != expected:
        raise ValueError(f"Decoded {len(values)} block IDs, expected {expected}")
    return np.asarray(values, dtype=np.int32)


def load_schematic(path: Path) -> LoadedSchematic:
    root = get_root(nbtlib.load(path))
    width = int(plain(root["Width"]))
    height = int(plain(root["Height"]))
    length = int(plain(root["Length"]))
    volume = width * height * length

    blocks_tag = root.get("Blocks", root)
    palette_tag = blocks_tag["Palette"]
    palette_by_id = {int(plain(value)): str(key) for key, value in palette_tag.items()}

    data_tag = blocks_tag.get("BlockData", blocks_tag.get("Data"))
    raw_values = [int(item) & 0xFF for item in data_tag]
    if len(raw_values) == volume:
        flat_ids = np.asarray(raw_values, dtype=np.int32)
    else:
        flat_ids = decode_varints(bytes(raw_values), volume)
    # Sponge schematic order is x + z * width + y * width * length.
    block_ids = flat_ids.reshape((height, length, width))
    return LoadedSchematic(path.stem, (height, length, width), block_ids, palette_by_id)


def parse_block_state(state: str) -> tuple[str, dict[str, str]]:
    if "[" not in state:
        return state, {}
    base, raw_props = state.split("[", 1)
    props = {}
    for item in raw_props.rstrip("]").split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            props[key] = value
    return base, props


def format_props(props: dict[str, str]) -> str:
    if not props:
        return ""
    return "[" + ",".join(f"{key}={props[key]}" for key in sorted(props)) + "]"


def parse_category(category: str) -> tuple[str, dict[str, str]]:
    if "[" not in category:
        return category, {}
    base, raw_props = category.split("[", 1)
    props = {}
    for item in raw_props.rstrip("]").split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            props[key] = value
    return base, props


def material_family(block_name: str) -> str:
    if block_name in {"air", "cave_air", "void_air"}:
        return AIR

    dark_stone_terms = (
        "deepslate",
        "blackstone",
        "basalt",
        "tuff",
        "polished_blackstone",
    )
    if any(term in block_name for term in dark_stone_terms):
        return "dark_stone"

    stone_terms = (
        "stone",
        "cobblestone",
        "andesite",
        "diorite",
        "granite",
        "calcite",
        "dripstone",
    )
    if any(term in block_name for term in stone_terms):
        return "stone"

    if any(term in block_name for term in ("leaves", "vine", "moss", "fern", "bush")):
        return "foliage"

    if any(term in block_name for term in ("spruce", "dark_oak")):
        return "wood_dark"
    if any(term in block_name for term in ("birch", "bamboo")):
        return "wood_light"
    if any(term in block_name for term in ("mangrove", "cherry", "crimson")):
        return "wood_red"
    if any(term in block_name for term in ("oak", "jungle", "acacia", "warped")):
        return "wood_medium"

    if any(term in block_name for term in ("glass", "pane")):
        return "glass"
    if any(term in block_name for term in ("wool", "carpet")):
        return "cloth"
    if any(term in block_name for term in ("terracotta", "brick", "mud")):
        return "masonry"
    if any(term in block_name for term in ("dirt", "grass", "podzol", "path")):
        return "ground"
    return block_name


def geometry_kind(block_name: str) -> str:
    suffixes = (
        "stairs",
        "slab",
        "wall",
        "fence_gate",
        "fence",
        "door",
        "trapdoor",
        "log",
        "wood",
        "planks",
        "leaves",
        "pane",
        "glass",
        "bricks",
        "tiles",
        "lantern",
        "torch",
        "pot",
        "button",
        "pressure_plate",
        "sign",
        "bed",
        "chest",
        "barrel",
        "full",
    )
    for suffix in suffixes:
        if block_name.endswith(suffix):
            return suffix
    return "full"


def kept_properties(kind: str, props: dict[str, str]) -> dict[str, str]:
    keep_by_kind = {
        "stairs": {"facing", "half", "shape"},
        "slab": {"type"},
        "wall": {"north", "south", "east", "west", "up"},
        "fence": {"north", "south", "east", "west"},
        "fence_gate": {"facing", "in_wall"},
        "door": {"facing", "half", "hinge"},
        "trapdoor": {"facing", "half", "open"},
        "log": {"axis"},
        "wood": {"axis"},
        "pane": {"north", "south", "east", "west"},
    }
    allowed = keep_by_kind.get(kind, set())
    return {key: props[key] for key in sorted(allowed) if key in props}


def is_ornamental(block_name: str, family: str, kind: str) -> bool:
    if family == AIR:
        return False
    ornamental_terms = (
        "amethyst",
        "anvil",
        "bee",
        "bell",
        "bookshelf",
        "brewing_stand",
        "cake",
        "cauldron",
        "composter",
        "flower",
        "potted",
        "sapling",
        "lantern",
        "torch",
        "candle",
        "carpet",
        "banner",
        "sign",
        "item_frame",
        "painting",
        "button",
        "pressure_plate",
        "bed",
        "chest",
        "barrel",
        "crafting_table",
        "furnace",
        "campfire",
        "chain",
        "cartography_table",
        "table",
    )
    return family == "foliage" or kind in {"lantern", "torch", "pot", "button", "pressure_plate", "sign", "bed", "chest", "barrel"} or any(
        term in block_name for term in ornamental_terms
    ) or any(
        term in block_name for term in DECORATIVE_NAME_TERMS
    )


def reduce_state(state: str) -> tuple[str, bool]:
    base, props = parse_block_state(state)
    block_name = base.removeprefix("minecraft:")
    family = material_family(block_name)
    if family == AIR:
        return AIR, False

    kind = geometry_kind(block_name)
    structural = not is_ornamental(block_name, family, kind)
    if not structural:
        return ORNAMENT, False

    kept = kept_properties(kind, props)
    return f"{family}:{kind}{format_props(kept)}", True


def swap_stair_handedness(shape: str) -> str:
    swaps = {
        "inner_left": "inner_right",
        "inner_right": "inner_left",
        "outer_left": "outer_right",
        "outer_right": "outer_left",
    }
    return swaps.get(shape, shape)


def transform_category(category: str, transform: str) -> str:
    if category in {AIR, ORNAMENT}:
        return category
    if transform == "original":
        return category

    direction_map = TRANSFORM_DIRECTION_MAPS[transform]
    base, props = parse_category(category)
    transformed: dict[str, str] = {}

    for key, value in props.items():
        new_key = direction_map[key] if key in DIRECTIONS else key
        new_value = direction_map[value] if value in DIRECTIONS else value
        transformed[new_key] = new_value

    if transform in {"mirror_x", "mirror_z"}:
        if "shape" in transformed:
            transformed["shape"] = swap_stair_handedness(transformed["shape"])
        if "hinge" in transformed:
            transformed["hinge"] = "right" if transformed["hinge"] == "left" else "left"

    if transform == "rot90_y" and transformed.get("axis") in {"x", "z"}:
        transformed["axis"] = "z" if transformed["axis"] == "x" else "x"

    return f"{base}{format_props(transformed)}"


def transform_volume(volume: np.ndarray, transform: str) -> np.ndarray:
    if transform == "original":
        return volume
    if transform == "mirror_x":
        return volume[:, :, ::-1]
    if transform == "mirror_z":
        return volume[:, ::-1, :]
    if transform == "rot90_y":
        return np.rot90(volume, k=1, axes=(1, 2))
    raise ValueError(transform)


def transform_category_array(
    volume: np.ndarray,
    transform: str,
    palette: list[str],
    category_to_id: dict[str, int],
) -> np.ndarray:
    spatial = transform_volume(volume, transform).copy()
    if transform == "original":
        return spatial

    remap = np.arange(len(palette), dtype=spatial.dtype)
    for old_id, category in enumerate(palette):
        transformed = transform_category(category, transform)
        remap[old_id] = category_to_id.get(transformed, old_id)
    return remap[spatial]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/schem"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/phase1"))
    parser.add_argument("--docs-path", type=Path, default=Path("docs/phase1_summary.md"))
    args = parser.parse_args()

    schem_paths = sorted(args.input_dir.glob("*.schem"))
    if not schem_paths:
        raise SystemExit(f"No .schem files found in {args.input_dir}")

    loaded = [load_schematic(path) for path in schem_paths]
    category_by_state: dict[str, str] = {}
    structural_by_state: dict[str, bool] = {}
    for schematic in loaded:
        for state in schematic.palette_by_id.values():
            if state not in category_by_state:
                category, structural = reduce_state(state)
                category_by_state[state] = category
                structural_by_state[state] = structural

    structural_categories = set(category_by_state.values()) - {ORNAMENT}
    for category in list(structural_categories):
        for transform in ("mirror_x", "mirror_z", "rot90_y"):
            transformed = transform_category(category, transform)
            if transformed != ORNAMENT:
                structural_categories.add(transformed)

    categories = sorted(structural_categories | {AIR})
    if AIR in categories:
        categories.remove(AIR)
    ordered_categories = [AIR] + categories
    category_to_id = {category: idx for idx, category in enumerate(ordered_categories)}

    arrays_dir = args.output_dir / "arrays"
    masks_dir = args.output_dir / "structural_masks"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    transforms = ("original", "mirror_x", "mirror_z", "rot90_y")
    metadata: dict[str, Any] = {
        "source_count": len(loaded),
        "augmentation_count": len(transforms),
        "array_count": len(loaded) * len(transforms),
        "array_shape_order": "y,z,x",
        "palette": ordered_categories,
        "air_id": category_to_id[AIR],
        "state_reductions": category_by_state,
        "category_transform_remapping": {
            "enabled": True,
            "transforms": list(transforms),
            "direction_maps": TRANSFORM_DIRECTION_MAPS,
            "notes": "Mirrors and rotations remap facing values, directional side keys, stair handedness, door hinges, and x/z axes.",
        },
        "outputs": [],
    }

    source_stats = []
    category_counts = Counter()
    raw_state_counts = Counter()
    for schematic in loaded:
        id_to_category = {
            raw_id: category_by_state[state]
            for raw_id, state in schematic.palette_by_id.items()
        }
        id_to_structural = {
            raw_id: structural_by_state[state]
            for raw_id, state in schematic.palette_by_id.items()
        }

        reduced = np.zeros(schematic.shape_yzx, dtype=np.uint16)
        structural_mask = np.zeros(schematic.shape_yzx, dtype=bool)
        for raw_id, category in id_to_category.items():
            raw_mask = schematic.block_ids == raw_id
            reduced[raw_mask] = category_to_id.get(category, category_to_id[AIR])
            structural_mask[raw_mask] = id_to_structural[raw_id]
            count = int(raw_mask.sum())
            category_counts[category] += count
            raw_state_counts[schematic.palette_by_id[raw_id]] += count

        structural = np.where(structural_mask, reduced, category_to_id[AIR]).astype(np.uint16)
        total_blocks = int(np.prod(schematic.shape_yzx))
        structural_blocks = int(structural_mask.sum())
        source_stats.append(
            {
                "name": schematic.name,
                "shape_yzx": list(schematic.shape_yzx),
                "total_voxels": total_blocks,
                "structural_voxels": structural_blocks,
                "structural_fraction": structural_blocks / total_blocks,
                "raw_palette_size": len(schematic.palette_by_id),
            }
        )

        for transform in transforms:
            transformed = transform_category_array(structural, transform, ordered_categories, category_to_id)
            transformed_mask = transform_volume(structural_mask, transform).copy()
            stem = f"{schematic.name}__{transform}"
            array_path = arrays_dir / f"{stem}.npy"
            mask_path = masks_dir / f"{stem}.npy"
            np.save(array_path, transformed)
            np.save(mask_path, transformed_mask)
            metadata["outputs"].append(
                {
                    "source": schematic.name,
                    "transform": transform,
                    "array": str(array_path),
                    "structural_mask": str(mask_path),
                    "shape_yzx": list(transformed.shape),
                    "structural_voxels": int(transformed_mask.sum()),
                }
            )

    stats = {
        "sources": source_stats,
        "category_counts": dict(category_counts.most_common()),
        "raw_state_counts": dict(raw_state_counts.most_common()),
    }
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "stats.json", stats)

    grouped_states: dict[str, list[str]] = defaultdict(list)
    for state, category in category_by_state.items():
        grouped_states[category].append(state)

    lines = [
        "# Phase I Observations",
        "",
        f"Loaded {len(loaded)} source schematics and wrote {len(metadata['outputs'])} augmented structural arrays.",
        "",
        "## Source Volumes",
        "",
        "| Source | Shape (Y,Z,X) | Structural Voxels | Structural Fraction | Raw Palette |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in source_stats:
        lines.append(
            f"| {item['name']} | {tuple(item['shape_yzx'])} | {item['structural_voxels']} | "
            f"{item['structural_fraction']:.3f} | {item['raw_palette_size']} |"
        )
    lines.extend(
        [
            "",
            "## Palette Reduction",
            "",
            f"Reduced {len(category_by_state)} raw block states to {len(ordered_categories)} output structural categories including air.",
            "An internal `ornament` marker is used during masking, but ornamental voxels are written as air in the cleaned arrays and are not part of the output palette.",
            "Material variants are collapsed before learning: dark stones such as deepslate, blackstone, basalt, and tuff map to `dark_stone`, while wood species map into broad tonal families and keep geometry/orientation where it affects structure.",
            "Augmentations remap directional categories after spatial transforms, including `facing` values, directional side keys, stair handedness, door hinges, and x/z axes.",
            "",
            "Most common raw reductions before masking:",
            "",
        ]
    )
    for category, count in category_counts.most_common(20):
        lines.append(f"- `{category}`: {count}")
    lines.extend(
        [
            "",
            "Noise note: the source data intentionally swaps deepslate bricks with nearby dark masonry variants for visual interest. This is treated as palette noise for learning and is collapsed into the same dark-stone family while preserving stairs/slabs/walls when present.",
            "",
            "Ornaments such as flowers, torches, lanterns, signs, buttons, carpets, containers, and foliage are masked to air in the structural arrays. This keeps the Phase II clustering focused on house massing and load-bearing shapes.",
        ]
    )
    args.docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote Phase I arrays to {arrays_dir}")
    print(f"Wrote metadata to {args.output_dir / 'metadata.json'}")
    print(f"Wrote observations to {args.docs_path}")


if __name__ == "__main__":
    main()
