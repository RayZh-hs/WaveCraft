#!/usr/bin/env python3
"""Export learned prototype tiles as an in-game Sponge schematic gallery."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import nbtlib
import numpy as np
from nbtlib import ByteArray, Compound, Int, List, Short


AIR_STATE = "minecraft:air"
VOID_AIR_STATE = "minecraft:void_air"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_category(category: str) -> tuple[str, str, dict[str, str]]:
    if category == "air":
        return "air", "air", {}
    base, raw_props = (category.split("[", 1) + [""])[:2] if "[" in category else (category, "")
    family, kind = base.split(":", 1) if ":" in base else (base, "full")
    props = {}
    for item in raw_props.rstrip("]").split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            props[key] = value
    return family, kind, props


def state_props(props: dict[str, str]) -> str:
    if not props:
        return ""
    return "[" + ",".join(f"{key}={props[key]}" for key in sorted(props)) + "]"


def normalize_wall_value(value: str | None) -> str:
    if value in {"low", "tall", "none"}:
        return value
    if value == "true":
        return "low"
    return "none"


def normalize_bool(value: str | None) -> str:
    return "true" if value == "true" else "false"


def material_blocks(family: str) -> dict[str, str]:
    families = {
        "wood_dark": {
            "full": "minecraft:spruce_planks",
            "planks": "minecraft:spruce_planks",
            "log": "minecraft:spruce_log",
            "wood": "minecraft:spruce_wood",
            "stairs": "minecraft:spruce_stairs",
            "slab": "minecraft:spruce_slab",
            "wall": "minecraft:cobblestone_wall",
            "fence": "minecraft:spruce_fence",
            "fence_gate": "minecraft:spruce_fence_gate",
            "door": "minecraft:spruce_door",
            "trapdoor": "minecraft:spruce_trapdoor",
            "pane": "minecraft:brown_stained_glass_pane",
        },
        "wood_medium": {
            "full": "minecraft:oak_planks",
            "planks": "minecraft:oak_planks",
            "log": "minecraft:oak_log",
            "wood": "minecraft:oak_wood",
            "stairs": "minecraft:oak_stairs",
            "slab": "minecraft:oak_slab",
            "wall": "minecraft:cobblestone_wall",
            "fence": "minecraft:oak_fence",
            "fence_gate": "minecraft:oak_fence_gate",
            "door": "minecraft:oak_door",
            "trapdoor": "minecraft:oak_trapdoor",
            "pane": "minecraft:glass_pane",
        },
        "wood_light": {
            "full": "minecraft:birch_planks",
            "planks": "minecraft:birch_planks",
            "log": "minecraft:birch_log",
            "wood": "minecraft:birch_wood",
            "stairs": "minecraft:birch_stairs",
            "slab": "minecraft:birch_slab",
            "wall": "minecraft:cobblestone_wall",
            "fence": "minecraft:birch_fence",
            "fence_gate": "minecraft:birch_fence_gate",
            "door": "minecraft:birch_door",
            "trapdoor": "minecraft:birch_trapdoor",
            "pane": "minecraft:glass_pane",
        },
        "wood_red": {
            "full": "minecraft:mangrove_planks",
            "planks": "minecraft:mangrove_planks",
            "log": "minecraft:mangrove_log",
            "wood": "minecraft:mangrove_wood",
            "stairs": "minecraft:mangrove_stairs",
            "slab": "minecraft:mangrove_slab",
            "wall": "minecraft:cobblestone_wall",
            "fence": "minecraft:mangrove_fence",
            "fence_gate": "minecraft:mangrove_fence_gate",
            "door": "minecraft:mangrove_door",
            "trapdoor": "minecraft:mangrove_trapdoor",
            "pane": "minecraft:red_stained_glass_pane",
        },
        "dark_stone": {
            "full": "minecraft:deepslate_bricks",
            "bricks": "minecraft:deepslate_bricks",
            "tiles": "minecraft:deepslate_tiles",
            "stairs": "minecraft:deepslate_brick_stairs",
            "slab": "minecraft:deepslate_brick_slab",
            "wall": "minecraft:deepslate_brick_wall",
        },
        "stone": {
            "full": "minecraft:stone_bricks",
            "bricks": "minecraft:stone_bricks",
            "stairs": "minecraft:stone_brick_stairs",
            "slab": "minecraft:stone_brick_slab",
            "wall": "minecraft:stone_brick_wall",
        },
        "masonry": {
            "full": "minecraft:bricks",
            "bricks": "minecraft:bricks",
            "stairs": "minecraft:brick_stairs",
            "slab": "minecraft:brick_slab",
            "wall": "minecraft:brick_wall",
        },
        "glass": {
            "full": "minecraft:glass",
            "glass": "minecraft:glass",
            "pane": "minecraft:glass_pane",
        },
        "cloth": {"full": "minecraft:white_wool"},
        "ground": {"full": "minecraft:grass_block"},
    }
    return families.get(family, {"full": f"minecraft:{family}"})


def wood_species_for_family(family: str) -> str:
    return {
        "wood_dark": "spruce",
        "wood_medium": "oak",
        "wood_light": "birch",
        "wood_red": "mangrove",
    }.get(family, "oak")


def category_to_block_state(category: str) -> str:
    family, kind, props = parse_category(category)
    if family == "air":
        return AIR_STATE

    if family == "hay_block":
        return "minecraft:hay_block"
    if family == "soul_sand":
        return "minecraft:soul_sand"
    if family == "nether_wart":
        return "minecraft:red_wool"

    blocks = material_blocks(family)
    block = blocks.get(kind, blocks.get("full", "minecraft:stone"))
    if props.get("stripped") == "true" and kind == "full" and family.startswith("wood_"):
        kind = "wood"
    if kind in {"log", "wood"} and props.get("stripped") == "true":
        species = wood_species_for_family(family)
        block = f"minecraft:stripped_{species}_{kind}"

    if kind == "stairs":
        state = {
            "facing": props.get("facing", "north"),
            "half": props.get("half", "bottom"),
            "shape": props.get("shape", "straight"),
            "waterlogged": "false",
        }
        return f"{block}{state_props(state)}"
    if kind == "slab":
        state = {"type": props.get("type", "bottom"), "waterlogged": "false"}
        return f"{block}{state_props(state)}"
    if kind == "wall":
        state = {
            "east": normalize_wall_value(props.get("east")),
            "north": normalize_wall_value(props.get("north")),
            "south": normalize_wall_value(props.get("south")),
            "up": normalize_bool(props.get("up", "true")),
            "waterlogged": "false",
            "west": normalize_wall_value(props.get("west")),
        }
        return f"{block}{state_props(state)}"
    if kind == "fence":
        state = {
            "east": normalize_bool(props.get("east")),
            "north": normalize_bool(props.get("north")),
            "south": normalize_bool(props.get("south")),
            "waterlogged": "false",
            "west": normalize_bool(props.get("west")),
        }
        return f"{block}{state_props(state)}"
    if kind == "fence_gate":
        state = {
            "facing": props.get("facing", "north"),
            "in_wall": normalize_bool(props.get("in_wall")),
            "open": "false",
            "powered": "false",
        }
        return f"{block}{state_props(state)}"
    if kind == "pane":
        state = {
            "east": normalize_bool(props.get("east")),
            "north": normalize_bool(props.get("north")),
            "south": normalize_bool(props.get("south")),
            "waterlogged": "false",
            "west": normalize_bool(props.get("west")),
        }
        return f"{block}{state_props(state)}"
    if kind == "door":
        half = props.get("half", "lower")
        if half == "bottom":
            half = "lower"
        if half == "top":
            half = "upper"
        state = {
            "facing": props.get("facing", "north"),
            "half": half,
            "hinge": props.get("hinge", "left"),
            "open": "false",
            "powered": "false",
        }
        return f"{block}{state_props(state)}"
    if kind == "trapdoor":
        state = {
            "facing": props.get("facing", "north"),
            "half": props.get("half", "bottom"),
            "open": props.get("open", "false"),
            "powered": "false",
            "waterlogged": "false",
        }
        return f"{block}{state_props(state)}"
    if kind in {"log", "wood"}:
        state = {"axis": props.get("axis", "y")}
        return f"{block}{state_props(state)}"
    return block


def add_state(palette: dict[str, int], state: str) -> int:
    if state not in palette:
        palette[state] = len(palette)
    return palette[state]


def set_block(blocks: np.ndarray, palette: dict[str, int], y: int, z: int, x: int, state: str) -> None:
    blocks[y, z, x] = add_state(palette, state)


def encode_varints(values: np.ndarray) -> list[int]:
    encoded: list[int] = []
    for raw_value in values:
        value = int(raw_value)
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                byte |= 0x80
            encoded.append(byte if byte < 128 else byte - 256)
            if not value:
                break
    return encoded


def write_sponge_schem(path: Path, blocks: np.ndarray, palette: dict[str, int], data_version: int) -> None:
    height, length, width = blocks.shape
    flat = blocks.reshape(-1)
    data = ByteArray(encode_varints(flat))
    root = Compound(
        {
            "Version": Int(3),
            "Width": Short(width),
            "Height": Short(height),
            "Length": Short(length),
            "Blocks": Compound(
                {
                    "Data": data,
                    "Palette": Compound({state: Int(index) for state, index in sorted(palette.items(), key=lambda item: item[1])}),
                    "BlockEntities": List[Compound]([]),
                }
            ),
            "Entities": List[Compound]([]),
            "DataVersion": Int(data_version),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    nbtlib.File({"Schematic": root}).save(path, gzipped=True)


def export_gallery(
    prototypes: np.ndarray,
    palette_categories: list[str],
    output_path: Path,
    columns: int,
    spacing: int,
    base: bool,
    data_version: int,
) -> None:
    tile_count, tile_h, tile_l, tile_w = prototypes.shape
    rows = math.ceil(tile_count / columns)
    cell_w = tile_w + spacing
    cell_l = tile_l + spacing
    base_offset = 1 if base else 0
    width = columns * cell_w - spacing
    length = rows * cell_l - spacing
    height = tile_h + base_offset + 2

    palette: dict[str, int] = {AIR_STATE: 0}
    blocks = np.zeros((height, length, width), dtype=np.int32)

    if base:
        for tile_index in range(tile_count):
            row = tile_index // columns
            col = tile_index % columns
            x0 = col * cell_w
            z0 = row * cell_l
            base_state = "minecraft:smooth_stone"
            stripe_state = "minecraft:yellow_concrete" if tile_index % 2 else "minecraft:light_gray_concrete"
            for z in range(z0, z0 + tile_l):
                for x in range(x0, x0 + tile_w):
                    set_block(blocks, palette, 0, z, x, base_state)
            set_block(blocks, palette, 0, z0, x0, stripe_state)
            set_block(blocks, palette, 0, z0, x0 + tile_w - 1, stripe_state)
            set_block(blocks, palette, 0, z0 + tile_l - 1, x0, stripe_state)
            set_block(blocks, palette, 0, z0 + tile_l - 1, x0 + tile_w - 1, stripe_state)

    for tile_index, patch in enumerate(prototypes):
        row = tile_index // columns
        col = tile_index % columns
        x0 = col * cell_w
        z0 = row * cell_l
        for y in range(tile_h):
            for z in range(tile_l):
                for x in range(tile_w):
                    category = palette_categories[int(patch[y, z, x])]
                    state = category_to_block_state(category)
                    if state == AIR_STATE:
                        continue
                    set_block(blocks, palette, y + base_offset, z0 + z, x0 + x, state)

    write_sponge_schem(output_path, blocks, palette, data_version)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", type=Path, default=Path("datasets/phase1"))
    parser.add_argument("--phase2-dir", type=Path, default=Path("datasets/phase2"))
    parser.add_argument("--output", type=Path, default=Path("datasets/phase2/inspection/tile_gallery.schem"))
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--spacing", type=int, default=3)
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--data-version", type=int, default=4556)
    args = parser.parse_args()

    metadata = load_json(args.phase1_dir / "metadata.json")
    tile_library = np.load(args.phase2_dir / "tile_library.npz")
    export_gallery(
        prototypes=tile_library["prototypes"],
        palette_categories=metadata["palette"],
        output_path=args.output,
        columns=args.columns,
        spacing=args.spacing,
        base=not args.no_base,
        data_version=args.data_version,
    )
    print(f"Wrote tile gallery schematic to {args.output}")


if __name__ == "__main__":
    main()
