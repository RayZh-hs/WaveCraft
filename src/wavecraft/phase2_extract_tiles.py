#!/usr/bin/env python3
"""Phase II: extract structural patches and cluster them into kitbash tiles."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score


AIR = "air"

WINDOW_KINDS = {"pane", "glass"}
OPENING_KINDS = {"door", "trapdoor", "fence_gate"}
ROOF_KINDS = {"stairs", "slab", "tiles"}
FRAME_KINDS = {"log", "wood", "fence"}
WALL_KINDS = {"bricks", "full", "planks", "wall"}
LOW_INFORMATION_KINDS = {"bricks", "full", "log", "planks", "slab", "tiles", "wall", "wood"}
ARCHITECTURAL_WEIGHTS = {
    "pane": 6.0,
    "glass": 5.0,
    "door": 5.0,
    "trapdoor": 4.0,
    "fence_gate": 4.0,
    "stairs": 3.0,
    "wall": 3.0,
    "slab": 2.0,
    "fence": 2.0,
    "tiles": 1.5,
    "log": 1.25,
    "wood": 1.0,
    "bricks": 0.8,
    "planks": 0.6,
    "full": 0.5,
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_phase1(metadata_path: Path) -> tuple[dict[str, Any], list[tuple[dict[str, Any], np.ndarray]]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays = []
    for item in metadata["outputs"]:
        arrays.append((item, np.load(item["array"])))
    return metadata, arrays


def category_kind(category: str) -> str:
    if category == AIR:
        return AIR
    without_props = category.split("[", 1)[0]
    if ":" not in without_props:
        return without_props
    return without_props.split(":", 1)[1]


def patch_kind_counts(patch: np.ndarray, kind_by_id: list[str], air_id: int) -> Counter[str]:
    values, counts = np.unique(patch, return_counts=True)
    result: Counter[str] = Counter()
    for value, count in zip(values, counts, strict=True):
        if int(value) == air_id:
            continue
        result[kind_by_id[int(value)]] += int(count)
    return result


def air_nonair_interface_count(patch: np.ndarray, air_id: int) -> int:
    solid = patch != air_id
    score = 0
    score += int(np.count_nonzero(solid[1:, :, :] != solid[:-1, :, :]))
    score += int(np.count_nonzero(solid[:, 1:, :] != solid[:, :-1, :]))
    score += int(np.count_nonzero(solid[:, :, 1:] != solid[:, :, :-1]))
    return score


def boundary_contact_score(patch: np.ndarray, air_id: int) -> int:
    score = 0
    score += int(np.count_nonzero(patch[0, :, :] != air_id))
    score += int(np.count_nonzero(patch[-1, :, :] != air_id))
    score += int(np.count_nonzero(patch[:, 0, :] != air_id))
    score += int(np.count_nonzero(patch[:, -1, :] != air_id))
    score += int(np.count_nonzero(patch[:, :, 0] != air_id))
    score += int(np.count_nonzero(patch[:, :, -1] != air_id))
    return score


def architectural_score(patch: np.ndarray, kind_by_id: list[str], air_id: int) -> float:
    counts = patch_kind_counts(patch, kind_by_id, air_id)
    weighted_shapes = sum(ARCHITECTURAL_WEIGHTS.get(kind, 0.0) * count for kind, count in counts.items())
    interface = air_nonair_interface_count(patch, air_id)
    total = patch.size
    non_air = int(np.count_nonzero(patch != air_id))
    non_air_fraction = non_air / total

    # Reward recognizable architectural components and exposed surfaces. Penalize
    # fully solid chunks because they cluster as material blobs instead of parts.
    score = weighted_shapes / max(non_air, 1)
    score += 5.0 * interface / max(3 * total, 1)
    if non_air_fraction > 0.9:
        score *= 0.35
    return score


def dominant_kind_share(counts: Counter[str]) -> tuple[str, float]:
    total = sum(counts.values())
    if total == 0:
        return AIR, 0.0
    kind, count = counts.most_common(1)[0]
    return kind, count / total


def is_low_information_patch(
    patch: np.ndarray,
    kind_by_id: list[str],
    air_id: int,
    max_simple_kind_fraction: float,
) -> bool:
    counts = patch_kind_counts(patch, kind_by_id, air_id)
    if not counts:
        return False
    if sum(counts[kind] for kind in WINDOW_KINDS | OPENING_KINDS) > 0:
        return False
    kind, share = dominant_kind_share(counts)
    return kind in LOW_INFORMATION_KINDS and share >= max_simple_kind_fraction


def semantic_bucket(patch: np.ndarray, kind_by_id: list[str], air_id: int) -> str:
    counts = patch_kind_counts(patch, kind_by_id, air_id)
    total = patch.size
    non_air = int(np.count_nonzero(patch != air_id))
    air_fraction = 1.0 - (non_air / total)
    windows = sum(counts[kind] for kind in WINDOW_KINDS)
    openings = sum(counts[kind] for kind in OPENING_KINDS)
    roof = sum(counts[kind] for kind in ROOF_KINDS)
    frame = sum(counts[kind] for kind in FRAME_KINDS)
    wall = sum(counts[kind] for kind in WALL_KINDS)
    contact = boundary_contact_score(patch, air_id)

    if windows >= 2:
        return "window"
    if openings >= 2:
        return "opening"
    if roof >= 8 and air_fraction >= 0.2:
        return "roof"
    if frame >= 8 and air_fraction >= 0.2:
        return "frame"
    if wall >= 12 and air_fraction >= 0.15 and contact >= 20:
        return "wall"
    if non_air / total >= 0.85:
        return "mass"
    return "transition"


def candidate_positions(
    volume: np.ndarray,
    window: int,
    air_id: int,
    max_air_fraction: float,
    max_non_air_fraction: float,
    min_architectural_score: float,
    max_simple_kind_fraction: float,
    kind_by_id: list[str],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if min(volume.shape) < window:
        return (
            np.empty((0, 3), dtype=np.int32),
            [],
            {"density_candidates": 0, "rejected_dense": 0, "rejected_score": 0, "rejected_low_information": 0},
        )
    air = volume == air_id
    windows = np.lib.stride_tricks.sliding_window_view(air, (window, window, window))
    air_counts = windows.sum(axis=(-1, -2, -3))
    threshold = int((window**3) * max_air_fraction)
    density_positions = np.argwhere(air_counts <= threshold).astype(np.int32)

    kept_positions = []
    buckets: list[str] = []
    rejected_dense = 0
    rejected_score = 0
    rejected_low_information = 0
    max_non_air = int(round((window**3) * max_non_air_fraction))
    for y, z, x in density_positions:
        patch = volume[y : y + window, z : z + window, x : x + window]
        non_air = int(np.count_nonzero(patch != air_id))
        if non_air > max_non_air:
            rejected_dense += 1
            continue
        score = architectural_score(patch, kind_by_id, air_id)
        if score < min_architectural_score:
            rejected_score += 1
            continue
        if is_low_information_patch(patch, kind_by_id, air_id, max_simple_kind_fraction):
            rejected_low_information += 1
            continue
        kept_positions.append((int(y), int(z), int(x)))
        buckets.append(semantic_bucket(patch, kind_by_id, air_id))

    stats = {
        "density_candidates": int(len(density_positions)),
        "rejected_dense": int(rejected_dense),
        "rejected_score": int(rejected_score),
        "rejected_low_information": int(rejected_low_information),
    }
    return np.asarray(kept_positions, dtype=np.int32).reshape((-1, 3)), buckets, stats


def allocate_balanced_counts(available_by_key: dict[Hashable, int], max_patches: int) -> dict[Hashable, int]:
    keys = sorted((key for key, count in available_by_key.items() if count > 0), key=str)
    if not keys:
        return {}

    target = min(max_patches, sum(available_by_key.values()))
    counts = {key: min(available_by_key[key], target // len(keys)) for key in keys}
    remaining = target - sum(counts.values())

    while remaining > 0:
        progressed = False
        for key in keys:
            if counts[key] < available_by_key[key]:
                counts[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return counts


def extract_patch_sample(
    arrays: list[tuple[dict[str, Any], np.ndarray]],
    palette: list[str],
    window: int,
    air_id: int,
    max_air_fraction: float,
    max_non_air_fraction: float,
    min_architectural_score: float,
    max_simple_kind_fraction: float,
    max_patches: int,
    rng: np.random.Generator,
    balanced_by_source: bool,
    balanced_by_bucket: bool,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    kind_by_id = [category_kind(category) for category in palette]
    candidates = []
    total_candidates = 0
    total_density_candidates = 0
    total_rejected_dense = 0
    total_rejected_score = 0
    total_rejected_low_information = 0
    per_array_counts = []
    quota_candidate_counts: Counter[Hashable] = Counter()
    source_candidate_counts: Counter[str] = Counter()
    bucket_candidate_counts: Counter[str] = Counter()
    source_bucket_candidate_counts: Counter[tuple[str, str]] = Counter()

    for index, (item, volume) in enumerate(arrays):
        positions, buckets, filter_stats = candidate_positions(
            volume,
            window,
            air_id,
            max_air_fraction,
            max_non_air_fraction,
            min_architectural_score,
            max_simple_kind_fraction,
            kind_by_id,
        )
        total_candidates += len(positions)
        total_density_candidates += filter_stats["density_candidates"]
        total_rejected_dense += filter_stats["rejected_dense"]
        total_rejected_score += filter_stats["rejected_score"]
        total_rejected_low_information += filter_stats["rejected_low_information"]
        source_candidate_counts[item["source"]] += len(positions)
        bucket_counts = Counter(buckets)
        bucket_candidate_counts.update(bucket_counts)
        for bucket, count in bucket_counts.items():
            source_bucket_candidate_counts[(item["source"], bucket)] += count
            quota_key: Hashable
            if balanced_by_source and balanced_by_bucket:
                quota_key = (item["source"], bucket)
            elif balanced_by_source:
                quota_key = item["source"]
            elif balanced_by_bucket:
                quota_key = bucket
            else:
                quota_key = "all"
            quota_candidate_counts[quota_key] += count
        per_array_counts.append(
            {
                "array": item["array"],
                "source": item["source"],
                "transform": item["transform"],
                "eligible_patches": int(len(positions)),
                "density_candidates": int(filter_stats["density_candidates"]),
                "rejected_dense": int(filter_stats["rejected_dense"]),
                "rejected_score": int(filter_stats["rejected_score"]),
                "rejected_low_information": int(filter_stats["rejected_low_information"]),
                "bucket_counts": {bucket: int(count) for bucket, count in sorted(bucket_counts.items())},
            }
        )
        if len(positions):
            candidates.append((index, positions, np.asarray(buckets, dtype=object)))

    if total_candidates == 0:
        raise SystemExit("No eligible patches found. Relax air/density/architectural filters or use smaller --window.")

    if balanced_by_source and balanced_by_bucket:
        source_quotas = allocate_balanced_counts(dict(source_candidate_counts), max_patches)
        quotas: dict[Hashable, int] = {}
        for source, source_quota in source_quotas.items():
            bucket_counts_for_source = {
                (bucket_source, bucket): count
                for (bucket_source, bucket), count in source_bucket_candidate_counts.items()
                if bucket_source == source
            }
            quotas.update(allocate_balanced_counts(bucket_counts_for_source, int(source_quota)))
    elif balanced_by_source or balanced_by_bucket:
        quotas = allocate_balanced_counts(dict(quota_candidate_counts), max_patches)
    else:
        quotas = dict(quota_candidate_counts)
        if total_candidates > max_patches:
            scale = max_patches / total_candidates
            quotas = {key: int(count * scale) for key, count in quotas.items()}

    quota_cursor: Counter[Hashable] = Counter()
    selected_source_counts: Counter[str] = Counter()
    selected_bucket_counts: Counter[str] = Counter()
    patches: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []

    for array_index, positions, buckets in candidates:
        item, volume = arrays[array_index]
        source = item["source"]
        for bucket in sorted(set(str(value) for value in buckets)):
            bucket_indices = np.flatnonzero(buckets == bucket)
            if not len(bucket_indices):
                continue
            if balanced_by_source and balanced_by_bucket:
                quota_key = (source, bucket)
            elif balanced_by_source:
                quota_key = source
            elif balanced_by_bucket:
                quota_key = bucket
            else:
                quota_key = "all"
            quota = quotas.get(quota_key, 0)
            remaining_for_key = quota - quota_cursor[quota_key]
            if remaining_for_key <= 0:
                continue

            remaining_candidates = quota_candidate_counts[quota_key]
            take = min(
                len(bucket_indices),
                remaining_for_key,
                int(round(remaining_for_key * (len(bucket_indices) / max(remaining_candidates, 1)))),
            )
            if take == 0 and remaining_for_key > 0 and len(bucket_indices) > 0:
                take = 1
            take = min(take, len(bucket_indices), remaining_for_key)
            quota_candidate_counts[quota_key] -= len(bucket_indices)

            if not take:
                continue

            local_indices = np.sort(rng.choice(bucket_indices, size=take, replace=False))
            for local_index in local_indices:
                y, z, x = positions[int(local_index)]
                patch = volume[y : y + window, z : z + window, x : x + window]
                patches.append(patch.astype(np.uint16, copy=True))
                sources.append(
                    {
                        "array": item["array"],
                        "source": item["source"],
                        "transform": item["transform"],
                        "y": int(y),
                        "z": int(z),
                        "x": int(x),
                        "center_y_fraction": float((y + window / 2) / max(volume.shape[0], 1)),
                        "semantic_bucket": bucket,
                    }
                )
            quota_cursor[quota_key] += take
            selected_source_counts[source] += take
            selected_bucket_counts[bucket] += take

    stats = {
        "density_candidates": int(total_density_candidates),
        "total_eligible_patches": int(total_candidates),
        "rejected_dense": int(total_rejected_dense),
        "rejected_score": int(total_rejected_score),
        "rejected_low_information": int(total_rejected_low_information),
        "sampled_patches": int(len(patches)),
        "balanced_by_source": balanced_by_source,
        "balanced_by_bucket": balanced_by_bucket,
        "max_non_air_fraction": max_non_air_fraction,
        "min_architectural_score": min_architectural_score,
        "max_simple_kind_fraction": max_simple_kind_fraction,
        "candidate_source_counts": {source: int(count) for source, count in sorted(source_candidate_counts.items())},
        "candidate_bucket_counts": {bucket: int(count) for bucket, count in sorted(bucket_candidate_counts.items())},
        "requested_quotas": {str(key): int(count) for key, count in sorted(quotas.items(), key=lambda item: str(item[0]))},
        "actual_source_counts": {source: int(count) for source, count in sorted(selected_source_counts.items())},
        "actual_bucket_counts": {bucket: int(count) for bucket, count in sorted(selected_bucket_counts.items())},
        "per_array_counts": per_array_counts,
    }
    return np.stack(patches), sources, stats


def sparse_onehot_patches(patches: np.ndarray, palette_size: int) -> sparse.csr_matrix:
    flat = patches.reshape((patches.shape[0], -1)).astype(np.int64)
    rows = np.repeat(np.arange(flat.shape[0], dtype=np.int64), flat.shape[1])
    voxel_offsets = np.tile(np.arange(flat.shape[1], dtype=np.int64) * palette_size, flat.shape[0])
    cols = voxel_offsets + flat.ravel()
    data = np.ones(cols.shape[0], dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(flat.shape[0], flat.shape[1] * palette_size))


def features_for_patches(
    patches: np.ndarray,
    sources: list[dict[str, Any]],
    palette_size: int,
    palette: list[str],
    air_id: int,
    svd_components: int,
    semantic_feature_weight: float,
    rng_seed: int,
) -> tuple[np.ndarray, dict[str, Any], sparse.csr_matrix]:
    onehot = sparse_onehot_patches(patches, palette_size)
    usable_components = min(svd_components, onehot.shape[0] - 1, onehot.shape[1] - 1)
    if usable_components < 2:
        raise SystemExit("Need at least two SVD components for feature extraction")

    reducer = TruncatedSVD(n_components=usable_components, random_state=rng_seed)
    reduced = reducer.fit_transform(onehot).astype(np.float32)
    semantic = semantic_descriptors(patches, palette, air_id).astype(np.float32)
    semantic *= np.float32(semantic_feature_weight)
    heights = np.asarray([source["center_y_fraction"] for source in sources], dtype=np.float32)
    height_feature = heights.reshape((-1, 1))
    features = np.concatenate([reduced, semantic, height_feature], axis=1)
    feature_info = {
        "encoding": "onehot_per_voxel_truncated_svd_plus_architectural_descriptors_plus_height",
        "onehot_shape": list(onehot.shape),
        "svd_components": int(usable_components),
        "svd_explained_variance_ratio_sum": float(reducer.explained_variance_ratio_.sum()),
        "semantic_feature_weight": float(semantic_feature_weight),
        "semantic_descriptor_columns": [
            "air_fraction",
            "boundary_contact_fraction",
            "air_nonair_interface_fraction",
            "window_fraction",
            "opening_fraction",
            "roof_fraction",
            "frame_fraction",
            "wall_fraction",
            "architectural_score",
        ],
    }
    return features, feature_info, onehot


def semantic_descriptors(patches: np.ndarray, palette: list[str], air_id: int) -> np.ndarray:
    kind_by_id = [category_kind(category) for category in palette]
    descriptors = np.zeros((patches.shape[0], 9), dtype=np.float32)
    total = float(patches.shape[1] * patches.shape[2] * patches.shape[3])
    max_boundary = float(6 * patches.shape[1] * patches.shape[2])
    max_interface = float(
        (patches.shape[1] - 1) * patches.shape[2] * patches.shape[3]
        + patches.shape[1] * (patches.shape[2] - 1) * patches.shape[3]
        + patches.shape[1] * patches.shape[2] * (patches.shape[3] - 1)
    )
    for index, patch in enumerate(patches):
        counts = patch_kind_counts(patch, kind_by_id, air_id)
        descriptors[index, 0] = float(np.count_nonzero(patch == air_id) / total)
        descriptors[index, 1] = boundary_contact_score(patch, air_id) / max_boundary
        descriptors[index, 2] = air_nonair_interface_count(patch, air_id) / max(max_interface, 1.0)
        descriptors[index, 3] = sum(counts[kind] for kind in WINDOW_KINDS) / total
        descriptors[index, 4] = sum(counts[kind] for kind in OPENING_KINDS) / total
        descriptors[index, 5] = sum(counts[kind] for kind in ROOF_KINDS) / total
        descriptors[index, 6] = sum(counts[kind] for kind in FRAME_KINDS) / total
        descriptors[index, 7] = sum(counts[kind] for kind in WALL_KINDS) / total
        descriptors[index, 8] = architectural_score(patch, kind_by_id, air_id) / 8.0
    return descriptors


def evaluate_k(features: np.ndarray, k_values: list[int], rng_seed: int, silhouette_samples: int) -> list[dict[str, Any]]:
    results = []
    n_samples = features.shape[0]
    if n_samples < 3:
        raise SystemExit("Need at least three patches for clustering")

    for k in k_values:
        if k < 2 or k >= n_samples:
            continue
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=rng_seed,
            batch_size=min(4096, max(256, n_samples)),
            n_init=8,
        )
        labels = model.fit_predict(features)
        score = None
        if len(set(labels)) > 1:
            sample_size = min(silhouette_samples, n_samples)
            score = float(
                silhouette_score(
                    features,
                    labels,
                    sample_size=sample_size,
                    random_state=rng_seed,
                )
            )
        results.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": score,
                "cluster_sizes": dict(Counter(int(label) for label in labels).most_common()),
            }
        )
    if not results:
        raise SystemExit("No usable k values for the available patch count")
    return results


def choose_k(results: list[dict[str, Any]]) -> int:
    with_silhouette = [item for item in results if item["silhouette"] is not None]
    if with_silhouette:
        return max(with_silhouette, key=lambda item: item["silhouette"])["k"]
    return min(results, key=lambda item: item["inertia"])["k"]


def select_medoids(
    patches: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    min_cluster_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    medoid_patches = []
    summaries = []
    for cluster_id in sorted(set(int(label) for label in labels)):
        indices = np.flatnonzero(labels == cluster_id)
        if len(indices) < min_cluster_size:
            continue
        cluster_features = features[indices]
        distances = np.sum((cluster_features - centers[cluster_id]) ** 2, axis=1)
        best_local = int(np.argmin(distances))
        best_index = int(indices[best_local])
        medoid_patches.append(patches[best_index])
        summaries.append(
            {
                "cluster_id": cluster_id,
                "size": int(len(indices)),
                "medoid_patch_index": best_index,
                "medoid_distance": float(distances[best_local]),
            }
        )
    return np.stack(medoid_patches), summaries


def embedded_holdout_evaluation(
    features: np.ndarray,
    sources: list[dict[str, Any]],
    chosen_k: int,
    rng_seed: int,
) -> list[dict[str, Any]]:
    source_names = sorted({source["source"] for source in sources})
    source_array = np.asarray([source["source"] for source in sources])
    results = []

    for heldout in source_names:
        train_mask = source_array != heldout
        test_mask = ~train_mask
        if train_mask.sum() < chosen_k or test_mask.sum() == 0:
            continue
        model = MiniBatchKMeans(
            n_clusters=chosen_k,
            random_state=rng_seed,
            batch_size=min(4096, max(256, int(train_mask.sum()))),
            n_init=8,
        )
        model.fit(features[train_mask])
        train_distances = model.transform(features[train_mask]).min(axis=1)
        test_distances = model.transform(features[test_mask]).min(axis=1)
        results.append(
            {
                "heldout_source": heldout,
                "train_patches": int(train_mask.sum()),
                "heldout_patches": int(test_mask.sum()),
                "train_mean_distance": float(train_distances.mean()),
                "heldout_mean_distance": float(test_distances.mean()),
                "heldout_to_train_distance_ratio": float(test_distances.mean() / max(train_distances.mean(), 1e-12)),
            }
        )
    return results


def short_label(category: str) -> str:
    if category == "air":
        return " . "
    props = {}
    if "[" in category:
        _, raw_props = category.split("[", 1)
        for item in raw_props.rstrip("]").split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                props[key] = value
    kind = category.split("[", 1)[0].split(":", 1)[1] if ":" in category.split("[", 1)[0] else ""
    if props.get("stripped") == "true" and kind in {"log", "wood"}:
        return f"S{kind[0].upper()}{props.get('axis', '')[:1]}".ljust(3)
    if props.get("stripped") == "true":
        return "SFL"
    base = category.split("[", 1)[0]
    pieces = base.replace(":", "_").split("_")
    letters = "".join(piece[0] for piece in pieces if piece)
    return (letters[:3] or "???").ljust(3)


def write_previews(output_dir: Path, prototypes: np.ndarray, palette: list[str]) -> None:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    labels = [short_label(category) for category in palette]
    legend_lines = [f"{label} {category}" for label, category in zip(labels, palette, strict=True)]
    (preview_dir / "legend.txt").write_text("\n".join(legend_lines) + "\n", encoding="utf-8")

    for tile_index, patch in enumerate(prototypes):
        lines = [f"tile {tile_index:03d}", ""]
        for y in range(patch.shape[0]):
            lines.append(f"y={y}")
            for z in range(patch.shape[1]):
                lines.append(" ".join(labels[int(value)] for value in patch[y, z]))
            lines.append("")
        (preview_dir / f"tile_{tile_index:03d}.txt").write_text("\n".join(lines), encoding="utf-8")


def color_for_category(category: str) -> str:
    if category == "air":
        return "#f8fafc"
    if category.startswith("wood_dark"):
        return "#7a543e" if "stripped=true" in category else "#5b3a29"
    if category.startswith("wood_medium"):
        return "#c08b51" if "stripped=true" in category else "#9a6a3a"
    if category.startswith("wood_light"):
        return "#e3c77f" if "stripped=true" in category else "#d3b06b"
    if category.startswith("wood_red"):
        return "#c45b4e" if "stripped=true" in category else "#9e3f36"
    if category.startswith("dark_stone"):
        return "#334155"
    if category.startswith("stone"):
        return "#8a8f93"
    if category.startswith("masonry"):
        return "#a45f48"
    if category.startswith("glass"):
        return "#8ecae6"
    if category.startswith("cloth"):
        return "#e8e2d0"
    if category.startswith("ground"):
        return "#6f7f3f"
    return "#cbd5e1"


def write_svg_previews(output_dir: Path, prototypes: np.ndarray, palette: list[str]) -> None:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    cell = 8
    gap = 3
    layer_gap = 10
    tile_gap = 28
    layer_size = prototypes.shape[2] * cell
    tile_width = prototypes.shape[0] * 0 + prototypes.shape[1] * (layer_size + gap) + layer_gap
    tile_height = prototypes.shape[2] * cell + 28
    sheet_width = tile_width
    sheet_height = prototypes.shape[0] * (tile_height + tile_gap) + 20

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{sheet_width}" height="{sheet_height}" viewBox="0 0 {sheet_width} {sheet_height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:monospace;font-size:11px;fill:#111827}.air{stroke:#e5e7eb;stroke-width:1}.block{stroke:#111827;stroke-opacity:.18;stroke-width:.5}</style>',
    ]

    for tile_index, patch in enumerate(prototypes):
        base_y = 20 + tile_index * (tile_height + tile_gap)
        parts.append(f'<text x="0" y="{base_y - 5}">tile {tile_index:03d}</text>')
        for y in range(patch.shape[0]):
            base_x = y * (layer_size + gap)
            parts.append(f'<text x="{base_x}" y="{base_y + layer_size + 15}">y={y}</text>')
            for z in range(patch.shape[1]):
                for x in range(patch.shape[2]):
                    category = palette[int(patch[y, z, x])]
                    css_class = "air" if category == "air" else "block"
                    fill = color_for_category(category)
                    opacity = "0.28" if category == "air" else "1"
                    parts.append(
                        f'<rect class="{css_class}" x="{base_x + x * cell}" y="{base_y + z * cell}" '
                        f'width="{cell}" height="{cell}" fill="{fill}" opacity="{opacity}"><title>{category}</title></rect>'
                    )
    parts.append("</svg>")
    (preview_dir / "prototypes.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_sources_csv(path: Path, sources: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["array", "source", "transform", "y", "z", "x", "center_y_fraction", "semantic_bucket"],
        )
        writer.writeheader()
        writer.writerows(sources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", type=Path, default=Path("datasets/phase1"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/phase2"))
    parser.add_argument("--docs-path", type=Path, default=Path("docs/phase2_evaluation.md"))
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--max-air-fraction", type=float, default=0.70)
    parser.add_argument("--max-non-air-fraction", type=float, default=0.75)
    parser.add_argument("--min-architectural-score", type=float, default=1.05)
    parser.add_argument("--max-simple-kind-fraction", type=float, default=0.94)
    parser.add_argument("--max-patches", type=int, default=60000)
    parser.add_argument("--k-values", default="30,40,50")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--silhouette-samples", type=int, default=3000)
    parser.add_argument("--svd-components", type=int, default=64)
    parser.add_argument("--semantic-feature-weight", type=float, default=4.0)
    parser.add_argument("--no-balanced-by-source", action="store_true")
    parser.add_argument("--no-balanced-by-bucket", action="store_true")
    parser.add_argument("--skip-heldout-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    metadata, arrays = load_phase1(args.phase1_dir / "metadata.json")
    palette = metadata["palette"]
    patches, sources, patch_stats = extract_patch_sample(
        arrays=arrays,
        palette=palette,
        window=args.window,
        air_id=int(metadata["air_id"]),
        max_air_fraction=args.max_air_fraction,
        max_non_air_fraction=args.max_non_air_fraction,
        min_architectural_score=args.min_architectural_score,
        max_simple_kind_fraction=args.max_simple_kind_fraction,
        max_patches=args.max_patches,
        rng=rng,
        balanced_by_source=not args.no_balanced_by_source,
        balanced_by_bucket=not args.no_balanced_by_bucket,
    )
    features, feature_info, _onehot = features_for_patches(
        patches,
        sources,
        len(palette),
        palette,
        int(metadata["air_id"]),
        args.svd_components,
        args.semantic_feature_weight,
        args.seed,
    )
    k_values = [int(value.strip()) for value in args.k_values.split(",") if value.strip()]
    evaluation = evaluate_k(features, k_values, args.seed, args.silhouette_samples)
    chosen_k = choose_k(evaluation)
    heldout_evaluation = []
    if not args.skip_heldout_eval:
        heldout_evaluation = embedded_holdout_evaluation(features, sources, chosen_k, args.seed)

    model = MiniBatchKMeans(
        n_clusters=chosen_k,
        random_state=args.seed,
        batch_size=min(4096, max(256, features.shape[0])),
        n_init=16,
    )
    labels = model.fit_predict(features)
    prototypes, cluster_summaries = select_medoids(
        patches,
        features,
        labels,
        model.cluster_centers_,
        args.min_cluster_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "sampled_patches.npy", patches)
    np.savez_compressed(
        args.output_dir / "tile_library.npz",
        prototypes=prototypes,
        labels=labels.astype(np.int32),
        centers=model.cluster_centers_.astype(np.float32),
    )
    write_sources_csv(args.output_dir / "sampled_patch_sources.csv", sources)
    write_previews(args.output_dir, prototypes, palette)
    write_svg_previews(args.output_dir, prototypes, palette)

    report = {
        "window": args.window,
        "max_air_fraction": args.max_air_fraction,
        "max_non_air_fraction": args.max_non_air_fraction,
        "min_architectural_score": args.min_architectural_score,
        "max_simple_kind_fraction": args.max_simple_kind_fraction,
        "max_patches": args.max_patches,
        "seed": args.seed,
        "feature_info": feature_info,
        "patch_stats": patch_stats,
        "k_evaluation": evaluation,
        "chosen_k": int(chosen_k),
        "final_inertia": float(model.inertia_),
        "cluster_count": int(len(set(labels))),
        "prototype_count_after_pruning": int(len(prototypes)),
        "min_cluster_size": args.min_cluster_size,
        "cluster_summaries": cluster_summaries,
        "heldout_evaluation": heldout_evaluation,
    }
    write_json(args.output_dir / "evaluation.json", report)

    lines = [
        "# Phase II Evaluation",
        "",
        f"Extracted {args.window}x{args.window}x{args.window} overlapping patches from the Phase I augmented structural arrays, excluding windows with more than {args.max_air_fraction:.0%} air.",
        f"Density candidates found: {patch_stats['density_candidates']}. Eligible architectural patches after filtering: {patch_stats['total_eligible_patches']}. Sampled patches clustered: {patch_stats['sampled_patches']}.",
        f"Rejected dense material chunks: {patch_stats['rejected_dense']}. Rejected low-salience chunks: {patch_stats['rejected_score']}. Rejected low-information single-kind chunks: {patch_stats['rejected_low_information']}.",
        f"Sampling is source-balanced: {not args.no_balanced_by_source}. Actual source sample counts: {patch_stats['actual_source_counts']}.",
        f"Sampling is semantic-bucket-balanced: {not args.no_balanced_by_bucket}. Actual bucket sample counts: {patch_stats['actual_bucket_counts']}.",
        f"Feature encoding: sparse one-hot per voxel -> {feature_info['svd_components']} SVD components plus architectural descriptors and relative height; explained variance ratio sum {feature_info['svd_explained_variance_ratio_sum']:.4f}.",
        "",
        "## K Selection",
        "",
        "| k | inertia | silhouette |",
        "|---:|---:|---:|",
    ]
    for item in evaluation:
        silhouette = "n/a" if item["silhouette"] is None else f"{item['silhouette']:.4f}"
        lines.append(f"| {item['k']} | {item['inertia']:.2f} | {silhouette} |")
    lines.extend(
        [
            "",
            f"Chosen k: {chosen_k}. Prototype tiles retained after pruning clusters smaller than {args.min_cluster_size}: {len(prototypes)}.",
            "",
            "## Held-Out Source Diagnostics",
            "",
            "| Held-out source | Train patches | Held-out patches | Held-out/train distance ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in heldout_evaluation:
        lines.append(
            f"| {item['heldout_source']} | {item['train_patches']} | {item['heldout_patches']} | "
            f"{item['heldout_to_train_distance_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Observations",
            "",
            "- The palette reduction is doing important denoising work: dark masonry variants are visually inconsistent at the block level but structurally equivalent for patch learning, so they are grouped before clustering.",
            "- One-hot/SVD features avoid treating arbitrary palette IDs as ordinal distances, while architectural descriptor features keep windows, openings, roof pieces, frames, and walls visible to clustering.",
            "- Source-balanced patch sampling prevents the largest structure from dominating the training sample. Semantic-bucket balancing prevents common solid material chunks from crowding out rare building features.",
            "- The 70% air filter keeps roof edges, wall faces, and openings while dropping mostly empty context patches. Dense material chunks, low-salience local fragments, and simple single-kind sheets/columns are filtered before clustering.",
            "- A sampled patch cap is used for tractable clustering. The script still counts all eligible windows before balanced sampling, so future runs can raise `--max-patches` when more memory or time is available.",
            "- The selected medoids are actual observed patches, not averaged centroids, so every exported prototype is a valid block arrangement from the source corpus.",
            "",
            "Preview text files and `prototypes.svg` are written to `datasets/phase2/previews`; use `legend.txt` there to decode compact category labels.",
        ]
    )
    args.docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote tile library to {args.output_dir / 'tile_library.npz'}")
    print(f"Wrote evaluation to {args.output_dir / 'evaluation.json'}")
    print(f"Wrote observations to {args.docs_path}")


if __name__ == "__main__":
    main()
