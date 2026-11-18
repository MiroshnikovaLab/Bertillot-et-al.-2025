"""Mask-level tissue architecture features used for the per-core tables of the paper.

All features are computed from a tumour–stroma segmentation mask: branch
lengths, branch types and straightness of the pruned tumour skeleton, skeleton
cycles, lumens (holes enclosed by the tumour mask), global contour roughness,
and the tumour–stroma pixel ratio (``tumor_area``).
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from scipy.stats import skew
from skan import Skeleton, summarize
from skimage import measure
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize

PathLike = Union[str, Path]


def preprocess_tumor_mask(tumor_mask: np.ndarray, min_object_area: int = 20, min_hole_area: int = 100) -> np.ndarray:
    """Remove objects smaller than ``min_object_area`` px and fill holes smaller than ``min_hole_area`` px."""
    binary_mask = remove_small_objects(np.asarray(tumor_mask, dtype=bool), min_size=min_object_area)
    return remove_small_holes(binary_mask, area_threshold=min_hole_area)


def prune_skeleton(skeleton: np.ndarray, min_branch_length: float = 5):
    """Keep skeleton branches with an arc length of at least ``min_branch_length`` px."""
    skeleton_image = skeleton.astype(np.uint8)
    ske = Skeleton(skeleton_image, source_image=None)
    ske_summary = summarize(ske)
    long_branches = ske_summary[ske_summary["branch-distance"] >= min_branch_length]

    pruned_skeleton = np.zeros_like(skeleton_image, dtype=bool)
    for idx in long_branches.index:
        branch_coords = ske.path_coordinates(idx)
        pruned_skeleton[branch_coords[:, 0], branch_coords[:, 1]] = True

    ske = Skeleton(pruned_skeleton.astype(np.uint8), source_image=None)
    return pruned_skeleton, ske, summarize(ske)


def measure_branch_properties(ske_summary: pd.DataFrame):
    """Branch arc lengths, straightness (end-to-end / arc length) and skan branch types."""
    branch_lengths = ske_summary["branch-distance"].values
    straightness = ske_summary["euclidean-distance"].values / branch_lengths
    straightness[branch_lengths == 0] = np.nan
    return branch_lengths, straightness, ske_summary["branch-type"].values


def detect_lumens(binary_mask: np.ndarray):
    """Count and measure the holes enclosed by the binary tumour mask."""
    holes = binary_fill_holes(binary_mask) & (~binary_mask)
    holes_props = regionprops(label(holes))
    return len(holes_props), [hole.centroid for hole in holes_props], [hole.area for hole in holes_props]


def measure_loops(ske: Skeleton):
    """Number and length of the cycles of the skeleton graph."""
    coords = ske.coordinates
    graph = nx.Graph()
    for path in ske.paths_list():
        path_coords = coords[path]
        for i in range(len(path_coords) - 1):
            graph.add_edge(tuple(path_coords[i]), tuple(path_coords[i + 1]))

    cycles = nx.cycle_basis(graph)
    loop_lengths = []
    for cycle in cycles:
        length = 0
        for i in range(len(cycle)):
            length += np.linalg.norm(np.array(cycle[i]) - np.array(cycle[(i + 1) % len(cycle)]))
        loop_lengths.append(length)
    return len(cycles), loop_lengths


def compute_global_contour_roughness(tumor_mask: np.ndarray) -> float:
    """Mean absolute curvature per contour point of the Gaussian-smoothed (sigma 1) tumour contours."""
    contours = measure.find_contours(np.asarray(tumor_mask, dtype=np.uint8), level=0.5)

    all_curvatures = []
    total_contour_length = 0
    for contour in contours:
        y_coords, x_coords = contour[:, 0], contour[:, 1]
        if len(x_coords) < 5:
            continue
        xs = gaussian_filter1d(x_coords, sigma=1, mode="wrap")
        ys = gaussian_filter1d(y_coords, sigma=1, mode="wrap")
        dx, dy = np.gradient(xs), np.gradient(ys)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8) ** 1.5
        all_curvatures.extend(curvature)
        total_contour_length += len(curvature)

    return np.sum(np.abs(np.array(all_curvatures))) / total_contour_length


def measure_architecture_morphology(
    mask: np.ndarray,
    *,
    tumor_label: int,
    stroma_label: int,
    image_name: Optional[str] = None,
    min_object_area: int = 20,
    min_hole_area: int = 100,
    min_branch_length: float = 5,
) -> dict:
    """Measure the architecture features of one tumour–stroma segmentation mask."""
    tumor_raw = mask == tumor_label
    binary_mask = preprocess_tumor_mask(tumor_raw, min_object_area=min_object_area, min_hole_area=min_hole_area)

    skeleton = skeletonize(binary_mask)
    _, ske, ske_summary = prune_skeleton(skeleton, min_branch_length=min_branch_length)
    branch_lengths, straightness, branch_types = measure_branch_properties(ske_summary)
    branch_counts = Counter(branch_types)
    total = sum(branch_counts.values())
    branch_props = {key: count / total for key, count in branch_counts.items()}

    lumen_count, _, lumen_areas = detect_lumens(binary_mask)
    number_of_cycles, cycle_lengths = measure_loops(ske)

    nb_tumor = np.count_nonzero(tumor_raw)
    nb_stroma = np.count_nonzero(mask == stroma_label)
    tumor_area = nb_tumor / nb_stroma if nb_stroma else float("nan")

    return {
        "image_name": image_name,
        "tumor_area": tumor_area,
        "number_of_cycles": number_of_cycles,
        "cycle_length_mean": np.mean(cycle_lengths),
        "cycle_length_skew": skew(cycle_lengths),
        "cycle_length_coef_variation": np.std(cycle_lengths) / np.mean(cycle_lengths),
        "prop_of_type0_branches": branch_props.get(0, 0),
        "prop_of_type1_branches": branch_props.get(1, 0),
        "prop_of_type2_branches": branch_props.get(2, 0),
        "lumen_count": lumen_count,
        "lumen_area_mean": np.mean(lumen_areas),
        "lumen_area_skew": skew(lumen_areas),
        "lumen_area_coef_variation": np.std(lumen_areas) / np.mean(lumen_areas),
        "lumen_ratio": lumen_count / tumor_area,
        "branch_length_mean": np.mean(branch_lengths),
        "branch_length_skew": skew(branch_lengths),
        "branch_length_coef_variation": np.std(branch_lengths) / np.mean(branch_lengths),
        "straightness_mean": np.mean(straightness),
        "straightness_skew": skew(straightness),
        "straightness_coef_variation": np.std(straightness) / np.mean(straightness),
        "roughness": compute_global_contour_roughness(tumor_raw),
    }


def batch_measure_architecture_morphology(
    mask_paths: Iterable[PathLike],
    *,
    tumor_label: int,
    stroma_label: int,
    mask_suffix: str = "",
    min_object_area: int = 20,
    min_hole_area: int = 100,
    min_branch_length: float = 5,
) -> pd.DataFrame:
    """Measure every mask; ``mask_suffix`` is removed from the file name to give ``image_name``.

    Masks whose skeleton cannot be analysed (for example masks without tumour)
    are skipped with a warning.
    """
    rows = []
    for mask_path in mask_paths:
        mask_path = Path(mask_path)
        image_name = mask_path.stem.removesuffix(mask_suffix) if mask_suffix else mask_path.stem
        mask = np.array(Image.open(mask_path))
        try:
            rows.append(
                measure_architecture_morphology(
                    mask,
                    tumor_label=tumor_label,
                    stroma_label=stroma_label,
                    image_name=image_name,
                    min_object_area=min_object_area,
                    min_hole_area=min_hole_area,
                    min_branch_length=min_branch_length,
                )
            )
        except Exception as exc:
            warnings.warn(f"Skipped {mask_path.name}: {exc}")
    return pd.DataFrame(rows)
