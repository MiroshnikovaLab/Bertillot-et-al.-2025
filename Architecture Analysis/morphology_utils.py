"""Morphological feature extraction helpers for tumor architecture masks.

The functions in this module operate on binary masks and return plain Python
dictionaries so they can be used directly from notebooks and saved as CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from skimage import io, measure, morphology, segmentation


PathLike = Union[str, Path]
Number = Union[int, float]
FeatureValue = Optional[Union[float, int, str]]


def load_binary_mask(path: PathLike, positive_value: Optional[Number] = None) -> np.ndarray:
    """Load an image mask and return a 2D boolean array.

    Parameters
    ----------
    path:
        Path to a mask image.
    positive_value:
        If supplied, only pixels equal to this value are considered positive.
        If omitted, all non-zero pixels are considered positive.
    """
    mask = io.imread(path)
    mask = np.asarray(mask)

    if mask.ndim > 2:
        mask = mask[..., 0]

    if positive_value is None:
        return mask > 0

    return mask == positive_value


def remove_small_objects(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    """Remove connected components smaller than ``min_area`` pixels."""
    mask = np.asarray(mask, dtype=bool)
    if min_area <= 0:
        return mask
    return morphology.remove_small_objects(mask, min_size=min_area)


def preprocess_tumor_mask(
    mask: np.ndarray,
    *,
    min_object_area: int = 20,
    min_hole_area: int = 100,
) -> np.ndarray:
    """Match manuscript preprocessing for architecture morphology masks."""
    mask = remove_small_objects(mask, min_object_area)
    if min_hole_area > 0:
        mask = morphology.remove_small_holes(mask, area_threshold=min_hole_area)
    return np.asarray(mask, dtype=bool)


def lumen_mask_from_tumor_mask(tumor_mask: np.ndarray, min_lumen_area: int = 0) -> np.ndarray:
    """Detect lumens as holes enclosed by the binary tumor mask."""
    tumor_mask = np.asarray(tumor_mask, dtype=bool)
    lumen = np.logical_and(ndi.binary_fill_holes(tumor_mask), ~tumor_mask)
    return remove_small_objects(lumen, min_lumen_area)


def skeleton_length(skeleton: np.ndarray, pixel_size: float = 1.0) -> float:
    """Estimate skeleton length using orthogonal and diagonal pixel links."""
    skel = np.asarray(skeleton, dtype=bool)
    if skel.size == 0 or not skel.any():
        return 0.0

    right = np.logical_and(skel[:, :-1], skel[:, 1:]).sum()
    down = np.logical_and(skel[:-1, :], skel[1:, :]).sum()
    diag_down_right = np.logical_and(skel[:-1, :-1], skel[1:, 1:]).sum()
    diag_down_left = np.logical_and(skel[:-1, 1:], skel[1:, :-1]).sum()

    return float((right + down + np.sqrt(2.0) * (diag_down_right + diag_down_left)) * pixel_size)


def skeleton_branch_lengths(
    skeleton: np.ndarray,
    *,
    pixel_size: float = 1.0,
    min_branch_length: float = 5.0,
) -> np.ndarray:
    """Measure geodesic skeleton branch lengths with ``skan`` when available."""
    skeleton = np.asarray(skeleton, dtype=bool)
    if not skeleton.any():
        return np.array([], dtype=float)

    try:
        from skan import Skeleton, summarize
    except ImportError:
        total = skeleton_length(skeleton, pixel_size=pixel_size)
        return np.array([total], dtype=float) if total >= min_branch_length else np.array([], dtype=float)

    summary = summarize(Skeleton(skeleton, spacing=pixel_size))
    branch_lengths = summary["branch-distance"].to_numpy(dtype=float)
    return branch_lengths[branch_lengths >= min_branch_length]


def skeleton_node_counts(skeleton: np.ndarray) -> dict[str, int]:
    """Count endpoints and branchpoints in a skeletonized binary mask."""
    skel = np.asarray(skeleton, dtype=bool)
    if skel.size == 0 or not skel.any():
        return {"skeleton_endpoints": 0, "skeleton_branchpoints": 0}

    padded = np.pad(skel.astype(np.uint8), 1, mode="constant")
    neighbor_count = np.zeros_like(skel, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor_count += padded[1 + dy : 1 + dy + skel.shape[0], 1 + dx : 1 + dx + skel.shape[1]]

    return {
        "skeleton_endpoints": int(np.logical_and(skel, neighbor_count == 1).sum()),
        "skeleton_branchpoints": int(np.logical_and(skel, neighbor_count >= 3).sum()),
    }


def mean_abs_boundary_curvature(
    mask: np.ndarray,
    pixel_size: float = 1.0,
    smoothing_sigma: float = 1.0,
) -> float:
    """Approximate mean absolute curvature across all mask contours."""
    mask = np.asarray(mask, dtype=bool)
    contours = measure.find_contours(mask.astype(float), level=0.5)
    if not contours:
        return float("nan")

    curvature_sum = 0.0
    contour_point_count = 0

    for contour in contours:
        if len(contour) < 5:
            continue

        xy = contour[:, ::-1].astype(float) * pixel_size
        if smoothing_sigma > 0:
            xy[:, 0] = gaussian_filter1d(xy[:, 0], sigma=smoothing_sigma, mode="wrap")
            xy[:, 1] = gaussian_filter1d(xy[:, 1], sigma=smoothing_sigma, mode="wrap")

        dx = np.gradient(xy[:, 0])
        dy = np.gradient(xy[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = np.power(dx * dx + dy * dy, 1.5)

        with np.errstate(divide="ignore", invalid="ignore"):
            curvature = np.abs(dx * ddy - dy * ddx) / denom

        curvature = curvature[np.isfinite(curvature)]
        if curvature.size == 0:
            continue

        curvature_sum += float(np.abs(curvature).sum())
        contour_point_count += int(curvature.size)

    if contour_point_count == 0:
        return float("nan")

    return curvature_sum / contour_point_count


def region_summary(mask: np.ndarray, pixel_size: float = 1.0) -> dict[str, float]:
    """Summarize connected-component morphology for a binary mask."""
    mask = np.asarray(mask, dtype=bool)
    labels = measure.label(mask)
    props = measure.regionprops(labels)

    area_px = int(mask.sum())
    if area_px == 0:
        return {
            "area_px": 0,
            "area": 0.0,
            "component_count": 0,
            "largest_component_area_px": 0,
            "largest_component_fraction": float("nan"),
            "mean_component_area_px": float("nan"),
            "median_component_area_px": float("nan"),
            "mean_solidity": float("nan"),
            "mean_eccentricity": float("nan"),
        }

    areas = np.array([p.area for p in props], dtype=float)
    solidities = np.array([p.solidity for p in props], dtype=float)
    eccentricities = np.array([p.eccentricity for p in props], dtype=float)

    return {
        "area_px": area_px,
        "area": float(area_px * pixel_size**2),
        "component_count": int(len(props)),
        "largest_component_area_px": int(areas.max()),
        "largest_component_fraction": float(areas.max() / area_px),
        "mean_component_area_px": float(areas.mean()),
        "median_component_area_px": float(np.median(areas)),
        "mean_solidity": float(np.nanmean(solidities)),
        "mean_eccentricity": float(np.nanmean(eccentricities)),
    }


def measure_architecture_morphology(
    tumor_mask: np.ndarray,
    *,
    stroma_mask: Optional[np.ndarray] = None,
    lumen_mask: Optional[np.ndarray] = None,
    image_id: Optional[str] = None,
    pixel_size: float = 1.0,
    min_tumor_area: int = 20,
    min_hole_area: int = 100,
    min_lumen_area: int = 0,
    min_branch_length: float = 5.0,
) -> dict[str, FeatureValue]:
    """Measure architecture morphology from tumor, optional stroma, and lumen masks."""
    tumor = preprocess_tumor_mask(tumor_mask, min_object_area=min_tumor_area, min_hole_area=min_hole_area)
    lumen = remove_small_objects(lumen_mask, min_lumen_area) if lumen_mask is not None else lumen_mask_from_tumor_mask(tumor)
    stroma = np.asarray(stroma_mask, dtype=bool) if stroma_mask is not None else None

    features: dict[str, FeatureValue] = {"image_id": image_id}
    tumor_summary = region_summary(tumor, pixel_size=pixel_size)
    features.update({f"tumor_{key}": value for key, value in tumor_summary.items()})

    tumor_area_px = tumor_summary["area_px"]
    tissue_area_px = int(tumor.size)
    features["tumor_area_fraction"] = float(tumor_area_px / tissue_area_px) if tissue_area_px else float("nan")
    features["connected_component_density_per_1e6_px"] = (
        float(tumor_summary["component_count"] / tumor_area_px * 1_000_000) if tumor_area_px else float("nan")
    )

    perimeter_px = measure.perimeter_crofton(tumor, directions=4)
    features["tumor_perimeter"] = float(perimeter_px * pixel_size)
    features["tumor_roughness_circularity_inverse"] = (
        float(perimeter_px / (2 * np.sqrt(np.pi * tumor_area_px))) if tumor_area_px else float("nan")
    )
    features["tumor_mean_abs_curvature"] = mean_abs_boundary_curvature(tumor, pixel_size=pixel_size)

    skeleton = morphology.skeletonize(tumor)
    branch_lengths = skeleton_branch_lengths(
        skeleton,
        pixel_size=pixel_size,
        min_branch_length=min_branch_length * pixel_size,
    )
    features["skeleton_length"] = float(branch_lengths.sum()) if branch_lengths.size else 0.0
    features["skeleton_branch_count"] = int(branch_lengths.size)
    features["skeleton_mean_branch_length"] = float(branch_lengths.mean()) if branch_lengths.size else float("nan")
    features["skeleton_median_branch_length"] = float(np.median(branch_lengths)) if branch_lengths.size else float("nan")
    features["skeleton_max_branch_length"] = float(branch_lengths.max()) if branch_lengths.size else float("nan")
    features.update(skeleton_node_counts(skeleton))

    if stroma is not None:
        stroma_area_px = int(stroma.sum())
        features["stroma_area_px"] = stroma_area_px
        features["stroma_area"] = float(stroma_area_px * pixel_size**2)
        features["tumor_stroma_ratio"] = float(tumor_area_px / stroma_area_px) if stroma_area_px else float("nan")
        interface = segmentation.find_boundaries(tumor, mode="outer") & stroma
        features["tumor_stroma_interface"] = float(interface.sum() * pixel_size)

    lumen_summary = region_summary(lumen, pixel_size=pixel_size)
    features.update({f"lumen_{key}": value for key, value in lumen_summary.items()})
    lumen_area_px = lumen_summary["area_px"]
    features["lumen_area_fraction_of_tumor"] = float(lumen_area_px / tumor_area_px) if tumor_area_px else float("nan")

    return features


def batch_measure_architecture_morphology(
    tumor_mask_paths: Iterable[PathLike],
    *,
    stroma_mask_dir: Optional[PathLike] = None,
    lumen_mask_dir: Optional[PathLike] = None,
    pixel_size: float = 1.0,
    min_tumor_area: int = 20,
    min_hole_area: int = 100,
    min_lumen_area: int = 0,
    min_branch_length: float = 5.0,
    mask_suffix: str = "",
) -> pd.DataFrame:
    """Measure morphology for a folder/list of tumor masks.

    Optional stroma and lumen masks are matched by the tumor mask stem plus
    ``mask_suffix``. For example, ``sample_mask.png`` matches
    ``sample_mask.png`` when ``mask_suffix=''``.
    """
    rows = []
    stroma_dir = Path(stroma_mask_dir) if stroma_mask_dir else None
    lumen_dir = Path(lumen_mask_dir) if lumen_mask_dir else None

    for tumor_path in tumor_mask_paths:
        tumor_path = Path(tumor_path)
        image_id = tumor_path.stem.replace(mask_suffix, "") if mask_suffix else tumor_path.stem
        tumor_mask = load_binary_mask(tumor_path)

        stroma_mask = None
        if stroma_dir is not None:
            stroma_path = stroma_dir / tumor_path.name
            stroma_mask = load_binary_mask(stroma_path) if stroma_path.exists() else None

        lumen_mask = None
        if lumen_dir is not None:
            lumen_path = lumen_dir / tumor_path.name
            lumen_mask = load_binary_mask(lumen_path) if lumen_path.exists() else None

        rows.append(
            measure_architecture_morphology(
                tumor_mask,
                stroma_mask=stroma_mask,
                lumen_mask=lumen_mask,
                image_id=image_id,
                pixel_size=pixel_size,
                min_tumor_area=min_tumor_area,
                min_hole_area=min_hole_area,
                min_lumen_area=min_lumen_area,
                min_branch_length=min_branch_length,
            )
        )

    return pd.DataFrame(rows)
