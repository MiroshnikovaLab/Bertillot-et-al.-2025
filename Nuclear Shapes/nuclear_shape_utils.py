"""Reusable nuclear shape feature extraction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import imageio.v3 as imageio
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

try:
    import spatial_efd
except ImportError:  # pragma: no cover - optional dependency error is raised when used.
    spatial_efd = None


PathLike = str | Path


def compute_polygon_area(vertices: List[Tuple[float, float]]) -> float:
    vertices_array = np.asarray(vertices, dtype=float)
    if len(vertices_array) < 3:
        return 0.0
    x = vertices_array[:, 0]
    y = vertices_array[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def compute_cell_orientation(coordinates: np.ndarray) -> Tuple[float, float]:
    """Compute orientation vector from object coordinates using PCA."""
    coordinates = np.asarray(coordinates, dtype=float)
    if len(coordinates) < 2:
        return float("nan"), float("nan")
    centered = coordinates - np.mean(coordinates, axis=0)
    covariance = np.dot(centered.T.copy(), centered)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    orientation_vector = eigenvectors[:, np.argmin(eigenvalues)]
    return float(orientation_vector[0]), float(orientation_vector[1])


def contour_to_polygon(contour: np.ndarray) -> list[Tuple[float, float]]:
    contour = np.squeeze(contour)
    if contour.ndim != 2:
        return []
    return [(float(point[0]), float(point[1])) for point in contour]


def calculate_polygon_perimeter(vertices: List[Tuple[float, float]]) -> float:
    vertices_array = np.asarray(vertices, dtype=float)
    if len(vertices_array) < 2:
        return 0.0
    diffs = np.diff(np.vstack([vertices_array, vertices_array[0]]), axis=0)
    return float(np.sqrt((diffs**2).sum(axis=1)).sum())


def calculate_polygon_axes(vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    points = np.asarray(vertices, dtype=float)
    if len(points) < 3:
        return float("nan"), float("nan")
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.sort(np.maximum(eigenvalues, 0))
    minor_axis = 4.0 * np.sqrt(eigenvalues[0])
    major_axis = 4.0 * np.sqrt(eigenvalues[-1])
    return float(major_axis), float(minor_axis)


def compute_efc_ratio(vertices: List[Tuple[float, float]], harmonics: int = 15) -> float:
    if spatial_efd is None:
        return float("nan")
    x_polygon = [point[0] for point in vertices]
    y_polygon = [point[1] for point in vertices]
    coeffs = spatial_efd.CalculateEFD(x_polygon, y_polygon, harmonics)
    coeffs, _ = spatial_efd.normalize_efd(coeffs, size_invariant=True)
    minor = []
    major = []
    for i in range(len(coeffs)):
        minor.append((coeffs[i][0] ** 2 + coeffs[i][2] ** 2) ** 0.5)
        major.append((coeffs[i][1] ** 2 + coeffs[i][3] ** 2) ** 0.5)
    denominator = sum(minor[1:]) + sum(major[1:])
    return float((minor[0] + major[0]) / denominator) if denominator else float("nan")


def compute_shape_features(vertices: List[Tuple[float, float]], harmonics: int = 15) -> dict[str, float]:
    area = compute_polygon_area(vertices)
    perimeter = calculate_polygon_perimeter(vertices)
    major_axis, minor_axis = calculate_polygon_axes(vertices)
    points = np.asarray(vertices, dtype=float)
    try:
        convex_area = float(ConvexHull(points).volume) if len(points) >= 3 else area
    except Exception:
        convex_area = area

    return {
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "perimeter": perimeter,
        "convex_area": convex_area,
        "area": area,
        "circularity": float(4 * np.pi * area / perimeter**2) if perimeter > 0 else float("nan"),
        "roundness": float(4 * area / (np.pi * major_axis**2)) if major_axis and major_axis > 0 else float("nan"),
        "solidity": float(area / convex_area) if convex_area > 0 else float("nan"),
        "aspect_ratio": float(major_axis / minor_axis) if minor_axis and minor_axis > 0 else float("nan"),
        "compactness": float(perimeter**2 / (4 * np.pi * area)) if area > 0 else float("nan"),
        "efc_ratio": compute_efc_ratio(vertices, harmonics=harmonics),
    }


def compute_features_from_labeled_mask(mask_path: PathLike, harmonics: int = 15) -> pd.DataFrame:
    """Compute nuclear shape features from a labeled mask image."""
    mask_path = Path(mask_path)
    mask = imageio.imread(mask_path)
    if mask.ndim > 2:
        mask = mask[..., 0]

    rows = []
    labels = [label for label in np.unique(mask) if label != 0]
    for label in labels:
        binary = (mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        vertices = contour_to_polygon(contour)
        if len(vertices) < 3:
            continue

        coordinates = np.column_stack(np.where(mask == label))
        y_centroid, x_centroid = coordinates.mean(axis=0)
        x_orientation, y_orientation = compute_cell_orientation(coordinates)
        features = compute_shape_features(vertices, harmonics=harmonics)
        features.update(
            {
                "image_id": mask_path.stem,
                "cell_id": int(label),
                "label": int(label),
                "x_centroid": float(x_centroid),
                "y_centroid": float(y_centroid),
                "x_orientation": x_orientation,
                "y_orientation": y_orientation,
            }
        )
        rows.append(features)

    return pd.DataFrame(rows)


def batch_compute_nuclear_shapes(
    mask_paths: Iterable[PathLike],
    output_dir: PathLike,
    *,
    harmonics: int = 15,
    overwrite: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in mask_paths:
        mask_path = Path(mask_path)
        output_path = output_dir / f"features_image={mask_path.stem}.h5"
        if output_path.exists() and not overwrite:
            continue
        features = compute_features_from_labeled_mask(mask_path, harmonics=harmonics)
        features.to_hdf(output_path, key="df", mode="w")
