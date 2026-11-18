"""Nuclear shape feature extraction used for the per-cell tables of the paper.

Perimeter and major/minor axes come from PyRadiomics ``shape2D``; area, convex
area and the elliptic Fourier descriptors come from the OpenCV contour of each
nucleus. Every nucleus is measured on a crop of the labeled mask around it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import radiomics.shape2D
import SimpleITK as sitk
import spatial_efd
from PIL import Image
from scipy.sparse import csr_matrix, find
from scipy.spatial import ConvexHull

logging.getLogger("radiomics").setLevel(logging.CRITICAL + 1)

PathLike = Union[str, Path]


def compute_polygon_area(vertices: List[Tuple[int, int]]) -> float:
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def compute_shape_features(
    mask_crop_sitk: sitk.Image,
    polygon_tuple: List[Tuple[int, int]],
    label: int,
    harmonics: int = 25,
    include_efd_coefficients: bool = True,
) -> dict:
    """Compute shape features of one nucleus from its mask crop and contour polygon."""
    # shape2D only reads the mask, so the mask also serves as the image argument.
    radiomics_shape_features = radiomics.shape2D.RadiomicsShape2D(mask_crop_sitk, mask_crop_sitk, label=label)
    perimeter = radiomics_shape_features.getPerimeterFeatureValue()
    major_axis = radiomics_shape_features.getMajorAxisLengthFeatureValue()
    minor_axis = radiomics_shape_features.getMinorAxisLengthFeatureValue()

    area = compute_polygon_area(polygon_tuple)
    convex_area = ConvexHull(np.array(polygon_tuple), incremental=True).volume

    x_polygon = [point[0] for point in polygon_tuple]
    y_polygon = [point[1] for point in polygon_tuple]
    efd_coeffs = spatial_efd.CalculateEFD(x_polygon, y_polygon, harmonics)
    efd_coeffs, _ = spatial_efd.normalize_efd(efd_coeffs, size_invariant=True)
    minor_axes = [(c[0] ** 2 + c[2] ** 2) ** 0.5 for c in efd_coeffs]
    major_axes = [(c[1] ** 2 + c[3] ** 2) ** 0.5 for c in efd_coeffs]
    efc_ratio = (minor_axes[0] + major_axes[0]) / (sum(minor_axes[1:]) + sum(major_axes[1:]))

    features = {
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "perimeter": perimeter,
        "convex_area": convex_area,
        "area": area,
        "circularity": 4 * np.pi * (area / perimeter**2),
        "roundness": 4 * (area / (np.pi * major_axis**2)),
        "solidity": area / convex_area,
        "aspect_ratio": major_axis / minor_axis,
        "compactness": perimeter**2 / (4 * np.pi * area),
        "efc_ratio": efc_ratio,
    }
    if include_efd_coefficients:
        features.update({f"efd_a{i}b{j}": efd_coeffs[i, j] for i in range(harmonics) for j in range(4)})
    return features


def convert_mask_to_polygon(mask: np.ndarray, label: int) -> Tuple[Tuple[int, int], ...]:
    gray = (mask == label).astype(np.uint8) * 255
    contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    longest = np.argmax([len(contour) for contour in contours])
    return tuple((point[0][0], point[0][1]) for point in contours[longest])


def generate_mask_crop(mask: np.ndarray, mask_csr: csr_matrix, label: int, delta: int = 5) -> np.ndarray:
    rows, cols, _ = find(mask_csr == label)
    return mask[rows.min() - delta : rows.max() + delta, cols.min() - delta : cols.max() + delta]


def compute_cell_orientation(mask_csr: csr_matrix, label: int) -> np.ndarray:
    """Orientation vector of a nucleus from the covariance of its pixel (row, column) coordinates."""
    rows, cols, _ = find(mask_csr == label)
    coordinates = np.column_stack([rows, cols]).astype("float64")
    coordinates -= coordinates.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eig(coordinates.T @ coordinates)
    # Taking a row of the eigenvector matrix reproduces the published orientation vectors.
    return eigenvectors[np.argmin(eigenvalues), :]


def remove_border_objects(mask: np.ndarray, border: int = 5) -> np.ndarray:
    """Set to 0 every object with a pixel in the outer ``border`` rows or columns."""
    mask = mask.copy()
    rows = np.r_[0:border, mask.shape[0] - border : mask.shape[0] - 1]
    cols = np.r_[0:border, mask.shape[1] - border : mask.shape[1] - 1]
    border_labels = np.union1d(np.unique(mask[rows, :]), np.unique(mask[:, cols]))
    mask[np.isin(mask, border_labels[border_labels != 0])] = 0
    return mask


def compute_features_from_labeled_mask(
    mask_path: PathLike,
    *,
    harmonics: int = 25,
    delta: int = 5,
    remove_border: bool = False,
    include_efd_coefficients: bool = True,
    mask_suffix: str = "_mask",
) -> pd.DataFrame:
    """Compute shape features for every nucleus of a labeled mask (0 = background).

    Centroids follow the convention of the paper tables: ``x_centroid`` is the
    mean row index and ``y_centroid`` the mean column index of the nucleus.
    Nuclei whose crop or contour cannot be measured (for example nuclei closer
    than ``delta`` pixels to the image border) are skipped, as in the paper.
    """
    mask_path = Path(mask_path)
    image_name = mask_path.stem.removesuffix(mask_suffix) if mask_suffix else mask_path.stem
    mask = np.array(Image.open(mask_path))
    if remove_border:
        mask = remove_border_objects(mask)
    mask_csr = csr_matrix(mask)

    rows = []
    for cell_id in [label for label in np.unique(mask) if label != 0]:
        try:
            mask_crop = generate_mask_crop(mask, mask_csr, cell_id, delta=delta)
            mask_crop_sitk = sitk.GetImageFromArray(mask_crop)
            orientation = compute_cell_orientation(mask_csr, cell_id)
            polygon = convert_mask_to_polygon(mask_crop, cell_id)
            features = compute_shape_features(
                mask_crop_sitk,
                polygon,
                int(cell_id),
                harmonics=harmonics,
                include_efd_coefficients=include_efd_coefficients,
            )
        except Exception:
            continue

        cell_rows, cell_cols, _ = find(mask_csr == cell_id)
        rows.append(
            {
                "image_name": image_name,
                "cell_id": int(cell_id),
                "x_centroid": cell_rows.mean(),
                "y_centroid": cell_cols.mean(),
                "x_orientation": orientation[0],
                "y_orientation": orientation[1],
                **features,
            }
        )

    return pd.DataFrame(rows)


def batch_compute_nuclear_shapes(
    mask_paths: Iterable[PathLike],
    output_dir: PathLike,
    *,
    harmonics: int = 25,
    delta: int = 5,
    remove_border: bool = False,
    include_efd_coefficients: bool = True,
    mask_suffix: str = "_mask",
    overwrite: bool = False,
) -> None:
    """Write one ``features_image=<mask stem>.h5`` table per labeled mask."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in mask_paths:
        mask_path = Path(mask_path)
        output_path = output_dir / f"features_image={mask_path.stem}.h5"
        if output_path.exists() and not overwrite:
            continue
        features = compute_features_from_labeled_mask(
            mask_path,
            harmonics=harmonics,
            delta=delta,
            remove_border=remove_border,
            include_efd_coefficients=include_efd_coefficients,
            mask_suffix=mask_suffix,
        )
        features.to_hdf(output_path, key="df", mode="w")
