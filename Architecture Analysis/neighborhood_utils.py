"""Neighborhood feature extraction for cell and ROI architecture tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from skimage import io
from sklearn.neighbors import NearestNeighbors


PathLike = Union[str, Path]


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find {label}. Tried columns: {list(candidates)}")


def _as_coordinate_array(df: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray:
    coords = df[[x_col, y_col]].to_numpy(dtype=float)
    if np.isnan(coords).any():
        raise ValueError(f"Coordinate columns {x_col!r} and {y_col!r} contain NaN values.")
    return coords


def orientation_angle(
    df: pd.DataFrame,
    *,
    angle_col: Optional[str] = None,
    x_orientation_col: str = "x_orientation",
    y_orientation_col: str = "y_orientation",
) -> np.ndarray:
    """Return orientation angles in radians from an angle column or x/y vector columns."""
    if angle_col is not None:
        return df[angle_col].to_numpy(dtype=float)

    if x_orientation_col not in df.columns or y_orientation_col not in df.columns:
        raise KeyError(
            "Provide angle_col or orientation vector columns "
            f"{x_orientation_col!r} and {y_orientation_col!r}."
        )

    x_orientation = df[x_orientation_col].to_numpy(dtype=float)
    y_orientation = df[y_orientation_col].to_numpy(dtype=float)
    return np.arctan2(y_orientation, x_orientation)


def _orientation_angles_from_vectors(x_orientation: np.ndarray, y_orientation: np.ndarray) -> np.ndarray:
    """Match the manuscript convention: project orientation vectors onto [0, 1]."""
    vectors = np.column_stack([x_orientation, y_orientation]).astype(float)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 0
    angles = np.full(len(vectors), np.nan, dtype=float)
    unit_vectors = vectors[valid] / norms[valid, None]
    dot_products = np.clip(unit_vectors @ np.array([0.0, 1.0]), -1.0, 1.0)
    angles[valid] = np.arccos(dot_products)
    return angles


def nematic_order_parameter(angles: Sequence[float]) -> float:
    """Compute the 2D nematic order parameter for orientation angles.

    This follows the figure convention: sqrt(mean(cos(2 theta))^2 +
    mean(sin(2 theta))^2). Values near 1 indicate aligned orientations; values
    near 0 indicate random orientations.
    """
    angles = np.asarray(angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return float("nan")

    return float(np.hypot(np.cos(2.0 * angles).mean(), np.sin(2.0 * angles).mean()))


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Return std / mean for the Cov_Distance feature columns."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    mean = float(np.mean(values))
    if mean == 0:
        return float("nan")
    return float(np.std(values) / mean)


def add_k_neighbor_features(
    df: pd.DataFrame,
    *,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    k_values: Iterable[int] = (3,),
    angle_col: Optional[str] = None,
    x_orientation_col: Optional[str] = None,
    y_orientation_col: Optional[str] = None,
    group_cols: Optional[Sequence[str]] = None,
    include_neighbor_distance_columns: bool = True,
    include_legacy_columns: bool = True,
) -> pd.DataFrame:
    """Add k-nearest-neighbor distances and local orientation order parameters.

    Parameters
    ----------
    df:
        Cell or ROI table with centroid coordinates and orientation information.
    k_values:
        Neighborhood sizes to compute, for example ``(3, 5, 10)``.
    group_cols:
        Optional columns used to keep neighborhoods within images, patients, or
        cores. If omitted, all rows are treated as one spatial field.
    """
    result = df.copy()
    x_col = x_col or _first_existing_column(result, ("x_centroid", "Centroid_X", "X"), "x coordinate")
    y_col = y_col or _first_existing_column(result, ("y_centroid", "Centroid_Y", "Y"), "y coordinate")
    x_orientation_col = x_orientation_col or _first_existing_column(
        result, ("x_orientation", "Orientation_X", "orientation_x"), "x orientation"
    )
    y_orientation_col = y_orientation_col or _first_existing_column(
        result, ("y_orientation", "Orientation_Y", "orientation_y"), "y orientation"
    )
    k_values = sorted({int(k) for k in k_values})
    if not k_values or min(k_values) < 1:
        raise ValueError("k_values must contain positive integers.")

    if angle_col is not None:
        angles = orientation_angle(result, angle_col=angle_col)
    else:
        angles = _orientation_angles_from_vectors(
            result[x_orientation_col].to_numpy(dtype=float),
            result[y_orientation_col].to_numpy(dtype=float),
        )
    max_k = max(k_values)

    if group_cols is None:
        groups = [(None, np.arange(len(result)))]
    else:
        missing = [col for col in group_cols if col not in result.columns]
        if missing:
            raise KeyError(f"Missing group columns: {missing}")
        groups = [
            (name, result.index.get_indexer_for(group.index))
            for name, group in result.groupby(list(group_cols), sort=False)
        ]

    for k in k_values:
        result[f"knn{k}_mean_distance"] = np.nan
        result[f"knn{k}_std_distance"] = np.nan
        result[f"knn{k}_cov_distance"] = np.nan
        result[f"knn{k}_max_distance"] = np.nan
        result[f"knn{k}_order_parameter"] = np.nan
        if include_legacy_columns:
            result[f"Mean_Distance_K={k}"] = np.nan
            result[f"Std_Distance_K={k}"] = np.nan
            result[f"Cov_Distance_K={k}"] = np.nan
            result[f"Max_Distance_K={k}"] = np.nan
            result[f"Order_K={k}"] = np.nan
        if include_neighbor_distance_columns:
            for neighbor_rank in range(1, k + 1):
                result[f"knn{k}_distance_{neighbor_rank}"] = np.nan

    result["Closest_Neighbor_Label"] = np.nan
    result["Closest_Distance"] = np.nan

    for _, positions in groups:
        positions = np.asarray(positions, dtype=int)
        positions = positions[positions >= 0]
        if len(positions) <= 1:
            continue

        group_df = result.iloc[positions]
        coords = _as_coordinate_array(group_df, x_col, y_col)
        n_neighbors = min(max_k + 1, len(positions))
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
        nn.fit(coords)
        distances, neighbor_positions = nn.kneighbors(coords)

        # First neighbor is the query point itself, so it is removed.
        distances = distances[:, 1:]
        neighbor_positions = neighbor_positions[:, 1:]

        closest_positions = neighbor_positions[:, 0]
        closest_distances = distances[:, 0]

        for row_position, position in enumerate(positions):
            row_index = result.index[position]
            closest_position = positions[closest_positions[row_position]]
            result.at[row_index, "Closest_Neighbor_Label"] = closest_position
            result.at[row_index, "Closest_Distance"] = float(closest_distances[row_position])
            available = distances.shape[1]
            for k in k_values:
                effective_k = min(k, available)
                if effective_k == 0:
                    continue

                selected_distances = distances[row_position, :effective_k]
                selected_positions = neighbor_positions[row_position, :effective_k]
                selected_angles = angles[positions[selected_positions]]

                result.at[row_index, f"knn{k}_mean_distance"] = float(np.mean(selected_distances))
                result.at[row_index, f"knn{k}_std_distance"] = float(np.std(selected_distances))
                result.at[row_index, f"knn{k}_cov_distance"] = coefficient_of_variation(selected_distances)
                result.at[row_index, f"knn{k}_max_distance"] = float(np.max(selected_distances))
                result.at[row_index, f"knn{k}_order_parameter"] = nematic_order_parameter(selected_angles)
                if include_legacy_columns:
                    result.at[row_index, f"Mean_Distance_K={k}"] = float(np.mean(selected_distances))
                    result.at[row_index, f"Std_Distance_K={k}"] = float(np.std(selected_distances))
                    result.at[row_index, f"Cov_Distance_K={k}"] = coefficient_of_variation(selected_distances)
                    result.at[row_index, f"Max_Distance_K={k}"] = float(np.max(selected_distances))
                    result.at[row_index, f"Order_K={k}"] = nematic_order_parameter(selected_angles)

                if include_neighbor_distance_columns:
                    for neighbor_rank, distance in enumerate(selected_distances, start=1):
                        result.at[row_index, f"knn{k}_distance_{neighbor_rank}"] = float(distance)

    return result


def add_distance_to_cell_population_features(
    df: pd.DataFrame,
    target_points: np.ndarray,
    *,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    feature_name: str = "Dist_To_Stroma",
) -> pd.DataFrame:
    """Add minimum centroid distance to a target cell population."""
    result = df.copy()
    if len(result) == 0:
        result[feature_name] = []
        return result

    x_col = x_col or _first_existing_column(result, ("x_centroid", "Centroid_X", "X"), "x coordinate")
    y_col = y_col or _first_existing_column(result, ("y_centroid", "Centroid_Y", "Y"), "y coordinate")
    target_points = np.asarray(target_points, dtype=float)

    if target_points.size == 0:
        result[feature_name] = np.nan
        return result

    centroids = result[[x_col, y_col]].to_numpy(dtype=float)
    result[feature_name] = cdist(centroids, target_points).min(axis=1)
    return result


def add_tumor_stroma_distance_features(
    df: pd.DataFrame,
    *,
    tissue_col: str = "Tissue_Type",
    tumor_label: str = "Tumor",
    stroma_label: str = "Stroma",
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    tumor_feature_name: str = "Dist_To_Stroma",
    stroma_feature_name: str = "Dist_To_Tumor",
) -> pd.DataFrame:
    """Add tumor-to-stroma and stroma-to-tumor centroid proximity features."""
    result = df.copy()
    x_col = x_col or _first_existing_column(result, ("x_centroid", "Centroid_X", "X"), "x coordinate")
    y_col = y_col or _first_existing_column(result, ("y_centroid", "Centroid_Y", "Y"), "y coordinate")

    tumor_mask = result[tissue_col] == tumor_label
    stroma_mask = result[tissue_col] == stroma_label
    tumor_points = result.loc[tumor_mask, [x_col, y_col]].to_numpy(dtype=float)
    stroma_points = result.loc[stroma_mask, [x_col, y_col]].to_numpy(dtype=float)

    result[tumor_feature_name] = np.nan
    result[stroma_feature_name] = np.nan
    if len(stroma_points) > 0 and tumor_mask.any():
        result.loc[tumor_mask, tumor_feature_name] = cdist(
            result.loc[tumor_mask, [x_col, y_col]].to_numpy(dtype=float), stroma_points
        ).min(axis=1)
    if len(tumor_points) > 0 and stroma_mask.any():
        result.loc[stroma_mask, stroma_feature_name] = cdist(
            result.loc[stroma_mask, [x_col, y_col]].to_numpy(dtype=float), tumor_points
        ).min(axis=1)

    return result


def fragmentation_index_from_centroids(
    df: pd.DataFrame,
    *,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    radius: float = 80.0,
) -> float:
    """Compute corrected fragmentation index: Louvain communities / cell count."""
    if len(df) == 0:
        return float("nan")

    x_col = x_col or _first_existing_column(df, ("x_centroid", "Centroid_X", "X"), "x coordinate")
    y_col = y_col or _first_existing_column(df, ("y_centroid", "Centroid_Y", "Y"), "y coordinate")
    coords = _as_coordinate_array(df, x_col, y_col)
    if len(coords) == 1:
        return 1.0

    try:
        import networkx as nx
        import community as community_louvain
    except ImportError as exc:
        raise ImportError(
            "fragmentation_index_from_centroids requires networkx and python-louvain."
        ) from exc

    nn = NearestNeighbors(radius=radius, algorithm="ball_tree")
    nn.fit(coords)
    adjacency = nn.radius_neighbors_graph(coords, mode="connectivity")
    graph = nx.from_scipy_sparse_array(adjacency)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if graph.number_of_edges() == 0:
        community_count = len(coords)
    else:
        partition = community_louvain.best_partition(graph)
        community_count = len(set(partition.values()))

    return float(community_count / len(coords))


def add_distance_to_mask_features(
    df: pd.DataFrame,
    mask_path: PathLike,
    *,
    x_col: str = "x_centroid",
    y_col: str = "y_centroid",
    pixel_size: float = 1.0,
    feature_name: str = "distance_to_stroma",
    positive_value: Optional[Union[int, float]] = None,
) -> pd.DataFrame:
    """Add distance from each centroid to the nearest positive pixel in a mask.

    For a stroma mask, positive pixels should correspond to stroma. Distances
    are zero for centroids already inside the positive mask.
    """
    result = df.copy()
    mask = io.imread(mask_path)
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask[..., 0]

    positive_mask = mask == positive_value if positive_value is not None else mask > 0
    distance_map = ndi.distance_transform_edt(~positive_mask) * pixel_size

    x = np.rint(result[x_col].to_numpy(dtype=float)).astype(int)
    y = np.rint(result[y_col].to_numpy(dtype=float)).astype(int)
    x = np.clip(x, 0, distance_map.shape[1] - 1)
    y = np.clip(y, 0, distance_map.shape[0] - 1)

    result[feature_name] = distance_map[y, x]
    return result


def summarize_neighborhood_features(
    df: pd.DataFrame,
    *,
    group_cols: Optional[Sequence[str]] = None,
    feature_prefixes: Sequence[str] = (
        "knn",
        "distance_to_",
        "Dist_To_",
        "Mean_Distance_K=",
        "Std_Distance_K=",
        "Cov_Distance_K=",
        "Max_Distance_K=",
        "Order_K=",
        "Closest_Distance",
    ),
) -> pd.DataFrame:
    """Summarize neighborhood features at image/core/patient level."""
    feature_cols = [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in feature_prefixes) and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        return pd.DataFrame()

    def _summary(frame: pd.DataFrame) -> pd.Series:
        values = {}
        for col in feature_cols:
            series = frame[col].dropna()
            mean = series.mean()
            std = series.std()
            values[f"{col}_mean"] = mean
            values[f"{col}_median"] = series.median()
            values[f"{col}_std"] = std
            values[f"{col}_cov"] = std / mean if pd.notna(mean) and mean != 0 else np.nan
        return pd.Series(values)

    if group_cols is None:
        return _summary(df).to_frame().T

    return df.groupby(list(group_cols), sort=False).apply(_summary).reset_index()
